#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, build_scaling
from torch import nn
import pytorch3d
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from pytorch3d.transforms import quaternion_to_matrix

# gaussian_model 脚本里的这俩函数在其他脚本里头都没有调用过
def dilate(bin_img, ksize=5):
    pad = (ksize - 1) // 2
    bin_img = torch.nn.functional.pad(bin_img, pad=[pad, pad, pad, pad], mode='reflect')
    out = torch.nn.functional.max_pool2d(bin_img, kernel_size=ksize, stride=1, padding=0)
    return out

def erode(bin_img, ksize=5):
    out = 1 - dilate(1 - bin_img, ksize)
    return out

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance) # 只保留对称矩阵的上三角矩阵部分节省内存
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._knn_f = torch.empty(0)
        self._features_dc = torch.empty(0) # 球谐函数的直流分量
        self._features_rest = torch.empty(0) # 球谐函数的高阶分量
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0) # 3DGS投影到平面后二维高斯的最大半径
        self.max_weight = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0) # 空间点位置梯度的累计值 应该是用来指导致密化和修剪的
        self.xyz_gradient_accum_abs = torch.empty(0)
        self.denom = torch.empty(0)
        self.denom_abs = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0 # 百分比密度用于密度控制
        self.spatial_lr_scale = 0 # 学习率因子——有啥用？
        self.knn_dists = None
        self.knn_idx = None
        self.setup_functions()
        self.use_app = False
        self.ids = torch.empty(0)
        self.knn_r = torch.empty(0)
        self.movable_mask = torch.empty(0)

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._knn_f,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.max_weight,
            self.xyz_gradient_accum,
            self.xyz_gradient_accum_abs,
            self.denom,
            self.denom_abs,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._knn_f,
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        self.max_weight,
        xyz_gradient_accum, 
        xyz_gradient_accum_abs,
        denom,
        denom_abs,
        opt_dict, 
        self.spatial_lr_scale,
        ) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.xyz_gradient_accum_abs = xyz_gradient_accum_abs
        self.denom = denom
        self.denom_abs = denom_abs
        self.optimizer.load_state_dict(opt_dict)

    # 为啥提取/获取变量的时候是获取激活函数激活过的结果呢？
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
        
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    # 相比gaussian-splatting原代码 以下三个函数是PGSR自己新定义的
    # 1.获得每个3DGS点云的最小坐标轴？
    def get_smallest_axis(self, return_idx=False):
        rotation_matrices = self.get_rotation_matrix()
        smallest_axis_idx = self.get_scaling.min(dim=-1)[1][..., None, None].expand(-1, 3, -1)
        smallest_axis = rotation_matrices.gather(2, smallest_axis_idx)
        if return_idx:
            return smallest_axis.squeeze(dim=2), smallest_axis_idx[..., 0, 0]
        return smallest_axis.squeeze(dim=2)
    
    # 2.获得每个3DGS点云自身的世界坐标系下法向量？按照论文法向量方向应该是3DGS最短轴的方向
    # 在特定相机观测下的法向量，如果光心方向同法向量夹角大于90°，那么将法向量反向才是法向量方向
    def get_normal(self, view_cam):
        normal_global = self.get_smallest_axis()
        gaussian_to_cam_global = view_cam.camera_center - self._xyz
        neg_mask = (normal_global * gaussian_to_cam_global).sum(-1) < 0.0
        normal_global[neg_mask] = -normal_global[neg_mask]
        return normal_global
    
    def get_rotation_matrix(self):
        return quaternion_to_matrix(self.get_rotation) # 果然每个3DGS的旋转是以四元数的方式存储的

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation) # 从比例S和旋转方向R中计算得到3D协方差

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1 # 这个函数会将3DGS球谐函数阶数迭代增加，就随着优化训练进行球谐函数表示精度会增加嘛

    # 这个应该是3DGS核心之一之从pcd中初始化3DGS点云
    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, hive_dict: dict, colorid_list: list):
        self.spatial_lr_scale = spatial_lr_scale # spatial_lr_scale好像是点云在空间中分布范围
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda()) # 我从深度中初始化的点云使用的是随机颜色 fused_color这里只存了直流分量
        
        #用于后续添加id，目前没有使用
        colorid_np = np.array(colorid_list).astype(np.float32)
        pcd_colors = np.array(pcd.colors).astype(np.float32)
        pcd_ids = []
        tolerance = 1e-2
        for pcd_color in pcd_colors:
            idx = np.where(np.all(np.isclose(colorid_np, pcd_color, atol=tolerance), axis=1))[0]
            if idx.size > 0:
                instance_id = idx[0]
            else:
                instance_id = 0
            pcd_ids.append(instance_id)
        pcd_ids = torch.tensor(pcd_ids).float().cuda()
        # print(f"len: {pcd_ids.shape[0]} valid ids: {torch.sum(pcd_ids>0).item()}")

        # ids = torch.tensor(np.sum(np.asarray(pcd.colors*255), axis=1)).float().cuda()
        # ids_flat = ids.view(-1)
        # ordered_ids = torch.zeros_like(ids_flat, dtype=torch.long)
        # seen = {}
        # current_id = 1
        # for i in range(len(ids_flat)):
        #     val = ids_flat[i].item()  # 获取标量值
        #     if val not in seen:
        #         seen[val] = current_id
        #         current_id += 1
        #     ordered_ids[i] = seen[val]
        # ordered_ids=ordered_ids.float().cuda()
        # #print("gaussians.ids 的形状:",ordered_ids.shape)

        init_normals= torch.tensor(np.asarray(pcd.normals)).float().cuda() 
        init_normals = init_normals / torch.norm(init_normals, dim=1, keepdim=True)   
       
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda() # features是整个高斯场景表示的完整球谐函数张量
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0 # feature的第2维度总共不也就3维吗 哪来的:3和3:

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        # print(f"Fix GS xyz:{hive_dict['fix_xyz']}, Fix GS normal:{hive_dict['fix_normal']}")
        # 1.计算每个点云最近邻三个点云距离平均值作为改点云位置3DGS球缩放因子的初始化半径
        dist = torch.sqrt(torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001))
        # print(f"new scale {torch.quantile(dist, 0.1)}")
        #scales = torch.log(dist)[...,None].repeat(1, 3) # 会发现这里算的是对数，因为get_scaling的时候还要指数函数激活
        log_dist = torch.log(dist)
        scales = torch.cat([
        torch.log(dist)[..., None],       # x 轴的缩放因子
        torch.log(dist/1.2)[..., None],       # y 轴的缩放因子
        torch.log(dist/10)[..., None]    # z 轴的缩放因子（缩小至 dist/15）
        ], dim=1)

        z_axis = torch.tensor([0.0, 0.0, 1.0], device=init_normals.device).view(1, 3)
        axis = torch.cross(init_normals, z_axis.expand(init_normals.shape[0], 3))
        axis_norm = torch.norm(axis, dim=1, keepdim=True)
        axis = axis / (axis_norm+1e-8)
        cos_theta1 = torch.clamp(torch.sum(init_normals * z_axis.expand(init_normals.shape[0], 3), dim=1), -1.0, 1.0)
        #sin_theta = torch.sqrt(1.0 - cos_theta ** 2)
        theta = torch.acos(cos_theta1)
        sin_theta = torch.sin(theta / 2)
        cos_theta = torch.cos(theta / 2)


        qw = cos_theta
        qx = axis[:, 0] * sin_theta
        qy = axis[:, 1] * sin_theta
        qz = axis[:, 2] * sin_theta
        rots = torch.stack([qw, qx, qy, qz], dim=1)
        print(rots.shape)
        # print(f"If initial rots contain nan values: {torch.isnan(rots).any()}")

        # rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        # rots[:, 0] = 1 # 每个3DGS点云的scales和rots初始化以及opacities初始化

        #opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        opacities = inverse_sigmoid(0.7 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        knn_f = torch.randn((fused_point_cloud.shape[0], 6)).float().cuda() # knn_f到底有啥用 original 3dgs里面都没有这个属性
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(not hive_dict['fix_xyz']))
        self._knn_f = nn.Parameter(knn_f.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True)) # 一个是主要颜色一个是残差？
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(not hive_dict['fix_normal']))
        self._opacity = nn.Parameter(opacities.requires_grad_(not hive_dict['fix_opa']))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda") # 这俩属于是在虚位以待但不知道啥用
        self.max_weight = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.knn_r = nn.Parameter(log_dist.requires_grad_(False))
        # self.ids= nn.Parameter(ordered_ids.unsqueeze(1).requires_grad_(False))
        self.ids = nn.Parameter(pcd_ids.unsqueeze(1).requires_grad_(False))
        # print("gaussians.ids 的形状:",self.ids.shape)

    
    def create_from_pcd_non(self, pcd : BasicPointCloud, spatial_lr_scale : float, hive_dict: dict, colorid_list: list):
        self.spatial_lr_scale = spatial_lr_scale # spatial_lr_scale好像是点云在空间中分布范围
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda()) # 我从深度中初始化的点云使用的是随机颜色 fused_color这里只存了直流分量
        
        #用于后续添加id，目前没有使用
        colorid_np = np.array(colorid_list).astype(np.float32)
        pcd_colors = np.array(pcd.colors).astype(np.float32)
        pcd_ids = []
        tolerance = 1e-2
        for pcd_color in pcd_colors:
            idx = np.where(np.all(np.isclose(colorid_np, pcd_color, atol=tolerance), axis=1))[0]
            if idx.size > 0:
                instance_id = idx[0]
            else:
                instance_id = 0
            pcd_ids.append(instance_id)
        pcd_ids = torch.tensor(pcd_ids).float().cuda()
        
        init_normals= torch.tensor(np.asarray(pcd.normals)).float().cuda() 
        init_normals = init_normals / torch.norm(init_normals, dim=1, keepdim=True)   
       
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda() # features是整个高斯场景表示的完整球谐函数张量
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0 # feature的第2维度总共不也就3维吗 哪来的:3和3:

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        # 1.计算每个点云最近邻三个点云距离平均值作为改点云位置3DGS球缩放因子的初始化半径
        dist = torch.sqrt(torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001))
        # print(f"new scale {torch.quantile(dist, 0.1)}")
        scales = torch.log(dist)[...,None].repeat(1, 3) # 会发现这里算的是对数，因为get_scaling的时候还要指数函数激活
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1 # 每个3DGS点云的scales和rots初始化以及opacities初始化
        print("Initialising the 3dgs without normal and scaling!!!!")
        
        opacities = inverse_sigmoid(0.7 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        knn_f = torch.randn((fused_point_cloud.shape[0], 6)).float().cuda() # knn_f到底有啥用 original 3dgs里面都没有这个属性
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(not hive_dict['fix_xyz']))
        self._knn_f = nn.Parameter(knn_f.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True)) # 一个是主要颜色一个是残差？
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(not hive_dict['fix_normal']))
        self._opacity = nn.Parameter(opacities.requires_grad_(not hive_dict['fix_opa']))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda") # 这俩属于是在虚位以待但不知道啥用
        self.max_weight = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        # self.knn_r = nn.Parameter(log_dist.requires_grad_(False))
        # self.ids= nn.Parameter(ordered_ids.unsqueeze(1).requires_grad_(False))
        self.ids = nn.Parameter(pcd_ids.unsqueeze(1).requires_grad_(False))
        # print("gaussians.ids 的形状:",self.ids.shape)
    
    
    # Borrowed from SAD-GS for gaussian initialization
    def create_from_gs(self, means, colors, covs, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = means.float().cuda()
        fused_color = RGB2SH(colors.float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        U, S, Vt = torch.svd(covs)

        init_scale = 2. #2.
        scales = torch.log(torch.sqrt(S)*init_scale).float().cuda()

        init_opa = 1. #1.
        opacities = inverse_sigmoid(init_opa * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        
        U[:,:,2] = U[:,:,2] * torch.linalg.det(U).unsqueeze(1)
        rots = pytorch3d.transforms.matrix_to_quaternion(U)
        
        knn_f = torch.randn((fused_point_cloud.shape[0], 6)).float().cuda()
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._knn_f = nn.Parameter(knn_f.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.max_weight = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.abs_split_radii2D_threshold = training_args.abs_split_radii2D_threshold
        self.max_abs_split_points = training_args.max_abs_split_points
        self.max_all_points = training_args.max_all_points
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._knn_f], 'lr': 0.01, "name": "knn_f"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15) # xyz中心位置、颜色sh特征、不透明度、缩放变换和旋转属性都是可优化的
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
    
    def clip_grad(self, norm=1.0):
        for group in self.optimizer.param_groups:
            torch.nn.utils.clip_grad_norm_(group["params"][0], norm)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                # 感觉只有位置参数xyz的更新学习率会自适应的变化嗷
                lr = self.xyz_scheduler_args(iteration) # 指数衰减结合延迟衰减使得学习率在训练过程中平滑地从初始值过渡到最终值
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC 3DGS模型所有的参数 就像那张最经典的ply属性图一样
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path, mask=None):
        mkdir_p(os.path.dirname(path))
        #mask = (self.ids == 18.0)
        #print("mask 中 True 的数量:", mask.sum().item())
        # print("gaussians.ids 的形状:", self.ids.shape)
        # tem_opa=torch.zeros((self.ids.shape[0], 1), dtype=torch.float, device="cuda")
        # tem_opa[mask]=1.0
        # self._opacity=torch.log(tem_opa/(1-tem_opa))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        ids=self.ids.detach().cpu().numpy()
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        #mask = mask.squeeze().cpu().numpy().astype(np.bool_)
        #xyz = xyz[mask]
        #normals = normals[mask]
        #f_dc = f_dc[mask]
        #f_rest = f_rest[mask]
        #opacities = opacities[mask]
        #scale = scale[mask]
        #rotation = rotation[mask]

        
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation,ids), axis=1)
        nan_valid = np.isnan(attributes).any(axis=1)
        attributes = attributes[~nan_valid]
        print(f"gaussian attributes = {attributes.shape}")
        #attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        # 如果 dtype_full 的字段数不足，补充额外的字段
        num_columns = attributes.shape[1]
        if len(dtype_full) < num_columns:
            for i in range(len(dtype_full), num_columns):
               dtype_full.append((f'extra_field_{i}', 'f4'))
        elements = np.empty(attributes.shape[0], dtype=dtype_full)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        #print(el.dtype)
        PlyData([el]).write(path)

    def reset_opacity(self):
        # get_opacity的过程默认对实际的_opacity属性进行了激活，因此get之后想获得真正的结果需要反激活  每隔一段时间就要重置不透明度 why？
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        pcd_ids = np.asarray(plydata.elements[0]["extra_field_62"])[..., np.newaxis]
        #pcd_ids = torch.tensor(ids).float().cuda()

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree # 读取已有ply的时候目前的球谐函数阶数就已经是最大球谐函数阶数了
        self.ids = nn.Parameter(torch.tensor(pcd_ids, dtype=torch.float, device="cuda").requires_grad_(False))

    def replace_tensor_to_optimizer(self, tensor, name):
        # 其目的在于动态改变优化器里的参数值——如重置不透明度但是保留之前的优化器状态——即训练进度
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        # 只保留掩码mask指定的那些优化器对应的参数状态
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        # 利用优化器参数剪枝算法对冗余的3DGS点云进行剪枝
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._knn_f = optimizable_tensors["knn_f"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.denom_abs = self.denom_abs[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.max_weight = self.max_weight[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        # 往优化器中添加新的张量
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_knn_f, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        # 对应的就是往3DGS中添加新的高斯点云以致密化
        d = {"xyz": new_xyz,
        "knn_f": new_knn_f,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._knn_f = optimizable_tensors["knn_f"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.max_weight = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, grads_abs, grad_abs_threshold, scene_extent, max_radii2D, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        padded_grads_abs = torch.zeros((n_init_points), device="cuda")
        padded_grads_abs[:grads_abs.shape[0]] = grads_abs.squeeze()
        padded_max_radii2D = torch.zeros((n_init_points), device="cuda")
        padded_max_radii2D[:max_radii2D.shape[0]] = max_radii2D.squeeze()

        # 高斯梯度大于一定阈值并且比例大于一定值的话就将其划分进致密分裂的mask中
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        
        # 这一部分貌似是PGSR自己加的内容？总之就是限定了一个最大高斯数量，就算允许高斯分裂也不能分裂那么多
        if selected_pts_mask.sum() + n_init_points > self.max_all_points:
            limited_num = self.max_all_points - n_init_points
            padded_grad[~selected_pts_mask] = 0
            ratio = limited_num / float(n_init_points)
            threshold = torch.quantile(padded_grad, (1.0-ratio))
            selected_pts_mask = torch.where(padded_grad > threshold, True, False)
            # print(f"split {selected_pts_mask.sum()}, raddi2D {padded_max_radii2D.max()} ,{padded_max_radii2D.median()}")
        else:
            padded_grads_abs[selected_pts_mask] = 0
            mask = (torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent) & (padded_max_radii2D > self.abs_split_radii2D_threshold)
            padded_grads_abs[~mask] = 0
            selected_pts_mask_abs = torch.where(padded_grads_abs >= grad_abs_threshold, True, False)
            limited_num = min(self.max_all_points - n_init_points - selected_pts_mask.sum(), self.max_abs_split_points)
            if selected_pts_mask_abs.sum() > limited_num:
                ratio = limited_num / float(n_init_points)
                threshold = torch.quantile(padded_grads_abs, (1.0-ratio))
                selected_pts_mask_abs = torch.where(padded_grads_abs > threshold, True, False)
            selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_abs)
            # print(f"split {selected_pts_mask.sum()}, abs {selected_pts_mask_abs.sum()}, raddi2D {padded_max_radii2D.max()} ,{padded_max_radii2D.median()}")

        stds = self.get_scaling[selected_pts_mask].repeat(N,1) # 每个点最多split分裂出两个高斯球吧
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_knn_f = self._knn_f[selected_pts_mask].repeat(N,1) # PGSR里为什么总是显式地调用knn_f有什么具体作用

        self.densification_postfix(new_xyz, new_knn_f, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter) # 分裂完之后原本的高斯中间变量都被清楚掉

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        if selected_pts_mask.sum() + n_init_points > self.max_all_points:
            limited_num = self.max_all_points - n_init_points
            grads_tmp = grads.squeeze().clone()
            grads_tmp[~selected_pts_mask] = 0
            ratio = limited_num / float(n_init_points)
            threshold = torch.quantile(grads_tmp, (1.0-ratio))
            selected_pts_mask = torch.where(grads_tmp > threshold, True, False)

        if selected_pts_mask.sum() > 0:
            # print(f"clone {selected_pts_mask.sum()}")
            new_xyz = self._xyz[selected_pts_mask]

            stds = self.get_scaling[selected_pts_mask]
            means =torch.zeros((stds.size(0), 3),device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(self._rotation[selected_pts_mask])
            new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask]
            
            new_features_dc = self._features_dc[selected_pts_mask]
            new_features_rest = self._features_rest[selected_pts_mask]
            new_opacities = self._opacity[selected_pts_mask]
            new_scaling = self._scaling[selected_pts_mask]
            new_rotation = self._rotation[selected_pts_mask]
            new_knn_f = self._knn_f[selected_pts_mask]

            self.densification_postfix(new_xyz, new_knn_f, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, abs_max_grad, min_opacity, extent, max_screen_size, hive_dict):
        grads = self.xyz_gradient_accum / self.denom
        grads_abs = self.xyz_gradient_accum_abs / self.denom_abs
        grads[grads.isnan()] = 0.0
        grads_abs[grads_abs.isnan()] = 0.0
        max_radii2D = self.max_radii2D.clone()

        if not hive_dict['no_densify']:
            self.densify_and_clone(grads, max_grad, extent)
            self.densify_and_split(grads, max_grad, grads_abs, abs_max_grad, extent, max_radii2D)

        prune_mask = (self.get_opacity < min_opacity).squeeze()

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        # print(f"all points {self._xyz.shape[0]}")
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, viewspace_point_tensor_abs, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.xyz_gradient_accum_abs[update_filter] += torch.norm(viewspace_point_tensor_abs.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1 # denom其实就是个用来记录各个点在致密化/修剪过程中梯度更新次数的一个计数器
        self.denom_abs[update_filter] += 1

    def get_points_depth_in_depth_map(self, fov_camera, depth, points_in_camera_space, scale=1, mask_plane=None, mask_depth=None):
        # 使用相机视场、深度图像、相机空间点云坐标 求这些点在相机上投影插值之后的深度值 即通过深度图像获得相机坐标系点云的深度值
        st = max(int(scale/2)-1,0)
        depth_view = depth[None,:,st::scale,st::scale]
        W, H = int(fov_camera.image_width/scale), int(fov_camera.image_height/scale)
        depth_view = depth_view[:H, :W]
        pts_projections = torch.stack(
                        [points_in_camera_space[:,0] * fov_camera.Fx / points_in_camera_space[:,2] + fov_camera.Cx,
                         points_in_camera_space[:,1] * fov_camera.Fy / points_in_camera_space[:,2] + fov_camera.Cy], -1).float()/scale

        mask = (pts_projections[:, 0] > 0) & (pts_projections[:, 0] < W) &\
               (pts_projections[:, 1] > 0) & (pts_projections[:, 1] < H) & (points_in_camera_space[:,2] > 0.1)
        if mask_depth is not None:
            mask = mask & mask_depth
        
        if mask_plane is not None:
            mask_plane = ~(mask_plane==0.0)
            mask_pts = torch.zeros(pts_projections.shape[0], dtype=torch.bool).to(pts_projections.device)
            x_coords = pts_projections[:,0].long()
            y_coords = pts_projections[:,1].long()
            mask_pts[mask] = mask_plane[y_coords[mask], x_coords[mask]]
            mask_pts[~mask] = False
        else:
            mask_pts = None

        pts_projections[..., 0] /= ((W - 1) / 2)
        pts_projections[..., 1] /= ((H - 1) / 2)
        pts_projections -= 1
        pts_projections = pts_projections.view(1, -1, 1, 2)
        map_z = torch.nn.functional.grid_sample(input=depth_view,
                                                grid=pts_projections,
                                                mode='bilinear',
                                                padding_mode='border',
                                                align_corners=True
                                                )[0, :, :, 0]
        return map_z, mask, mask_pts
    
    # 相邻帧重重投影误差计算优化的时候肯定用的是渲染相机图像，只有在监督的时候才用真值RGB-D深度，我之前的LCP-Fusion直接使用观测深度图像进行计算有点草率
    def get_points_from_depth(self, fov_camera, depth, scale=1, mask_plane=None):
        st = int(max(int(scale/2)-1,0))
        depth_view = depth.squeeze()[st::scale,st::scale]
        rays_d = fov_camera.get_rays(scale=scale) # rays_d: (H, W, 3)
        depth_view = depth_view[:rays_d.shape[0], :rays_d.shape[1]]
        if mask_plane is not None:
            mask_plane = ~(mask_plane==0.0)
            pts = pts = (rays_d * depth_view[..., None])[mask_plane].reshape(-1,3)
        else:
            pts = (rays_d * depth_view[..., None]).reshape(-1,3)
        # 所以你这个R和T应该就是世界坐标系变换到相机坐标系的坐标变换
        R = torch.tensor(fov_camera.R).float().cuda()
        T = torch.tensor(fov_camera.T).float().cuda()
        pts = (pts-T)@R.transpose(-1,-2) # 从深度图像获取密集的空间点云点
        return pts
    