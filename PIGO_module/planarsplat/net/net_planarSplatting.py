import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random
import trimesh
from loguru import logger
import open3d as o3d
from diff_rect_rasterization import RectRasterizationSettings, RectRasterizer 
from quaternion_utils._C import q2RCUDA, qMultCUDA

import sys
from pathlib import Path
current_dir = Path(__file__).resolve().parents[1]
sys.path.append(current_dir)

from planarsplat.utils import model_util
from planarsplat.utils.plot_util import plot_rectangle_planes


class PlanarSplat_Network(nn.Module):
    def __init__(self, cfg, plot_dir='',):
        super().__init__()
        # ==================================
        self.rot_delta = None
        self.bg = torch.tensor([0., 0., 0.]).cuda()
        # ================================== get config ======================
        self.plane_cfg = cfg.get_config('plane_model')
        self.pose_cfg = cfg.get_config('pose')
        self.data_cfg = cfg.get_config('dataset')
        self.plot_dir = plot_dir
        # ---------mvsa config
        self.color_path = self.data_cfg.color_path
        self.color_lut = torch.from_numpy(np.load(self.color_path)).float().cuda()
        # ---------debug
        self.grads = None
        self.debug_on = self.plane_cfg.get_bool("debug_on", default=False)
        # data and scene setting
        self.H = self.data_cfg.img_res[0]
        self.W = self.data_cfg.img_res[1]
        self.scene_bounding_sphere = self.data_cfg.get_float('scene_bounding_sphere')
        # training setting
        self.max_total_iters = cfg.get_int('train.max_total_iters', default=5000)
        self.coarse_stage_ite = cfg.get_int('train.coarse_stage_ite')
        self.initialized = False
        # plane setting: plane splatting
        self.RBF_type = self.plane_cfg.get_string('RBF_type')
        self.RBF_weight_change_type = self.plane_cfg.get_string('RBF_weight_change_type')
        # plane setting: plane radii
        self.radii_dir_type = self.plane_cfg.get_string('radii_dir_type')
        logger.info(f"--------------------> radii type = {self.radii_dir_type}")
        self.init_radii = self.plane_cfg.get_float('radii_init')
        self.radii_max_list = self.plane_cfg.radii_max_list # 所有平面的半径必须在0.001-0.5之间，0.5都是50厘米了..
        self.radii_min_list = self.plane_cfg.radii_min_list
        self.radii_milestone_list = self.plane_cfg.radii_milestone_list
        self.min_radii = 0.
        self.max_radii = 1000.

        # ================================== define plane parameters ======================
        # fixed plane parameters
        self.plane_normal_standard = torch.tensor([0., 0., 1.]).reshape(1, 3).cuda()
        self.plane_xAxis_standard = torch.tensor([1., 0., 0.]).reshape(1, 3).cuda()
        self.plane_yAxis_standard = torch.tensor([0., 1., 0.]).reshape(1, 3).cuda()
        self.plane_rot_q_normal_z = torch.Tensor(1, 1).fill_(0).cuda()
        self.plane_rot_q_xyAxis_xy = torch.Tensor(1, 2).fill_(0).cuda()
        self.plane_rot_q_xyAxis_xy_padded = self.plane_rot_q_xyAxis_xy.repeat(50000, 1) # 平面数量最多5w个呗
        self.plane_rot_q_normal_z_padded = self.plane_rot_q_normal_z.repeat(50000, 1)
        # learnable plane parameters
        self._plane_center = torch.empty(0)
        self._plane_radii_xy_p = torch.empty(0) # 非对称可偏心矩形 -r_x-, +r_x+
        self._plane_radii_xy_n = torch.empty(0) # 非对称可偏心矩形 -r_y-, -r_y-
        self._plane_rot_q_normal_wxy = torch.empty(0) # (N,3) 这个才是真正的法向方向 和rot_q_normal_z一同构成四元数
        self._plane_rot_q_xyAxis_w = torch.empty(0) # (N,1)平面内部两个方向的旋转？
        self._plane_rot_q_xyAxis_z = torch.empty(0) # (N,1) 和 rot_q_xyAxis_xy一同构成四元数
        self._plane_ids = torch.empty(0) # (N) 平面实例id

    @property
    def get_plane_center(self):
        return self._plane_center
    
    @property
    def get_plane_radii_xy_p(self):
        return self._plane_radii_xy_p
    
    @property
    def get_plane_radii_xy_n(self):
        return self._plane_radii_xy_n
    
    @property
    def get_plane_rot_q_normal_wxy(self):
        return self._plane_rot_q_normal_wxy
    
    @property
    def get_plane_rot_q_xyAxis_w(self):
        return self._plane_rot_q_xyAxis_w
    
    @property
    def get_plane_rot_q_xyAxis_z(self):
        return self._plane_rot_q_xyAxis_z
    
    @property
    def get_plane_ids(self):
        return self._plane_ids
    
    def initialize_from_sphere(self):
        logger.info("Initializing planes from sphere...")
        self.initialized = True
        plane_num = self.plane_cfg.get_int('init_plane_num')
        ratio = self.data_cfg.get_float('sphere_ratio', default=0.5)
        radius = self.scene_bounding_sphere * ratio
        init_centers, init_rot_q_normal, init_rot_q_xyAxis = model_util.get_plane_param_from_sphere(plane_num, radius)

        # =========================  define model parameters  ======================
        self._plane_center = nn.Parameter(init_centers.cuda().requires_grad_(True))
        self._plane_radii_xy_p = nn.Parameter(torch.Tensor(plane_num, 2).fill_(self.init_radii).cuda(), requires_grad=True)
        self._plane_radii_xy_n = nn.Parameter(torch.Tensor(plane_num, 2).fill_(self.init_radii).cuda(), requires_grad=True)
        self._plane_rot_q_normal_wxy = nn.Parameter(init_rot_q_normal[:, :3].cuda(), requires_grad=True)
        self._plane_rot_q_xyAxis_w = nn.Parameter(init_rot_q_xyAxis[:, 0:1].cuda(), requires_grad=True)
        self._plane_rot_q_xyAxis_z = nn.Parameter(init_rot_q_xyAxis[:, 3:4].cuda(), requires_grad=True)

        # =========================  plane visualization  ======================
        self.draw_plane(epoch=-1, suffix='initial-sphere')

    def initialize_with_given_pts(self, pts, normals, radiis=None):
        self.initialized = True
        init_centers = pts
        plane_num = init_centers.shape[0]

        init_rot_q_normal = model_util.get_rotation_quaternion_of_normal(normals)
        init_rot_angle_xyAxis = torch.tensor([0.]).reshape(1, 1).repeat(normals.shape[0], 1)
        init_rot_q_xyAxis = model_util.get_rotation_quaternion_of_xyAxis(normals.shape[0], angle=init_rot_angle_xyAxis)

        # =========================  define model parameters  ======================
        self._plane_center = nn.Parameter(init_centers.cuda().requires_grad_(True))
        if radiis is None:
            self._plane_radii_xy_p = nn.Parameter(torch.Tensor(plane_num, 2).fill_(self.init_radii).cuda(), requires_grad=True)
            self._plane_radii_xy_n = nn.Parameter(torch.Tensor(plane_num, 2).fill_(self.init_radii).cuda(), requires_grad=True)
        else:
            self._plane_radii_xy_p = nn.Parameter(radiis[:, :2].cuda(), requires_grad=True)
            self._plane_radii_xy_n = nn.Parameter(radiis[:, 2:].cuda(), requires_grad=True)
        self._plane_rot_q_normal_wxy = nn.Parameter(init_rot_q_normal[:, :3].cuda(), requires_grad=True)
        self._plane_rot_q_xyAxis_w = nn.Parameter(init_rot_q_xyAxis[:, 0:1].cuda(), requires_grad=True)
        self._plane_rot_q_xyAxis_z = nn.Parameter(init_rot_q_xyAxis[:, 3:4].cuda(), requires_grad=True)


    def initialize_from_mvsa(self, mvsa_path, color_path):
        from simple_knn._C import distCUDA2
        # load mvsa pcd from path
        mvsa_pcd = o3d.io.read_point_cloud(mvsa_path)
        mvsa_points = np.array(mvsa_pcd.points).astype(np.float32)
        mvsa_normals = np.array(mvsa_pcd.normals).astype(np.float32)
        mvsa_colors = np.array(mvsa_pcd.colors).astype(np.float32)
        inst_color_list = np.load(color_path).astype(np.float32)

        # 首先构造plane_ids
        def color_key(c, scale=255):
            return tuple((c * scale).round().astype(int))
        color2id = {color_key(c): i for i, c in enumerate(inst_color_list)}
        pcd_ids = np.zeros(len(mvsa_colors), dtype=np.int32)
        for i, c in enumerate(mvsa_colors):
            pcd_ids[i] = color2id.get(color_key(c), 0)
        unique_ids = np.unique(pcd_ids)
        print(f'{unique_ids.shape[0]} planar instance loaded from mvsa points')

        # 在每个实例点云内部初始化对应的planarsplat平面属性，然后最后链接起来
        init_center_list = []
        init_radii_list = []
        init_rot_q_normal_list = []
        init_ids_list = []

        sample_ratio = 0.1
        max_per_plane = 400
        min_per_plane = 10
        for idx in unique_ids:
            if idx == 0:
                continue
            # idx 对应的平面索引，然后进行0.5倍降采样，planarsplat不需要那么稠密
            mask_idx = (pcd_ids == idx)
            inst_indices = np.where(mask_idx)[0]
            num_keep = int(len(inst_indices) * sample_ratio)
            num_keep = min(num_keep, max_per_plane)
            num_keep = max(num_keep, min_per_plane)

            keep_indices = np.random.choice(inst_indices, size=num_keep, replace=False)
            sampled_mask = np.zeros_like(mask_idx, dtype=bool)
            sampled_mask[keep_indices] = True

            points_idx = torch.from_numpy(mvsa_points[sampled_mask]).float().cuda()
            normals_idx = torch.from_numpy(mvsa_normals[sampled_mask]).float().cuda()
            dist = torch.sqrt(torch.clamp_min(distCUDA2(points_idx), 0.001)) * 0.4

            init_centers_idx = points_idx
            init_radii_idx = torch.stack([dist, dist, dist, dist], dim=1)
            init_rot_q_normal_idx = model_util.get_rotation_quaternion_of_normal(normals_idx)
            init_ids_idx = torch.ones(num_keep).float().cuda()*idx
            
            init_center_list += [init_centers_idx]
            init_radii_list += [init_radii_idx]
            init_rot_q_normal_list += [init_rot_q_normal_idx]
            init_ids_list += [init_ids_idx]
        
        init_center_total = torch.cat(init_center_list, dim=0)
        init_radii_total = torch.cat(init_radii_list, dim=0)
        init_rot_q_normal_total = torch.cat(init_rot_q_normal_list, dim=0)
        init_ids_total = torch.cat(init_ids_list, dim=0)
        init_rot_angle_xyAxis = torch.tensor([0.]).reshape(1, 1).repeat(init_center_total.shape[0], 1)
        init_rot_q_xyAxis = model_util.get_rotation_quaternion_of_xyAxis(init_center_total.shape[0], angle=init_rot_angle_xyAxis)
        
        print(f'{init_center_total.shape[0]} primitives initialized from mvsa points')

        # =========================  define model parameters  ======================
        self._plane_center = nn.Parameter(init_center_total.cuda(), requires_grad=True)
        self._plane_radii_xy_p = nn.Parameter(init_radii_total[:, :2].cuda(), requires_grad=True)
        self._plane_radii_xy_n = nn.Parameter(init_radii_total[:, 2:].cuda(), requires_grad=True)
        self._plane_rot_q_normal_wxy = nn.Parameter(init_rot_q_normal_total[:, :3].cuda(), requires_grad=True)
        self._plane_rot_q_xyAxis_w = nn.Parameter(init_rot_q_xyAxis[:, 0:1].cuda(), requires_grad=True)
        self._plane_rot_q_xyAxis_z = nn.Parameter(init_rot_q_xyAxis[:, 3:4].cuda(), requires_grad=True)
        self._plane_ids = nn.Parameter(init_ids_total.float().cuda(), requires_grad=True)

        # =========================  plane visualization  ======================
        self.draw_plane(epoch=-1, suffix='initial-mvsa', color_path=color_path)
    

    def initialize_from_mvsa_2(self, mvsa_path, color_path):
        from simple_knn._C import distCUDA2
        # load mvsa pcd from path
        mvsa_pcd = o3d.io.read_point_cloud(mvsa_path)
        mvsa_points = np.array(mvsa_pcd.points).astype(np.float32)
        mvsa_normals = np.array(mvsa_pcd.normals).astype(np.float32)
        mvsa_colors = np.array(mvsa_pcd.colors).astype(np.float32)
        inst_color_list = np.load(color_path).astype(np.float32)

        # 首先构造plane_ids
        def color_key(c, scale=255):
            return tuple((c * scale).round().astype(int))
        color2id = {color_key(c): i for i, c in enumerate(inst_color_list)}
        pcd_ids = np.zeros(len(mvsa_colors), dtype=np.int32)
        for i, c in enumerate(mvsa_colors):
            pcd_ids[i] = color2id.get(color_key(c), 0)
        unique_ids = np.unique(pcd_ids)
        print(f'{unique_ids.shape[0]} planar instance loaded from mvsa points')

        # 在每个实例点云内部初始化对应的planarsplat平面属性，然后最后链接起来
        init_center_list = []
        init_radii_list = []
        init_rot_q_normal_list = []
        init_ids_list = []

        voxel_size = 0.1
        for idx in unique_ids:
            if idx == 0:
                continue
            # idx 对应的平面索引，然后进行0.5倍降采样，planarsplat不需要那么稠密
            mask_idx = (pcd_ids == idx)
            points_idx_many = mvsa_points[mask_idx]
            pcd_many = o3d.geometry.PointCloud()
            pcd_many.points = o3d.utility.Vector3dVector(points_idx_many)
            pcd_many = pcd_many.voxel_down_sample(voxel_size)
            points_idx = torch.from_numpy(np.asarray(pcd_many.points)).float().cuda()

            # normals_mean
            normals_mean = np.mean(mvsa_normals[mask_idx], axis=0)
            normals_idx_np = np.tile(normals_mean[None, :], (points_idx.shape[0], 1))
            normals_idx = torch.from_numpy(normals_idx_np).float().cuda()
            normals_idx = F.normalize(normals_idx, dim=1)
            dist = torch.sqrt(torch.clamp_min(distCUDA2(points_idx), 0.001)) * 0.5

            init_centers_idx = points_idx
            init_radii_idx = torch.stack([dist, dist, dist, dist], dim=1)
            init_rot_q_normal_idx = model_util.get_rotation_quaternion_of_normal(normals_idx)
            init_ids_idx = torch.ones(points_idx.shape[0]).float().cuda()*idx
            
            init_center_list += [init_centers_idx]
            init_radii_list += [init_radii_idx]
            init_rot_q_normal_list += [init_rot_q_normal_idx]
            init_ids_list += [init_ids_idx]
        
        init_center_total = torch.cat(init_center_list, dim=0)
        init_radii_total = torch.cat(init_radii_list, dim=0)
        init_rot_q_normal_total = torch.cat(init_rot_q_normal_list, dim=0)
        init_ids_total = torch.cat(init_ids_list, dim=0)
        init_rot_angle_xyAxis = torch.tensor([0.]).reshape(1, 1).repeat(init_center_total.shape[0], 1)
        init_rot_q_xyAxis = model_util.get_rotation_quaternion_of_xyAxis(init_center_total.shape[0], angle=init_rot_angle_xyAxis)
        
        print(f'{init_center_total.shape[0]} primitives initialized from mvsa points')

        # =========================  define model parameters  ======================
        self._plane_center = nn.Parameter(init_center_total.cuda(), requires_grad=True)
        self._plane_radii_xy_p = nn.Parameter(init_radii_total[:, :2].cuda(), requires_grad=True)
        self._plane_radii_xy_n = nn.Parameter(init_radii_total[:, 2:].cuda(), requires_grad=True)
        self._plane_rot_q_normal_wxy = nn.Parameter(init_rot_q_normal_total[:, :3].cuda(), requires_grad=True)
        self._plane_rot_q_xyAxis_w = nn.Parameter(init_rot_q_xyAxis[:, 0:1].cuda(), requires_grad=True)
        self._plane_rot_q_xyAxis_z = nn.Parameter(init_rot_q_xyAxis[:, 3:4].cuda(), requires_grad=True)
        self._plane_ids = nn.Parameter(init_ids_total.float().cuda(), requires_grad=True)

        # =========================  plane visualization  ======================
        self.draw_plane(epoch=-1, suffix='initial-mvsa', color_path=color_path)


    def initialize_from_mesh(self, mesh_path):
        self.initialized = True
        plane_num = self.plane_cfg.get_int('init_plane_num')
        ratio = self.data_cfg.get_float('sphere_ratio', default=0.5)
        radius = self.scene_bounding_sphere * ratio
        _, _, init_rot_q_xyAxis = model_util.get_plane_param_from_sphere(plane_num, radius)

        mesh = trimesh.load_mesh(mesh_path)
        # target_vertex_count = min(plane_num * 2, 5000)
        target_vertex_count = min(plane_num * 2, 10000)
        simplified_mesh = mesh.simplify_quadratic_decimation(target_vertex_count)
        vertices = simplified_mesh.vertices
        normals = simplified_mesh.vertex_normals
        faces = simplified_mesh.faces
        if faces.shape[0] < target_vertex_count:
            faces = np.concatenate((faces, faces[:target_vertex_count-faces.shape[0]]), axis=0)
        elif faces.shape[0] > target_vertex_count:
            faces = faces[:target_vertex_count]
        else:
            pass
        faces_v = vertices[faces.reshape(-1)].reshape(target_vertex_count, 3, 3)
        faces_n = normals[faces.reshape(-1)].reshape(target_vertex_count, 3, 3)
        plane_centers = (np.mean(faces_v, axis=1) - self.pose_cfg.offset) * self.pose_cfg.scale
        plane_normals = np.mean(faces_n, axis=1)
        plane_radii = np.mean(np.linalg.norm(plane_centers[:, None] -(faces_v- self.pose_cfg.offset) * self.pose_cfg.scale, axis=-1), axis=-1)

        sampled_idx = random.sample(range(0, target_vertex_count), plane_num)
        plane_centers = plane_centers[sampled_idx]
        plane_normals = plane_normals[sampled_idx]
        plane_radii = plane_radii[sampled_idx]

        init_centers_new = torch.from_numpy(plane_centers).float()
        init_normals_new = torch.from_numpy(plane_normals).float()
        init_rot_q_normal_new = model_util.get_rotation_quaternion_of_normal(init_normals_new).cuda()
        init_rot_q_xyAxis_new = init_rot_q_xyAxis
        init_radii = torch.from_numpy(plane_radii.reshape(-1, 1))
        init_radii = torch.cat([init_radii, init_radii], dim=-1).float() * 0.3

        # =========================  define model parameters  ======================
        self._plane_center = nn.Parameter(init_centers_new.cuda().requires_grad_(True))
        self._plane_radii_xy_p = nn.Parameter(init_radii.cuda(), requires_grad=True)
        self._plane_radii_xy_n = nn.Parameter(init_radii.cuda(), requires_grad=True)
        self._plane_rot_q_normal_wxy = nn.Parameter(init_rot_q_normal_new[:, :3].cuda(), requires_grad=True)
        self._plane_rot_q_xyAxis_w = nn.Parameter(init_rot_q_xyAxis_new[:, 0:1].cuda(), requires_grad=True)
        self._plane_rot_q_xyAxis_z = nn.Parameter(init_rot_q_xyAxis_new[:, 3:4].cuda(), requires_grad=True)
        self._plane_ids = nn.Parameter(torch.arange(plane_num).float().cuda(), requires_grad=True)

         # =========================  plane visualization  ======================
        self.draw_plane(epoch=-1, suffix='initial-mesh')
    
    def initialize_as_zero(self, plane_num):
        # 可学习参数全都初始化为0可以理解
        self.initialized = True
        # =========================  define model parameters  ======================
        self._plane_center = nn.Parameter(torch.zeros(plane_num, 3).cuda().requires_grad_(True))
        self._plane_radii_xy_p = nn.Parameter(torch.zeros(plane_num, 2).cuda().requires_grad_(True))
        self._plane_radii_xy_n = nn.Parameter(torch.zeros(plane_num, 2).cuda().requires_grad_(True))
        self._plane_rot_q_normal_wxy = nn.Parameter(torch.zeros(plane_num, 3).cuda().requires_grad_(True))
        self._plane_rot_q_xyAxis_w = nn.Parameter(torch.zeros(plane_num, 1).cuda().requires_grad_(True))
        self._plane_rot_q_xyAxis_z = nn.Parameter(torch.zeros(plane_num, 1).cuda().requires_grad_(True))

    def check_model(self):
        # 判断在prune和split的过程中有没有漏了哪个参数的复制和删除吧
        assert self.get_plane_center.shape[0] == self.get_plane_radii_xy_p.shape[0]
        assert self.get_plane_center.shape[0] == self.get_plane_radii_xy_n.shape[0]
        assert self.get_plane_center.shape[0] == self.get_plane_rot_q_normal_wxy.shape[0]
        assert self.get_plane_center.shape[0] == self.get_plane_rot_q_xyAxis_w.shape[0]
        assert self.get_plane_center.shape[0] == self.get_plane_rot_q_xyAxis_z.shape[0]
        assert self.get_plane_center.shape[0] == self.get_plane_ids.shape[0]

    def get_plane_num(self):
        self.check_model()
        return self.get_plane_center.shape[0]
    
    def set_max_and_min_radii(self, ite):
        # radii_max_list 和 radii_min_list 随着训练的进行应该是会进行动态更新的
        max_radii, min_radii = model_util.get_max_and_min_radii(ite, self.radii_max_list, self.radii_min_list, self.radii_milestone_list)
        self.max_radii = max_radii
        self.min_radii = min_radii
 
    def get_plane_rot_q(self):
        # 将平面法相和平面内部旋转的三个可优化向量和两个固定向量按照[w,x,y,z]的顺序进行拼接
        # 得到平面整体法相朝向和平面内部的旋转
        plane_num = self.get_plane_num()
        plane_rot_q_xyAxis = torch.cat([self._plane_rot_q_xyAxis_w, self.plane_rot_q_xyAxis_xy_padded[:plane_num], self._plane_rot_q_xyAxis_z], dim=-1)
        plane_rot_q_normal = torch.cat([self._plane_rot_q_normal_wxy, self.plane_rot_q_normal_z_padded[:plane_num]], dim=-1)
        plane_rot_q_xyAxis = F.normalize(plane_rot_q_xyAxis, dim=-1)
        plane_rot_q_normal = F.normalize(plane_rot_q_normal, dim=-1)
        return plane_rot_q_xyAxis, plane_rot_q_normal
    
    def get_plane_radii(self):
        # 平面半径，如果是double说明是非对称可偏心矩形
        if self.radii_dir_type == 'double':
            plane_radii = torch.cat([self._plane_radii_xy_p, self._plane_radii_xy_n], dim=-1)
        elif self.radii_dir_type == 'single':
            plane_radii = torch.cat([self._plane_radii_xy_p, self._plane_radii_xy_p], dim=-1)
        else:
            raise
        return plane_radii.clamp(min=self.min_radii, max=self.max_radii)

    def get_plane_geometry(self, ite=-1):
        # 这是比较核心的内容，从矩形基元上获得了真正可用的平面参数之后才能用于rasterization
        fix_rot_normal = self.plane_cfg.fix_rot_normal
        fix_rot_xy = self.plane_cfg.fix_rot_xy
        fix_center = self.plane_cfg.fix_center
        fix_radii = True if (ite < self.coarse_stage_ite and not self.debug_on) else self.plane_cfg.fix_radii

        plane_num = self.get_plane_num()
        plane_rot_q_xyAxis, plane_rot_q_normal = self.get_plane_rot_q()
        plane_radii = self.get_plane_radii()
        
        if fix_rot_normal:
            plane_rot_q_normal = plane_rot_q_normal.detach()
        if fix_rot_xy:
            plane_rot_q_xyAxis = plane_rot_q_xyAxis.detach()
        if fix_radii:
            plane_radii = plane_radii.detach()
        
        # 平面整体旋转在法相方向和面内解耦，先把z轴转成plane normal，在绕这个normal水平旋转矩形
        plane_rot_q = model_util.quaternion_mult(plane_rot_q_normal, plane_rot_q_xyAxis)
        plane_rot_q = F.normalize(plane_rot_q, dim=-1)
        if self.rot_delta is not None:
            assert self.rot_delta.shape[0] == plane_rot_q.shape[0]
            plane_rot_q = model_util.quaternion_mult(self.rot_delta, plane_rot_q)
        plane_rot_matrix = model_util.quat_to_rot(plane_rot_q) # 四元数转换为旋转矩阵

        # plane_xxx_standard是标准平面参数，按照旋转矩阵进行旋转即可得到真实的平面法向量、平面内x、y轴的朝向
        plane_normal = torch.bmm(plane_rot_matrix, self.plane_normal_standard.reshape(-1, 3, 1).expand(plane_num, 3, 1)).squeeze(-1)
        plane_xAxis = torch.bmm(plane_rot_matrix, self.plane_xAxis_standard.reshape(-1, 3, 1).expand(plane_num, 3, 1)).squeeze(-1)
        plane_yAxis = torch.bmm(plane_rot_matrix, self.plane_yAxis_standard.reshape(-1, 3, 1).expand(plane_num, 3, 1)).squeeze(-1)

        plane_center = self.get_plane_center
        if fix_center:
            plane_center = plane_center.detach()

        # plane_offset则是平面沿着法线方向的offset，合理
        plane_offset = model_util.compute_offset(plane_center, plane_normal).reshape(-1, 1).float()

        return plane_normal, plane_offset, plane_center, plane_radii, plane_rot_q, plane_xAxis, plane_yAxis
    
    def get_plane_geometry_simple(self, ite=-1, in_fix_rot_n=False, in_fix_rot_xy=False, in_fix_radii=False, in_fix_center=False):
        fix_rot_normal = self.plane_cfg.fix_rot_normal or in_fix_rot_n
        fix_rot_xy = self.plane_cfg.fix_rot_xy or in_fix_rot_xy
        fix_center = self.plane_cfg.fix_center or in_fix_center
        fix_radii = True if (ite < self.coarse_stage_ite and not self.debug_on) else self.plane_cfg.fix_radii
        fix_radii = fix_radii or in_fix_radii

        # 简单版本中，也即forward函数中，在coarse stage没有对平面尺寸进行优化呢
        plane_rot_q_xyAxis, plane_rot_q_normal = self.get_plane_rot_q()
        plane_radii = self.get_plane_radii()
        if ite < self.coarse_stage_ite:
            plane_radii = plane_radii.detach()
        
        if fix_rot_normal:
            plane_rot_q_normal = plane_rot_q_normal.detach()
        if fix_rot_xy:
            plane_rot_q_xyAxis = plane_rot_q_xyAxis.detach()
        if fix_radii:
            plane_radii = plane_radii.detach()
        
        plane_rot_q = model_util.quaternion_mult(plane_rot_q_normal, plane_rot_q_xyAxis)
        plane_rot_q = F.normalize(plane_rot_q, dim=-1)
        if self.rot_delta is not None:
            assert self.rot_delta.shape[0] == plane_rot_q.shape[0]
            plane_rot_q = model_util.quaternion_mult(self.rot_delta, plane_rot_q)

        plane_center = self._plane_center 
        if fix_center:
            plane_center = plane_center.detach()

        # 在forward中使用的简单版本，只获取了平面中心、平面半径和平面旋转四元数，并非具体的平面几何参数
        return plane_center, plane_radii, plane_rot_q

    def get_plane_geometry_for_regularize(self):
        # 这个只返回了平面半径，以及平面内实际的x、y轴朝向
        plane_num = self.get_plane_num()
        plane_rot_q_xyAxis, plane_rot_q_normal = self.get_plane_rot_q()
        plane_radii = self.get_plane_radii()
        plane_rot_q = qMultCUDA(plane_rot_q_normal, plane_rot_q_xyAxis)

        plane_rot_q = F.normalize(plane_rot_q, dim=-1)
        if self.rot_delta is not None:
            raise ValueError
        plane_rot_matrix = q2RCUDA(plane_rot_q).permute(0, 2, 1)
        plane_xAxis = torch.bmm(plane_rot_matrix, self.plane_xAxis_standard.view(-1, 3, 1).expand(plane_num, 3, 1)).squeeze(-1)
        plane_yAxis = torch.bmm(plane_rot_matrix, self.plane_yAxis_standard.view(-1, 3, 1).expand(plane_num, 3, 1)).squeeze(-1)
        return plane_radii, plane_xAxis, plane_yAxis
    
    def get_splat_weight(self, ite=-1):
        # 这个splat weight了，这就不是我关心的部分了
        if self.RBF_type == 'rectangle':
            if ite == -1 or self.RBF_weight_change_type == 'max':
                weight = 300.
            elif self.RBF_weight_change_type == 'increase':
                max_weight = 300.
                ratio = ite / (self.max_total_iters // 10)
                weight = min(math.exp(-(1 - ratio)) * 20, max_weight)
            elif self.RBF_weight_change_type == 'min':
                max_weight = 300.
                ratio = 0.
                weight = min(math.exp(-(1 - ratio)) * 20, max_weight)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        return weight

    @ torch.no_grad()
    def instance_splat_mask(self, view_info, iter=-1):
        # 一系列raster所需的相机参数格式
        raster_cam_w2c = view_info.raster_cam_w2c
        raster_cam_fullproj = view_info.raster_cam_fullproj
        raster_cam_center = view_info.raster_cam_center
        tanfovx = view_info.tanfovx
        tanfovy = view_info.tanfovy
        raster_img_center = view_info.raster_img_center

        # ======================================= get mvsa_mask from view_info
        mvsa_mask = view_info.mask_mvsa # shape = (hw)
        mvsa_ids = torch.unique(mvsa_mask)
        raster_count = torch.zeros_like(mvsa_mask, dtype=torch.int32, device=mvsa_mask.device)

        # ======================================= set up plane model
        self.set_max_and_min_radii(iter)
        plane_center, plane_radii, plane_rot_q = self.get_plane_geometry_simple(ite=iter)
        plane_ids = self.get_plane_ids
        
        # ======================================= set up rasterization configuration
        splat_weight = self.get_splat_weight(ite=iter)
        raster_settings = RectRasterizationSettings(
                    image_height=self.H,
                    image_width=self.W,
                    tanfovx=tanfovx,
                    tanfovy=tanfovy,
                    bg=self.bg,
                    scale_modifier=1.0,
                    viewmatrix=raster_cam_w2c,
                    projmatrix=raster_cam_fullproj,
                    sh_degree=0,
                    campos=raster_cam_center,
                    prefiltered=False,
                    debug=False,
                    lambdaw=splat_weight * 5.0,
                    image_center=raster_img_center,
                    scales2=plane_radii[:, :2].detach()
        )
        rasterizer = RectRasterizer(raster_settings=raster_settings)

        # ======================================= rasterize each instance inner mvsa mask
        for idx in mvsa_ids:
            plane_valid_mask = (plane_ids == idx)
            if ~torch.any(plane_valid_mask): 
                continue
            plane_center_idx = plane_center[plane_valid_mask]
            plane_radii_idx = plane_radii[plane_valid_mask]
            plane_rot_q_idx = plane_rot_q[plane_valid_mask]
            screenspace_points_idx = torch.zeros_like(plane_center_idx, dtype=plane_center_idx.dtype, requires_grad=True, device="cuda")
            _, _, allmap_idx = rasterizer(
                    means3D = plane_center_idx,
                    means2D = screenspace_points_idx,
                    shs = None,
                    colors_precomp = torch.rand_like(plane_center_idx),
                    opacities = torch.ones_like(plane_center_idx)[:, :1],
                    scales = plane_radii_idx,
                    rotations = plane_rot_q_idx,
                    cov3D_precomp = None
            )
            rendered_depth = allmap_idx[0:1].squeeze().view(-1)
            raster_count[(mvsa_mask == idx) & (rendered_depth > 0)] += 1
        
        mvsa_mask[~(raster_count == 1)] = 0.0
        return mvsa_mask


    def forward(self, view_info, iter=-1, return_rgb=False, fix_rot_n=False, fix_rot_xy=False, fix_radii=False, fix_center=False):
        # 一系列raster所需的相机参数格式
        raster_cam_w2c = view_info.raster_cam_w2c
        raster_cam_fullproj = view_info.raster_cam_fullproj
        raster_cam_center = view_info.raster_cam_center
        tanfovx = view_info.tanfovx
        tanfovy = view_info.tanfovy
        raster_img_center = view_info.raster_img_center

        # ======================================= set up plane model
        self.set_max_and_min_radii(iter) # 这个函数也只在这里调用了，也就是每次获取forward所需的平面参数前需要规范一下
        # 如果我的想通过多次forward获取instance optimize mask的话，我调用get_plane_geometry_simple获取对应id的参数即可
        plane_center, plane_radii, plane_rot_q = self.get_plane_geometry_simple(
                                                    ite=iter, 
                                                    in_fix_rot_n=fix_rot_n, 
                                                    in_fix_rot_xy=fix_rot_xy, 
                                                    in_fix_radii=fix_radii, 
                                                    in_fix_center=fix_center
                                                    )

        # ======================================= set up rasterization configuration
        splat_weight = self.get_splat_weight(ite=iter)
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        # gridients应该是用来指导split或者prune的，但是我并不希望你prune
        screenspace_points = torch.zeros_like(plane_center, dtype=plane_center.dtype, requires_grad=True, device="cuda")
        try:
            screenspace_points.retain_grad()
        except:
            pass
        raster_settings = RectRasterizationSettings(
                    image_height=self.H,
                    image_width=self.W,
                    tanfovx=tanfovx,
                    tanfovy=tanfovy,
                    bg=self.bg,
                    scale_modifier=1.0,
                    viewmatrix=raster_cam_w2c,
                    projmatrix=raster_cam_fullproj,
                    sh_degree=0,
                    campos=raster_cam_center,
                    prefiltered=False,
                    debug=False,
                    lambdaw=splat_weight * 5.0,
                    image_center=raster_img_center,
                    scales2=plane_radii[:, :2].detach()
        )

        # ======================================= plane model forward
        colors_precomp = self.color_lut[self.get_plane_ids.long()]
        rasterizer = RectRasterizer(raster_settings=raster_settings)
        rgb, _, allmap = rasterizer(
                    means3D = plane_center,
                    means2D = screenspace_points,
                    shs = None,
                    colors_precomp = colors_precomp,
                    opacities = torch.ones_like(plane_center)[:, :1],
                    scales = plane_radii,
                    rotations = plane_rot_q,
                    cov3D_precomp = None
        )
        if return_rgb:
            return rgb, allmap
        else:
            return allmap

    def draw_plane(self, suffix='', epoch=-1, to_unscaled_coord=True, plane_id=None, save_mesh=True, color_path=None):
        plane_normal, _, plane_center, plane_radii, plane_rot_q, _, _ = self.get_plane_geometry()
        mesh_n = plot_rectangle_planes(
            plane_center, plane_normal, plane_radii, plane_rot_q, 
            epoch=epoch, 
            suffix='%s'%(suffix), 
            to_unscaled_coord=to_unscaled_coord, 
            pose_cfg=self.pose_cfg, 
            out_path=self.plot_dir if save_mesh else None,
            plane_id=plane_id, 
            color_type='normal')
        mesh_p = plot_rectangle_planes(
            plane_center, plane_normal, plane_radii, plane_rot_q, 
            epoch=epoch, 
            suffix='%s'%(suffix), 
            to_unscaled_coord=to_unscaled_coord, 
            pose_cfg=self.pose_cfg, 
            out_path=self.plot_dir if save_mesh else None,
            plane_id=plane_id, 
            color_type='prim')
        mesh_i = plot_rectangle_planes(
            plane_center, plane_normal, plane_radii, plane_rot_q, 
            epoch=epoch, 
            suffix='%s'%(suffix), 
            to_unscaled_coord=to_unscaled_coord, 
            pose_cfg=self.pose_cfg, 
            out_path=self.plot_dir if save_mesh else None,
            plane_id=self.get_plane_ids, 
            color_type='mvsa',
            color_path=color_path)
        return mesh_n, mesh_p