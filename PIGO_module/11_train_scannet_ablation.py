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

import os
from datetime import datetime
import torch
import random
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim, lncc, get_img_grad_weight, l1_depth_loss, mask_normal_loss_l1, mask_distance_loss_l1,mask_alpha_loss_l1
from utils.loss_utils import distance_link_update, mask_distance_loss_link
from utils.graphics_utils import patch_offsets, patch_warp
from gaussian_renderer import render, network_gui
import sys, time
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import cv2
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, erode
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.app_model import AppModel
from scene.cameras import Camera, DepthCamera
from simple_knn._C import distCUDA2
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import time
import torch.nn.functional as F
torch.cuda.device_count()
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(22)

def gen_virtul_cam(cam, trans_noise=1.0, deg_noise=15.0):
    # 生成虚拟相机用来为没有nearestid的相机帧计算跨视图几何损失 这不就利用了3DGS的新视角合成能力嘛~
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = cam.R.transpose()
    Rt[:3, 3] = cam.T
    Rt[3, 3] = 1.0
    # 所以这里可以看出每一帧上的R应该是从世界坐标系变换到相机坐标系
    C2W = np.linalg.inv(Rt)

    translation_perturbation = np.random.uniform(-trans_noise, trans_noise, 3)
    rotation_perturbation = np.random.uniform(-deg_noise, deg_noise, 3)
    rx, ry, rz = np.deg2rad(rotation_perturbation)
    Rx = np.array([[1, 0, 0],
                    [0, np.cos(rx), -np.sin(rx)],
                    [0, np.sin(rx), np.cos(rx)]])
    
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, np.cos(ry)]])
    
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                    [np.sin(rz), np.cos(rz), 0],
                    [0, 0, 1]])
    R_perturbation = Rz @ Ry @ Rx

    C2W[:3, :3] = C2W[:3, :3] @ R_perturbation
    C2W[:3, 3] = C2W[:3, 3] + translation_perturbation
    Rt = np.linalg.inv(C2W)
    virtul_cam = Camera(100000, Rt[:3, :3].transpose(), Rt[:3, 3], cam.FoVx, cam.FoVy,
                        cam.image_width, cam.image_height,
                        cam.image_path, cam.image_name, 100000,
                        trans=np.array([0.0, 0.0, 0.0]), scale=1.0, 
                        preload_img=False, data_device = "cuda")
    return virtul_cam

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, voxel_size, init_w_gaussian, args):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    # backup main code
    cmd = f'cp ./train_scannet_ablation.py {dataset.model_path}/'
    os.system(cmd)
    cmd = f'cp -rf ./arguments {dataset.model_path}/'
    os.system(cmd)
    cmd = f'cp -rf ./gaussian_renderer {dataset.model_path}/'
    os.system(cmd)
    cmd = f'cp -rf ./scene {dataset.model_path}/'
    os.system(cmd)
    cmd = f'cp -rf ./utils {dataset.model_path}/'
    os.system(cmd)

    # 0.统一读取coarse层 plane 3dgs fix配置参数
    hive_dict = {}
    hive_dict['fix_xyz'] = args.fix_xyz         # 是否固定高斯点的位置坐标
    hive_dict['fix_normal'] = args.fix_normal   # 是否固定高斯点的法向量（没实现）
    hive_dict['use_mask'] = args.use_mask       # 是否启用平面mask监督
    hive_dict['no_rgb'] = args.no_rgb           # 是否启用RGB监督（包括ncc_loss）
    hive_dict['use_opa'] = args.use_opa         # 是否将场景透明度二值化
    hive_dict['mask_type'] = args.mask_type     # GHPS mask 类型
    hive_dict['fix_opa'] = args.fix_opa   # 是否固定高斯不透明度
    hive_dict['use_alpha'] = args.use_alpha   # 是否使用输入平面mask监督渲染alpha
    hive_dict['use_depth'] = args.use_depth   # 是否使用单目估计深度监督mask区域深度
    hive_dict['fix_rgb'] = args.fix_rgb       # 是否固定高斯RGB
    hive_dict['use_temp'] = args.use_temp       # 是否引入临时GS
    hive_dict['no_initial'] = args.no_initial   # 是否不使用已有几何初始化高斯

    weight_normal_p = 1.0
    weight_distance = 5.0
    weight_normal_l = 2.0

    print(f"Loss Weights:\n")
    print(f"  Planar Normal Weight: {weight_normal_p}\n")
    print(f"  Local Normal Weight: {weight_normal_l}\n")
    print(f"  Planar Dist Weight: {weight_distance}\n")

    albation_flag = False
    if args.no_normal_p:
        weight_normal_p = 0.0
        albation_flag = True
        print("★ Ablations settings: No Planar Normal Loss")
    if args.no_distance:
        weight_distance = 0.0
        albation_flag = True
        print("★ Ablations settings: No Distance Loss")
    if not args.increment:
        albation_flag = True
        print("★ Ablations settings: naive distance loss")
    if not args.use_temp:
        albation_flag = True
        print("★ Ablations settings: No Instance-level Splatting")
    if args.no_initial:
        albation_flag = True
        print("★ Ablations settings: No Geometric Initialization")
    if not albation_flag:
        print("★ Ablations settings: Full version of PIGS Running")
    
    # 1.构建高斯场景表示对象 2.构建场景scene对象
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, voxel_size=voxel_size, init_w_gaussian=init_w_gaussian, hive_dict=hive_dict)
    gaussians.training_setup(opt) # 将相关的优化配置参数传入高斯模型中

    app_model = AppModel()
    app_model.train()
    app_model.cuda()
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        app_model.load_weights(scene.model_path)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # torch.cuda.Event作用是用来精确记录GPU的运行时间
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    viewpoint_index = 0
    inst_num = None # maskcluster 对mask聚类得到的mask 实例数目
    # tran_xyz 是计算得到当前inst_id distance值时的相机坐标 N_i表示该mask已经进行了多少次distance_link
    # distance_i是当前算出来的link之后的distance值 w_i是link之后的权重值 表示distance_i的加权方差值
    distance_link = None # shape=(inst_num, 6)--[tran_x, tran_y, tran_z, distance_i, w_i, N_i]
    loss_mask_distance, loss_mask_normal, loss_mask_alpha = None, None, None
    ema_loss_for_log = 0.0
    ema_single_view_for_log = 0.0
    ema_multi_view_geo_for_log = 0.0
    ema_multi_view_pho_for_log = 0.0
    ema_normal_loss_for_log = 0.0
    ema_distance_loss_for_log = 0.0
    ema_alpha_loss_for_log = 0.0
    ema_nan_num_for_log = 0
    normal_loss, geo_loss, ncc_loss = None, None, None
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    debug_path = os.path.join(scene.model_path, "debug")
    os.makedirs(debug_path, exist_ok=True)

    if args.increment:
        inst_num = len(scene.colorid_list)
        distance_link = torch.zeros(inst_num, 6).cuda()

    for iteration in range(first_iter, opt.iterations + 1):

        # if iteration > args.plane_normal and iteration % 1000 == 0 and distance_link is not None:
        #     np.savetxt(os.path.join(scene.model_path, f"distance_link_{iteration}.txt"), distance_link.cpu().numpy())

        iter_start.record()
        gaussians.update_learning_rate(iteration)

        
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        # Pick a random Camera 从训练集中选择随机的相机用来渲染debug 一个优化iteration中只选择了一个相机
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))      

        gt_image, gt_image_gray, gt_depth_image, mask_normal_com = viewpoint_cam.get_image()
        
        # 读取当前相机的平移主要是为了用来做distance_link
        # 而读取当前相机的rot主要是为了将当前相机的法向量变换到世界坐标系
        view_trans = viewpoint_cam.camera_center # c2w trans of viewpoint_cam
        view_rot = viewpoint_cam.camera_rot # c2w rot of viewpoint_cam
        
        if mask_normal_com is not None:
            im_width = mask_normal_com.shape[0]
            im_height = mask_normal_com.shape[1]
            mask_normal = mask_normal_com[...,:-1]
            mask_xpd = mask_normal_com[...,-1]
            unique_values = torch.unique(mask_xpd)
            skip_flag = (unique_values.shape[0] == 1)
        else:
            mask_xpd = None
        
        # skip_flag = True 说明没有mask阔以用来监督跳过
        if skip_flag:
            continue

        if iteration > 1000 and opt.exposure_compensation:
            # 感觉app_model是不是就是文中提到的用来做光度补偿的那个可优化线性参数啊
            gaussians.use_app = True

        # Render debug渲染参数设置
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background # 为什么背景颜色都需要专门设置 bg_color是在配置raster参数时需要设置的
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, app_model=app_model,
                            return_plane=iteration>opt.single_view_weight_from_iter, return_depth_normal=iteration>opt.single_view_weight_from_iter)
        
        # instance-level splatting
        if hive_dict["use_temp"] and (iteration > args.plane_normal):
            mask_xpd_temp=mask_xpd-mask_xpd
            for value in unique_values:
                mask_temp = (gaussians.ids == value).squeeze(1)
                if ~torch.any(mask_temp): 
                    continue
                gaussians_temp=GaussianModel(dataset.sh_degree)
                gaussians_temp._xyz=gaussians._xyz[mask_temp]
                gaussians_temp._features_dc=gaussians._features_dc[mask_temp]
                gaussians_temp._features_rest=gaussians._features_rest[mask_temp]
                gaussians_temp._opacity=gaussians._opacity[mask_temp]
                gaussians_temp._scaling=gaussians._scaling[mask_temp]
                gaussians_temp._rotation=gaussians._rotation[mask_temp]
                gaussians_temp.active_sh_degree=gaussians.active_sh_degree
                gaussians_temp._scaling=gaussians._scaling[mask_temp]
                out_temp=render(viewpoint_cam, gaussians_temp, pipe, bg, app_model=app_model,
                     return_plane=iteration>opt.single_view_weight_from_iter, return_depth_normal=iteration>opt.single_view_weight_from_iter)
                depth_temp = out_temp["plane_depth"]
                alpha_temp = out_temp["rendered_alpha"]

                index = (mask_xpd == value).nonzero(as_tuple=True)
                row, col = index[0][0], index[1][0]
                normal_xpd = mask_normal[row, col]

                mask_id_temp=(depth_temp.squeeze(0)!=0.0)&(mask_xpd==value) # use 3dgs for projected mask
                mean_alpha = torch.mean(alpha_temp[alpha_temp>0.0])
                mask_alpha=alpha_temp.squeeze(0)<0.35#(mean_alpha*0.65)
                mask_alpha=mask_alpha & mask_id_temp
                mask_xpd_temp[mask_id_temp]+=1.0
        
            mask_cross = ~(mask_xpd_temp==1.0)
            mask_xpd[mask_cross]=0.0
            mask_normal_com[...,-1]=mask_xpd

        if (~(mask_xpd==0.0)).sum() == 0:
            continue

        # rasterization calling
        image, viewspace_point_tensor, visibility_filter, radii = \
            render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        # Loss
        if hive_dict['no_rgb'] and (iteration < args.plane_normal):
            weight_rgb = 0.0
        else:
            weight_rgb = 1.0
        
        # RGB Loss vs. Mask normal input
        mask_array = mask_xpd.detach().cpu().numpy().astype(np.uint8)
        mask_color_np = np.array(scene.colorid_list)
        mask_color_np[0] = np.array([0,0,0])
        mask_xpd_color = np.take(mask_color_np, mask_array, axis=0)
        mask_xpd_color = torch.tensor(mask_xpd_color).to(image.device).permute(2,0,1).clamp(0,1)
        Ll1 = l1_loss(image, mask_xpd_color, mask_plane=mask_xpd)
        image_loss = Ll1
        loss = weight_rgb * image_loss.clone()

        # Depth Loss
        if gt_depth_image is not None and (iteration > args.plane_normal) and hive_dict['use_depth']:#and not any(hive_dict.values()):
            navie_loss_d = l1_depth_loss(render_pkg['plane_depth'], gt_depth_image, viewpoint_cam.max_depth, mask_plane=mask_xpd)
            loss += navie_loss_d
        
        # Opacity Loss 
        if mask_normal_com is not None and hive_dict["use_alpha"] and (iteration > args.plane_normal):
            alpha=render_pkg["rendered_alpha"]
            loss_mask_alpha = mask_alpha_loss_l1(alpha,mask_plane=mask_xpd)
            loss += loss_mask_alpha
        
        # Planar Normal Loss
        if mask_normal_com is not None and (iteration > args.plane_normal):
            loss_mask_normal = mask_normal_loss_l1(render_pkg["rendered_normal"], mask_normal, mask_plane=mask_xpd)
            loss += weight_normal_p*loss_mask_normal

        # Planar Distance Loss
        if mask_normal_com is not None and (iteration > args.plane_distance):
            if args.increment:
                with torch.no_grad():
                    distance_link, distance_mean = distance_link_update(distance_link, render_pkg['rendered_distance'], view_trans, view_rot, mask_normal_com, alpha=render_pkg["rendered_alpha"])
                loss_mask_distance = mask_distance_loss_link(render_pkg['rendered_distance'], distance_mean, mask_plane=mask_xpd, alpha=render_pkg["rendered_alpha"])
            else:
                loss_mask_distance = mask_distance_loss_l1(render_pkg['rendered_distance'], mask_plane=mask_xpd)
            loss += weight_distance*loss_mask_distance

        # whether to fix certain gaussian attributes without optimizing
        if hive_dict["fix_xyz"]:
            gaussians._xyz = gaussians._xyz.detach()
        if hive_dict["fix_normal"]:
            gaussians._rotation = gaussians._rotation.detach()
        if hive_dict["fix_opa"]:
            gaussians._opacity= gaussians._opacity.detach()
        if hive_dict["fix_rgb"]:
            gaussians._features_dc = gaussians._features_dc.detach()
            gaussians._features_rest = gaussians._features_rest.detach()

            
        # (Not Used) 在所有迭代都加入不透明度二值化约束
        if (iteration > args.plane_normal) and hive_dict['use_opa']:
            opac_ = gaussians.get_opacity
            opac_mask0 = torch.gt(opac_, 0.01) * torch.le(opac_, 0.5)
            opac_mask1 = torch.gt(opac_, 0.5) * torch.le(opac_, 0.99)
            opac_mask = opac_mask0 * 0.01 + opac_mask1
            loss_opac = (torch.exp(-(opac_ - 0.5)**2 * 20) * opac_mask).mean()
            loss+=loss_opac
        
        # Scaling Loss
        ema_nan_num_for_log = torch.sum(~(render_pkg["nan_valid"].detach()))
        if visibility_filter.sum() > 0:
            scale = gaussians.get_scaling[render_pkg["nan_valid"]][visibility_filter]
            sorted_scale, _ = torch.sort(scale, dim=-1)
            min_scale_loss = sorted_scale[...,0]
            loss += 100.0*min_scale_loss.mean()

        # single-view loss adopted from PGSR
        if iteration > opt.single_view_weight_from_iter:
            weight = opt.single_view_weight
            normal = render_pkg["rendered_normal"]
            depth_normal = render_pkg["depth_normal"] # depth normal是利用渲染深度重新计算得来的局部法向量
            with torch.no_grad():
                image_weight = (1.0 - get_img_grad_weight(depth_normal.detach().clone()))
                image_weight = (image_weight).clamp(0,1).detach() ** 5
                image_weight = erode(image_weight[None,None]).squeeze()

            if iteration < args.plane_normal:
                normal_loss = weight * (image_weight * (((depth_normal - normal)).abs().sum(0))).mean()
            else:
                normal_loss = weight * ((image_weight * ((depth_normal - normal).abs().sum(0)))[~(mask_xpd==0.0)]).mean()
            loss += weight_normal_l*(normal_loss)

        # multi-view loss adopted from PGSR
        if iteration > opt.multi_view_weight_from_iter:
            # geo_loss用于渲染深度跨视图几何约束 会传递到法向量、distance和高斯属性
            nearest_cam = None if len(viewpoint_cam.nearest_id) == 0 else scene.getTrainCameras()[random.sample(viewpoint_cam.nearest_id,1)[0]]
            use_virtul_cam = False
            if opt.use_virtul_cam and (np.random.random() < opt.virtul_cam_prob or nearest_cam is None): # 意思是虚拟相机也不是每次都会采用 有一定概率的
                nearest_cam = gen_virtul_cam(viewpoint_cam, trans_noise=dataset.multi_view_max_dis, deg_noise=dataset.multi_view_max_angle)
                # 如果在训练的过程中没有最近帧的话可以在当前帧附近创建一帧虚拟帧位置来协助进行跨视角约束损失计算
                use_virtul_cam = True
            if nearest_cam is not None:

                _, _, _, mask_normal_com_near = nearest_cam.get_image()
                if mask_normal_com_near is not None:
                    mask_xpd_near = mask_normal_com_near[...,-1]
                else:
                    mask_xpd_near = None

                patch_size = opt.multi_view_patch_size # patch_size和sample_num都是用来计算跨视图光度损失的
                sample_num = opt.multi_view_sample_num
                pixel_noise_th = opt.multi_view_pixel_noise_th
                total_patch_size = (patch_size * 2 + 1) ** 2 # patch是一个正方形块
                ncc_weight = opt.multi_view_ncc_weight
                geo_weight = opt.multi_view_geo_weight
                ## compute geometry consistency mask and loss
                H, W = render_pkg['plane_depth'].squeeze().shape # plane_depth是最终渲染得到的深度图像吗
                ix, iy = torch.meshgrid(
                    torch.arange(W), torch.arange(H), indexing='xy')
                pixels = torch.stack([ix, iy], dim=-1).float().to(render_pkg['plane_depth'].device)

                nearest_render_pkg = render(nearest_cam, gaussians, pipe, bg, app_model=app_model,
                                            return_plane=True, return_depth_normal=False)

                # 1.从当前帧获取世界坐标系下点云坐标 2.将世界坐标系变换到邻居坐标系 3.判断有哪些点云在邻居图像中找到了对应返回mask
                pts = gaussians.get_points_from_depth(viewpoint_cam, render_pkg['plane_depth'])
                # 不只考虑mask内的区域 这样没有意义 而是考虑凡是存在深度的区域更合理些
                depth_valid_cam = ((render_pkg['plane_depth'] > 0.1) & (render_pkg['plane_depth'] < viewpoint_cam.max_depth)).squeeze().reshape(-1)
                pts = pts[depth_valid_cam]
                pixels = pixels.reshape(-1,2)[depth_valid_cam]
                pts_in_nearest_cam = pts @ nearest_cam.world_view_transform[:3,:3] + nearest_cam.world_view_transform[3,:3]

                # mask_xpd_near不全为0的时候计算才比较保险
                map_z, d_mask, mask_pts_near = gaussians.get_points_depth_in_depth_map(nearest_cam, nearest_render_pkg['plane_depth'], pts_in_nearest_cam)
                # 1.邻居坐标系下点云坐标归一化 2.使用邻居图像深度信息投影到新的相机坐标系点云点 3.将该点云点重新变换到世界坐标系
                # 4.新世界坐标系点变换到参考坐标系 5.将点云投影回参考图像并使用有效深度mask和像素噪声阈值进行过滤
                pts_in_nearest_cam = pts_in_nearest_cam / (pts_in_nearest_cam[:,2:3])
                pts_in_nearest_cam = pts_in_nearest_cam * map_z.squeeze()[...,None]
                R = torch.tensor(nearest_cam.R).float().cuda()
                T = torch.tensor(nearest_cam.T).float().cuda()
                pts_ = (pts_in_nearest_cam-T)@R.transpose(-1,-2)
                pts_in_view_cam = pts_ @ viewpoint_cam.world_view_transform[:3,:3] + viewpoint_cam.world_view_transform[3,:3]
                pts_projections = torch.stack(
                            [pts_in_view_cam[:,0] * viewpoint_cam.Fx / pts_in_view_cam[:,2] + viewpoint_cam.Cx,
                            pts_in_view_cam[:,1] * viewpoint_cam.Fy / pts_in_view_cam[:,2] + viewpoint_cam.Cy], -1).float()
                pixel_noise = torch.norm(pts_projections - pixels.reshape(*pts_projections.shape), dim=-1)
                # 这里的d_mask尺寸已经是经过depth_valid_cam过滤之后的长度了
                d_mask = d_mask & (pixel_noise < pixel_noise_th) & (map_z.squeeze() > 0) & (map_z.squeeze() < nearest_cam.max_depth)
                # print(f"d_mask:{d_mask.shape}")

                # 6.得到的weights图像其实是重重投影误差指数倒数图像 其中重投影失败和重重投影失败的像素权值置为0表示误差无穷大
                # 我现在猜他这个nearest_cam应该就是针对某一帧在众多帧中挑选位姿最接近的哪一帧吧
                weights = (1.0 / torch.exp(pixel_noise)).detach()
                weights[~d_mask] = 0
                if iteration % 200 == 0:
                    # 真值图像可视化（这里是rgb图像，跟本工作无关）
                    gt_img_show = ((gt_image).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                    if 'app_image' in render_pkg:
                        img_show = ((render_pkg['app_image']).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                    else:
                        img_show = ((image).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                    # 监督平面mask实例颜色可视化
                    mask_color = ((mask_xpd_color).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                    # 渲染distance图像可视化
                    distance=render_pkg["rendered_distance"].squeeze()
                    distance = distance.detach().cpu().numpy()
                    distance_i = (distance - distance.min()) / (distance.max() - distance.min() + 1e-20)
                    distance_i = (distance_i * 255).clip(0, 255).astype(np.uint8)
                    distance_color = cv2.applyColorMap(distance_i, cv2.COLORMAP_JET)
                    # 渲染normal图像与深度渲染depth_normal图像可视化
                    normal_show = (((normal+1.0)*0.5).permute(1,2,0).clamp(0,1)*255).detach().cpu().numpy().astype(np.uint8)
                    depth_normal_show = (((depth_normal+1.0)*0.5).permute(1,2,0).clamp(0,1)*255).detach().cpu().numpy().astype(np.uint8)
                    mask_normal_show = (((mask_normal+1.0)*0.5).clamp(0,1)*255).detach().cpu().numpy().astype(np.uint8)
                    # 渲染深度图像可视化
                    depth = render_pkg['plane_depth'].squeeze().detach().cpu().numpy()
                    depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
                    depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
                    depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET) # 这里绘制的depth_color是归一化之后的甚至
                    row0 = np.concatenate([mask_color, img_show, normal_show], axis=1)
                    row1 = np.concatenate([distance_color, depth_color, depth_normal_show], axis=1)
                    # row1 = np.concatenate([distance_color, depth_color, mask_normal_show], axis=1)
                    image_to_show = np.concatenate([row0, row1], axis=0)
                    cv2.imwrite(os.path.join(debug_path, "%05d"%iteration + "_" + viewpoint_cam.image_name + ".jpg"), image_to_show)

                if d_mask.sum() > 0:
                    geo_loss = geo_weight * ((weights * pixel_noise)[d_mask]).mean() # geo_loss是算的图像上所有有效损失的平均值
                    loss += geo_loss
                    if use_virtul_cam is False:
                        with torch.no_grad():
                            ## sample mask 为啥ncc这部分的投影过程要torch.no_grad()呢？
                            d_mask = d_mask.reshape(-1)
                            ori_valid_indices = torch.arange(depth_valid_cam.shape[0], device=d_mask.device)[depth_valid_cam][d_mask]
                            valid_indices = torch.arange(d_mask.shape[0], device=d_mask.device)[d_mask]
                            if d_mask.sum() > sample_num:
                                index = np.random.choice(d_mask.sum().cpu().numpy(), sample_num, replace = False)
                                ori_valid_indices = ori_valid_indices[index]
                                valid_indices = valid_indices[index]

                            weights = weights.reshape(-1)[valid_indices] # 这里的weights还是重重投影过程中误差的指数倒数
                            ## sample ref frame patch
                            pixels = pixels.reshape(-1,2)[valid_indices]
                            # 不使用patch like ncc loss 而是直接针对渲染颜色使用mse损失呢
                            H, W = gt_image_gray.squeeze().shape
                            ori_pixels_patch = pixels.reshape(-1,1,2) # [pixel_num, 1, 2]
                            pixels_patch = ori_pixels_patch.clone()
                            pixels_patch[:, :, 0] = 2 * pixels_patch[:, :, 0] / (W - 1) - 1.0
                            pixels_patch[:, :, 1] = 2 * pixels_patch[:, :, 1] / (H - 1) - 1.0
                            # print(pixels_patch.view(1, -1, 1, 2).shape)
                            ref_mask_val = F.grid_sample(render_pkg["render"].unsqueeze(0), pixels_patch.view(1, -1, 1, 2),align_corners=True)
                            ref_mask_val = ref_mask_val.reshape(-1, 3)
                            
                            # 这个是参考视角变换到邻居视角的变换矩阵 world_view_transform矩阵本来就是世界坐标系变换到相机坐标系下的变换
                            ref_to_neareast_r = nearest_cam.world_view_transform[:3,:3].transpose(-1,-2) @ viewpoint_cam.world_view_transform[:3,:3]
                            ref_to_neareast_t = -ref_to_neareast_r @ viewpoint_cam.world_view_transform[3,:3] + nearest_cam.world_view_transform[3,:3]

                        ## compute Homography ★为什么这里的法向量要把第一维度放到最后呀 [C,H,W] to [H,W,C]
                        ref_local_n = render_pkg["rendered_normal"].permute(1,2,0)
                        ref_local_n = ref_local_n.reshape(-1,3)[ori_valid_indices] # [H*W,3] to [valid,3]

                        ref_local_d = render_pkg['rendered_distance'].squeeze()
                        ref_local_d = ref_local_d.reshape(-1)[ori_valid_indices] # ref_local_n 怎么expand到 ref_local_d的第一维度呢 ref_local_d.shape[0]一定相等吗
                        
                        H_ref_to_neareast = ref_to_neareast_r[None] - \
                            torch.matmul(ref_to_neareast_t[None,:,None].expand(ref_local_d.shape[0],3,1), 
                                        ref_local_n[:,:,None].expand(ref_local_d.shape[0],3,1).permute(0, 2, 1))/ref_local_d[...,None,None]
                        H_ref_to_neareast = torch.matmul(nearest_cam.get_k(nearest_cam.ncc_scale)[None].expand(ref_local_d.shape[0], 3, 3), H_ref_to_neareast)
                        H_ref_to_neareast = H_ref_to_neareast @ viewpoint_cam.get_inv_k(viewpoint_cam.ncc_scale)
                        
                        ## compute neareast frame patch # 利用计算得到的Hrn来计算邻居帧上的patch位置
                        grid = patch_warp(H_ref_to_neareast.reshape(-1,3,3), ori_pixels_patch)
                        grid[:, :, 0] = 2 * grid[:, :, 0] / (W - 1) - 1.0
                        grid[:, :, 1] = 2 * grid[:, :, 1] / (H - 1) - 1.0
                        _, nearest_image_gray, _ ,_= nearest_cam.get_image()
                        sampled_mask_val = F.grid_sample(nearest_render_pkg['render'].unsqueeze(0), grid.reshape(1, -1, 1, 2), align_corners=True)
                        sampled_mask_val = sampled_mask_val.reshape(-1, 3)

                        if torch.isnan(ref_mask_val).any() or torch.isnan(sampled_mask_val).any():
                            print("nan values found in multi-view photometric loss")
                        
                        mask_color_diff = torch.abs((ref_mask_val - sampled_mask_val)).mean(dim=-1)
                        mask_color_loss = (weights * mask_color_diff).mean()
                        loss += ncc_weight * mask_color_loss

        loss.backward()
        iter_end.record() # 这一个迭代运行结束了，GPU的真实运行时间

        with torch.no_grad():
            # Progress bar 为啥要每次0.4 0.6比例加权输出log
            ema_loss_for_log = 0.4 * image_loss.item() + 0.6 * ema_loss_for_log
            ema_single_view_for_log = 0.4 * normal_loss.item() if normal_loss is not None else 0.0 + 0.6 * ema_single_view_for_log
            ema_multi_view_geo_for_log = 0.4 * geo_loss.item() if geo_loss is not None else 0.0 + 0.6 * ema_multi_view_geo_for_log
            ema_multi_view_pho_for_log = 0.4 * ncc_loss.item() if ncc_loss is not None else 0.0 + 0.6 * ema_multi_view_pho_for_log
            ema_normal_loss_for_log = 0.4 * loss_mask_normal.item() if loss_mask_normal is not None else 0.0 + 0.6 * ema_normal_loss_for_log
            ema_distance_loss_for_log = 0.4 * loss_mask_distance.item() if loss_mask_distance is not None else 0.0 + 0.6 * ema_distance_loss_for_log
            ema_alpha_loss_for_log = 0.4 * loss_mask_alpha.item() if loss_mask_alpha is not None else 0.0 + 0.6 * ema_alpha_loss_for_log
            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "Single": f"{ema_single_view_for_log:.{5}f}", 
                    "Nor": f"{ema_normal_loss_for_log:.{5}f}",
                    "Dis": f"{ema_distance_loss_for_log:.{5}f}",
                    "NaN": f"{ema_nan_num_for_log}"
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save 保存日志 但是实际的时候应该没有用？应该不是，这一步应该输出了PSNR等数据 然后将当前训练的高斯保存到了本地路径中
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), app_model)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration) # 保存高斯到本地

            # Optimizer step
            if iteration < opt.iterations: # 需要优化的对象也不过就是作为场景表示的gaussian和曝光辐射的app_model
                gaussians.optimizer.step()
                app_model.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                app_model.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                app_model.save_weights(scene.model_path, iteration)

            if iteration % 300 == 0:
                torch.cuda.empty_cache() # 为什么每隔500迭代清空一下cuda的缓存 PGSR这类方法是不是批量式的方法啊，每次迭代用来监督优化的帧有几个？
    
    torch.cuda.empty_cache()

def prepare_output_and_logger(args):    
    # 保存本次运行的所有arguments
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

        
    # Set up output folder
    # print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, app_model):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    out = renderFunc(viewpoint, scene.gaussians, *renderArgs, app_model=app_model)
                    image = out["render"]
                    if 'app_image' in out:
                        image = out['app_image']
                    image = torch.clamp(image, 0.0, 1.0)
                    gt_image, _, _ ,_= viewpoint.get_image()
                    gt_image = torch.clamp(gt_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def list_of_ints(arg):
    return np.array(arg.split(',')).astype(int)

def debug_logs(args):
    sections = {
        "Gaussian Attribute Fixed Flags": [
            ("Fix XYZ:", args.fix_xyz),
            ("Fix Normal:", args.fix_normal),
            ("Fix Opacity:", args.fix_opa),
            ("Fix RGB:", args.fix_rgb),
        ],
        "Pipeline Basic Configuration":[
            ("Use Mask:", args.use_mask),
            ("Mask Type:", args.mask_type)
        ],
        "Loss Function Configuration": [
            ("RGB Loss:", not args.no_rgb),
            ("Depth Loss:", args.use_depth),
            ("Opacity Loss:", args.use_alpha),
            ("Scaling Loss:", True),
            ("Normal Loss:", not args.no_normal_p),
            ("Distance Loss:", not args.no_distance),
        ]
    }

    def print_section(title, items):
        print(f"[{title}]")
        for label, value in items:
            if type(value) == str:
                print(f" > {label:<15} {value}")
                continue
            symbol = "\033[92m√\033[0m" if value else "\033[91mx\033[0m"
            print(f" > {label:<15} {symbol}")
        print("-" * 35)

    for title, items in sections.items():
        print_section(title, items)


if __name__ == "__main__":
    torch.set_num_threads(8)
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6007)
    parser.add_argument('--debug_from', type=int, default=-100)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[20000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1, 5000, 9000, 10000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument('--init_w_gaussian', action='store_true', default=False)
    parser.add_argument('--voxel_size', type=float, default=0.1)
    
    # for basic normal and distance regularization
    parser.add_argument('--fix_xyz', action='store_true', default=False)
    parser.add_argument('--fix_normal', action='store_true', default=False)
    parser.add_argument('--fix_opa', action='store_true', default=False)
    parser.add_argument('--fix_rgb', action='store_true', default=True)

    parser.add_argument('--use_mask', action='store_true', default=True)
    parser.add_argument('--use_opa', action='store_true', default=False)
    parser.add_argument('--use_alpha', action='store_true', default=True)
    parser.add_argument('--use_depth', action='store_true', default=True)

    parser.add_argument('--plane_normal', type=int, default=0)
    parser.add_argument('--plane_distance', type=int, default=0)
    parser.add_argument('--mask_type', type=str, default='fusion')
    parser.add_argument('--no_rgb', action='store_true', default=False)

    # pigs ablation parameters
    parser.add_argument('--increment', action='store_true', default=False)
    parser.add_argument('--use_temp', action='store_true', default=False)
    parser.add_argument('--no_distance', action='store_true', default=False)
    parser.add_argument('--no_normal_p', action='store_true', default=False)
    parser.add_argument('--no_initial', action='store_true', default=False)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)
    print(f"preload_img: {args.preload_img}")

    # Initialize system state (RNG)
    safe_state(args.quiet)
    debug_logs(args)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.voxel_size, args.init_w_gaussian, args)

    # All done
    print("\nTraining complete.")