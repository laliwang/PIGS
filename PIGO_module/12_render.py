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
from scene import Scene
import os
import json
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import cv2
import open3d as o3d
from scene.app_model import AppModel
import copy
import shutil
from collections import deque
from simple_knn._C import distCUDA2

def rebuild_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    if not os.path.exists(path):
        os.makedirs(path)

def normal_2_gray(depth_normal, pose):
    H, W = depth_normal.shape[:2]
    depth_normal_local = depth_normal.reshape(-1,3)
    depth_normal_world = depth_normal_local @ pose.transpose(-1,-2) # 局部坐标系到世界坐标系
    depth_normal_world = ((depth_normal_world+1)*127.5).clip(0.0, 255.0) # 转为0-255区间的rgb图像 shape (H*W, 3)
    depth_normal_world = depth_normal_world.reshape(H, W, 3)
    depth_normal_gray = 0.299 * depth_normal_world[...,0] + 0.587 * depth_normal_world[...,1] + 0.114 * depth_normal_world[...,2]
    depth_normal_gray = (depth_normal_gray / 255.0).unsqueeze(dim=-1).permute(2, 0, 1)
    return depth_normal_gray.clamp(0.0, 1.0).to(depth_normal.device)

def clean_mesh(mesh, min_len=1000):
    # 其实就是clean了一下mesh得到了post的结果 如果网格中的mesh表面簇三角形数量小于1000那么删除这个簇 这个后处理的操作是不是能让gof的mesh变干净？ 答案是否定的
    # gof方法提取pgsr模型得到的mesh不知为何有很大的异常面片
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (mesh.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < min_len
    mesh_0 = copy.deepcopy(mesh)
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    return mesh_0

def render_set(model_path, name, iteration, views, scene, gaussians, pipeline, background, gaussians_temp,
               app_model=None, max_depth=5.0, volume=None, use_depth_filter=False, not_pgsr=False, only_mesh=False, use_mask=False):

    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    # render_depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_depth")
    render_depth16_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_depth16")
    render_normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_normal")
    # render_distance_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_distance")

    makedirs(render_path, exist_ok=True)
    # makedirs(render_depth_path, exist_ok=True)
    makedirs(render_depth16_path, exist_ok=True)
    makedirs(render_normal_path, exist_ok=True)
    # makedirs(render_distance_path, exist_ok=True)
    print("gaussians.ids 的形状:", gaussians.ids.shape)
   
    depths_tsdf_fusion = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        gt, gt_image_gray, depth_ob ,mask_normal_com= view.get_image()

        if mask_normal_com is not None:
            im_width = mask_normal_com.shape[0]
            im_height = mask_normal_com.shape[1]
            mask_normal = mask_normal_com[...,:-1]
            mask_xpd = mask_normal_com[...,-1]
            mask_binary = ~(mask_xpd==0.0)
            mask_xpd_copy = mask_xpd.clone()
            unique_values = torch.unique(mask_xpd)
        else:
            mask_xpd = None
            mask_binary = None

        # 从零初始化渲染内容
        depth_id = torch.zeros(im_width,im_height).float().to(gaussians._xyz.device)
        color_id_d = torch.zeros(3,im_width,im_height).float().to(gaussians._xyz.device)
        normal_id = torch.zeros(3,im_width,im_height).float().to(gaussians._xyz.device)
        mask_cross_temp = mask_xpd-mask_xpd
        # 遍历所有平面掩码id进行对应渲染
        for value in unique_values:
            mask_temp = (gaussians.ids == value).squeeze(1)
            if  ~torch.any(mask_temp):
                    continue
            
            # 创建id对应的临时高斯模型
            gaussians_temp._xyz = gaussians._xyz[mask_temp]
            gaussians_temp._features_dc = gaussians._features_dc[mask_temp]
            gaussians_temp._features_rest = gaussians._features_rest[mask_temp]
            gaussians_temp._opacity = gaussians._opacity[mask_temp]
            gaussians_temp._scaling = gaussians._scaling[mask_temp]
            gaussians_temp._rotation = gaussians._rotation[mask_temp]
            gaussians_temp.active_sh_degree = gaussians.active_sh_degree
            gaussians_temp._scaling = gaussians._scaling[mask_temp]

            # 渲染id对应的临时高斯模型
            out_temp = render(view, gaussians_temp, pipeline, background, app_model=app_model)
            depth_temp = out_temp["plane_depth"].squeeze()
            distance_temp = out_temp["rendered_distance"].squeeze()
            rendering_temp = out_temp["render"]
            normal_temp = out_temp["rendered_normal"]
            alpha_temp = out_temp["rendered_alpha"]

            # 获取本id对应的二维掩码区域
            mask_id = mask_xpd==value

            # 在非本id区域令深度和距离置零
            depth_temp[~mask_id] = 0.0
            distance_temp[~mask_id] = 0.0

            # 对rgb图像约束的mask定义为在分割id内且要求渲染深度大于0.0
            mask_id_rgb = mask_id&(depth_temp>0.0)
            mask_rgb = mask_id_rgb.unsqueeze(0).expand(3,-1,-1)
            rendering_temp[~mask_rgb] = 0.0
            color_mask_neg = (rendering_temp ==0.0).all(dim=0)& ~torch.isnan(rendering_temp).any(dim=0)
            color_mask = ~color_mask_neg

            # 获取彩色id掩码渲染
            if color_mask.any().item():
                mean_color = rendering_temp[:, color_mask].mean(dim=1, keepdim=True)
            else:
                mean_color = torch.ones((3, 1), device=rendering_temp.device)*255
            rendering_temp[:, mask_id_rgb] = mean_color
            normal_temp[~mask_rgb] = 0.0
            
            # 在完整的渲染变量中增量加入临时渲染变量
            depth_id += depth_temp
            color_id_d += rendering_temp
            normal_id += normal_temp

            # 标记本次局部渲染区域，防止重复
            mask_id_temp = (depth_temp!=0.0)&(mask_xpd==value)
            mask_cross_temp[mask_id_temp] += 1.0

        mask_cross = mask_cross_temp != 1.0
        rgb_mask_cross = mask_cross.unsqueeze(0).expand(3,-1,-1)
        depth_id[mask_cross] = 0.0
        color_id_d[rgb_mask_cross] = 0.0
        normal_id[rgb_mask_cross] = 0.0



        out = render(view, gaussians, pipeline, background, app_model=app_model)
        alpha = out["rendered_alpha"].squeeze()

        #rendering=rendering.clamp(0.0, 1.0)
        rendering = color_id_d.clamp(0.0, 1.0)
        _, H, W = rendering.shape

        #depth = out["plane_depth"].squeeze() # depth的尺寸应该是(H,W)
        depth=depth_id
        #alpha = out["rendered_alpha"].squeeze()
        distance=out["rendered_distance"].squeeze()


        depth_tsdf = depth_id.clone()
        
        
        depth_render_16 = depth.clone().detach().cpu().numpy()
        depth_render_16 = (depth_render_16 * 1000.0).astype(np.uint16)
        depth = depth.detach().cpu().numpy()
        depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
        depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)
        #cv2.imwrite(os.path.join(render_depth_path, view.image_name + ".jpg"), depth_color)
        cv2.imwrite(os.path.join(render_depth16_path, view.image_name + ".png"), depth_render_16)

        distance[alpha!=0.0] = distance[alpha!=0.0] / alpha[alpha!=0.0]
        distance_np = distance.clone().detach().cpu().numpy().astype(np.float16)
        distance = distance.detach().cpu().numpy()
        distance_i = (distance - distance.min()) / (distance.max() - distance.min() + 1e-20)
        distance_i = (distance_i * 255).clip(0, 255).astype(np.uint8)
        distance_color = cv2.applyColorMap(distance_i, cv2.COLORMAP_JET)
        # cv2.imwrite(os.path.join(render_distance_path, view.image_name + ".jpg"), distance_color)
        # np.save(os.path.join(render_distance_path, view.image_name + ".npy"), distance_np)

        normal=normal_id.permute(1,2,0)
        normal = normal/(normal.norm(dim=-1, keepdim=True)+1.0e-8) # 归一化
        normal_np = normal.detach().cpu().numpy().astype(np.float16)
        normal = ((normal_np+1) * 127.5).astype(np.uint8).clip(0, 255)
        cv2.imwrite(os.path.join(render_normal_path, view.image_name + ".png"), normal)
        np.save(os.path.join(render_normal_path, view.image_name + ".npy"), normal_np)

        if name == 'test':
            # 利用torchvision保存图像和用opencv保存图像有什么区别呢
            torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name + ".png"))
        else:
            rendering_np = (rendering.permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
            cv2.imwrite(os.path.join(render_path, view.image_name + ".png"), rendering_np)
        
        if use_depth_filter:
            view_dir = torch.nn.functional.normalize(view.get_rays(), p=2, dim=-1)
            depth_normal = out["depth_normal"].permute(1,2,0)
            depth_normal = torch.nn.functional.normalize(depth_normal, p=2, dim=-1)
            dot = torch.sum(view_dir*depth_normal, dim=-1)
            angle = torch.acos(dot)
            mask = angle > (100.0 / 180 * 3.14159)
            depth_tsdf[mask] = 0 # depth_filter让射线法向量夹角超过一定角度的depth_tsdf赋0 不提取这些位置的mesh结果 效果反而变差。。
        depths_tsdf_fusion.append(depth_tsdf.squeeze()) # depths_tsdf_fusion用的是所有训练帧渲染深度图像depth_plane 在此基础上进行tsdf-fusion咯 合理
        
    if volume is not None:
        # depths_tsdf_fusion = torch.stack(depths_tsdf_fusion, dim=0) # 你tsdf-fusion的时候nearest camera有啥用嘞
        for idx, view in enumerate(tqdm(views, desc="TSDF Fusion progress")):
            ref_depth = depths_tsdf_fusion[idx]
            
            if use_depth_filter and len(view.nearest_id) > 2 and not pgsr:
                print("Using Depth Filter ... Using Nearest IDs ...")
                # nearest_id到底是在哪一步获取的，， 其实nearest_id的作用也就是让提取得到的depth_tsdf图像上不满足跨视图约束的点云不参与tsdf-fusion过程
                nearest_world_view_transforms = scene.world_view_transforms[view.nearest_id]
                num_n = nearest_world_view_transforms.shape[0]
                ## compute geometry consistency mask
                H, W = ref_depth.squeeze().shape

                ix, iy = torch.meshgrid(
                    torch.arange(W), torch.arange(H), indexing='xy')
                pixels = torch.stack([ix, iy], dim=-1).float().to(out['plane_depth'].device)

                pts = gaussians.get_points_from_depth(view, ref_depth)
                pts_in_nearest_cam = torch.matmul(nearest_world_view_transforms[:,None,:3,:3].expand(num_n,H*W,3,3).transpose(-1,-2), 
                                                  pts[None,:,:,None].expand(num_n,H*W,3,1))[...,0] + nearest_world_view_transforms[:,None,3,:3] # b, pts, 3

                depths_nearest = depths_tsdf_fusion[view.nearest_id][:,None]
                pts_projections = torch.stack(
                                [pts_in_nearest_cam[...,0] * view.Fx / pts_in_nearest_cam[...,2] + view.Cx,
                                pts_in_nearest_cam[...,1] * view.Fy / pts_in_nearest_cam[...,2] + view.Cy], -1).float()
                d_mask = (pts_projections[..., 0] > 0) & (pts_projections[..., 0] < view.image_width) &\
                         (pts_projections[..., 1] > 0) & (pts_projections[..., 1] < view.image_height)
                # d_mask是ref_view 到 nearest_view 的重投影图像mask

                pts_projections[..., 0] /= ((view.image_width - 1) / 2)
                pts_projections[..., 1] /= ((view.image_height - 1) / 2)
                pts_projections -= 1
                pts_projections = pts_projections.view(num_n, -1, 1, 2)
                # 重投影点直接使用grid_sample在nearest_view的plane_depth上面找的新的深度值
                map_z = torch.nn.functional.grid_sample(input=depths_nearest,
                                                        grid=pts_projections,
                                                        mode='bilinear',
                                                        padding_mode='border',
                                                        align_corners=True
                                                        )[:,0,:,0]
                
                pts_in_nearest_cam[...,0] = pts_in_nearest_cam[...,0]/pts_in_nearest_cam[...,2]*map_z.squeeze()
                pts_in_nearest_cam[...,1] = pts_in_nearest_cam[...,1]/pts_in_nearest_cam[...,2]*map_z.squeeze()
                pts_in_nearest_cam[...,2] = map_z.squeeze()
                pts_ = (pts_in_nearest_cam-nearest_world_view_transforms[:,None,3,:3])
                pts_ = torch.matmul(nearest_world_view_transforms[:,None,:3,:3].expand(num_n,H*W,3,3), 
                                    pts_[:,:,:,None].expand(num_n,H*W,3,1))[...,0]

                pts_in_view_cam = pts_ @ view.world_view_transform[:3,:3] + view.world_view_transform[None,None,3,:3]
                pts_projections = torch.stack(
                            [pts_in_view_cam[...,0] * view.Fx / pts_in_view_cam[...,2] + view.Cx,
                            pts_in_view_cam[...,1] * view.Fy / pts_in_view_cam[...,2] + view.Cy], -1).float()
                pixel_noise = torch.norm(pts_projections.reshape(num_n, H, W, 2) - pixels[None], dim=-1) # 重重投影误差 但是这里的误差是用depth_plane计算的
                d_mask_all = d_mask.reshape(num_n,H,W) & (pixel_noise < 1.0) & (pts_in_view_cam[...,2].reshape(num_n,H,W) > 0.1)
                d_mask_all = (d_mask_all.sum(0) > 1)
                ref_depth[~d_mask_all] = 0

            ref_depth[ref_depth>max_depth] = 0
            ref_depth = ref_depth.detach().cpu().numpy()
            
            pose = np.identity(4)
            pose[:3,:3] = view.R.transpose(-1,-2)
            pose[:3, 3] = view.T
            color = o3d.io.read_image(os.path.join(render_path, view.image_name + ".png"))

            # 生成深度图像（单位转换为毫米并转为uint16）
            depth = o3d.geometry.Image((ref_depth * 1000).astype(np.uint16))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color, depth, depth_scale=1000.0, depth_trunc=max_depth, convert_rgb_to_intensity=False)
            volume.integrate(
                rgbd,
                o3d.camera.PinholeCameraIntrinsic(W, H, view.Fx, view.Fy, view.Cx, view.Cy),
                pose) # 使用渲染的rgbd图像执行简单的tsdf fusion获取的 原理非常简单 那rtg-slam的3dgs表示可以吗？

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool,
                 max_depth : float, voxel_size : float, use_depth_filter : bool, not_pgsr : bool, args):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        gaussians_temp=GaussianModel(dataset.sh_degree)
        hive_dict = {}
        hive_dict['fix_xyz'] = False
        hive_dict['use_mask'] = args.use_mask
        hive_dict['mask_type'] = args.mask_type
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, not_pgsr = not_pgsr, hive_dict=hive_dict)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        print(f"TSDF voxel_size {voxel_size}")
        # 这一步利用open3d内置TSDF-Fusion功能开始提取mesh面
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=4.0*voxel_size,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

        if not skip_train:
            # 核心代码是render_set里头的
            render_set(dataset.model_path, "train_pigo", scene.loaded_iter, scene.getTrainCameras(), scene, gaussians, pipeline, background, gaussians_temp,
                       max_depth=max_depth, volume=volume, use_depth_filter=use_depth_filter, not_pgsr=not_pgsr, only_mesh=args.only_mesh, use_mask=args.use_mask)
            print(f"extract_triangle_mesh")
            mesh = volume.extract_triangle_mesh()

            path = os.path.join(dataset.model_path, "mesh")
            os.makedirs(path, exist_ok=True)
            
            o3d.io.write_triangle_mesh(os.path.join(path, "tsdf_fusion.ply"), mesh, 
                                       write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)
            mesh = clean_mesh(mesh)
            mesh.remove_unreferenced_vertices()
            mesh.remove_degenerate_triangles()
            o3d.io.write_triangle_mesh(os.path.join(path, "tsdf_fusion_post.ply"), mesh, 
                                       write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)

        if not skip_test:
            render_set(dataset.model_path, "test_pigo", scene.loaded_iter, scene.getTestCameras(), scene, gaussians, pipeline, background,gaussians_temp)

if __name__ == "__main__":
    torch.set_num_threads(8)
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--max_depth", default=5.0, type=float)
    parser.add_argument("--voxel_size", default=0.002, type=float)
    parser.add_argument("--use_depth_filter", action="store_true") # 从使用经历来看使用use_depth_filter好像会让结果变得有点差，，
    parser.add_argument("--not_pgsr", action="store_true")
    parser.add_argument("--only_mesh", action="store_true")
    parser.add_argument("--use_mask", action="store_true", default=True)
    parser.add_argument("--mask_type", type=str, default='fusion')

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.max_depth, args.voxel_size, args.use_depth_filter, args.not_pgsr, args)