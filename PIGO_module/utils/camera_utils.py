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

from scene.cameras import Camera, DepthCamera
import numpy as np
from utils.graphics_utils import fov2focal
import sys
import open3d as o3d
from PIL import Image
import cv2
import torch

WARNED = False

def loadCam(args, id, cam_info, resolution_scale, use_depth=False,use_mask=False):
    orig_w, orig_h = cam_info.width, cam_info.height
    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global_down = orig_w / 1600
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    print(f"scale {float(global_down) * float(resolution_scale)}")
                    WARNED = True
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    sys.stdout.write('\r')
    sys.stdout.write("load camera {}".format(id))
    sys.stdout.flush()
    if use_mask:
        #print(cam_info.mask_xpd_path)已验证此处成功进入判断，cam_info.mask_xpd_path正确
        return DepthCamera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, mat=cam_info.mat,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  image_width=resolution[0], image_height=resolution[1],
                  image_path=cam_info.image_path,
                  image_name=cam_info.image_name,
                  depth_path=cam_info.depth_path,
                  depth_name=cam_info.depth_name,uid=id,
                  mask_xpd_path=cam_info.mask_xpd_path,
                  mask_xpd_name=cam_info.mask_xpd_name,
                  depth_scale=cam_info.depth_scale,
                  max_depth=cam_info.max_depth, 
                  preload_img=args.preload_img, 
                  ncc_scale=args.ncc_scale,
                  data_device=args.data_device)

    elif use_depth:
        return DepthCamera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, mat=cam_info.mat,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  image_width=resolution[0], image_height=resolution[1],
                  image_path=cam_info.image_path,
                  image_name=cam_info.image_name,
                  depth_path=cam_info.depth_path,
                  depth_name=cam_info.depth_name,uid=id,
                  depth_scale=cam_info.depth_scale,
                  max_depth=cam_info.max_depth, 
                  preload_img=args.preload_img, 
                  ncc_scale=args.ncc_scale,
                  data_device=args.data_device,
                  mask_xpd_path=None,
                  mask_xpd_name=None)
    else:
        return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                    FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                    image_width=resolution[0], image_height=resolution[1],
                    image_path=cam_info.image_path,
                    image_name=cam_info.image_name, uid=id, 
                    preload_img=args.preload_img, 
                    ncc_scale=args.ncc_scale,
                    data_device=args.data_device)

def cameraList_from_camInfos(cam_infos, resolution_scale, args, use_depth=False,use_mask=False):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale, use_depth,use_mask))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry

def camera_dense_pcd(viewpoint_cam, downsample_size=0.03, down_flag=True):
    gt_depth = cv2.imread(viewpoint_cam.depth_path, -1)
    gt_K = viewpoint_cam.get_k()
    o3d_depth = o3d.geometry.Image(gt_depth)
    o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(viewpoint_cam.image_width, viewpoint_cam.image_height, gt_K[0][0], gt_K[1][1], gt_K[0][2], gt_K[1][2])
    o3d_pc = o3d.geometry.PointCloud.create_from_depth_image(
    depth=o3d_depth,
    intrinsic=o3d_intrinsic,
    extrinsic=np.identity(4),
    depth_scale=viewpoint_cam.depth_scale,  # 根据实际情况调整
    depth_trunc=viewpoint_cam.max_depth,)
    o3d_pc = o3d_pc.transform(viewpoint_cam.mat)
    if down_flag:
        o3d_pc = o3d_pc.voxel_down_sample(voxel_size=downsample_size)
    pc_points = np.array(o3d_pc.points)
    return pc_points

def create_free_pc(cam_pos, pc, near_bound=0.5, far_bound=0.5, margin=0.05, sample_res=0.1, perturb=0, device='cpu'):
    origin = cam_pos.view(1,-1).to(device)
    dist = torch.norm(pc-origin, dim=1)
    dir = (pc-origin) / dist.view(-1,1)
        
    start_points = torch.maximum(torch.zeros_like(dist), dist-near_bound)
    end_points = dist+far_bound
    num_points = (far_bound+near_bound) / sample_res
    
    # Generate linearly spaced points without using torch.linspace
    weights = torch.arange(num_points, dtype=torch.float32) / num_points #(num_points - 1)
    weights = weights.view(1, -1).to(device)
    
    sampled_dist = start_points.view(-1, 1) + weights * (end_points - start_points).view(-1, 1)
    
    dist_perturb = perturb * (torch.rand(sampled_dist.shape).to(device) - 0.5)
    sampled_dist += dist_perturb
    
    valid_mask = (sampled_dist<(dist-margin).unsqueeze(-1)) + (sampled_dist>(dist+margin).unsqueeze(-1))
    
    # sampled_points = sampled_dist.unsqueeze(2)[:,...] @ dir.unsqueeze(1)[:,...] # slow
    sampled_points = sampled_dist.unsqueeze(2) * dir.unsqueeze(1) # fast
    sampled_points = sampled_points[valid_mask]
    
    sampled_points += origin
    return sampled_points

def compute_grids(gaussian_xyz, occ_pc, free_pc, grid_size=0.2, perturb=0):

    xyz = torch.cat((gaussian_xyz, occ_pc, free_pc), dim=0)

    xyz_offset_perturb = 0.5*(torch.rand(3)).cuda() * perturb
    xyz_offset = xyz.min(dim=0)[0] - xyz_offset_perturb
    xyz_norm = xyz - xyz_offset
    grid_dim_idxs = torch.floor(xyz_norm / grid_size).long()
    n_cells_per_dim = torch.max(grid_dim_idxs, dim=0)[0] + 1

    grid_indices = grid_dim_idxs[:,2]*(n_cells_per_dim[0]*n_cells_per_dim[1]) \
                + grid_dim_idxs[:,1]*n_cells_per_dim[0] \
                + grid_dim_idxs[:,0]
                
    unique_indices, inverse_indices = grid_indices.unique(return_inverse=True)
    mapping_tensor = torch.arange(unique_indices.size(0)).to(grid_indices.device)
    grid_indices = mapping_tensor[inverse_indices]
    return grid_indices[:len(gaussian_xyz)], grid_indices[len(gaussian_xyz):len(gaussian_xyz)+len(occ_pc)], grid_indices[len(gaussian_xyz)+len(occ_pc):len(gaussian_xyz)+len(occ_pc)+len(free_pc)]


def query_gaussians(query_xyz, mvn, alpha, device="cpu", chunk_size=50000):
    probs = []
    for query_xyz_ in torch.split(query_xyz.to(torch.float), chunk_size, dim=0):
        individual_probs_ = torch.exp(mvn.log_prob(query_xyz_.view(query_xyz_.shape[0], 1, -1)))
        individual_probs_ = individual_probs_ * alpha.view(1,-1)
        probs_ = torch.sum(individual_probs_, axis=1)
        probs_ = torch.clamp(probs_, 0., 1.)
        probs.append(probs_)
        
    probs = torch.cat(probs, dim=0)

    return probs