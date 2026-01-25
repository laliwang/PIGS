import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm, trange
from natsort import natsorted
from typing import NamedTuple, List, Dict
from loguru import logger
from pyhocon import ConfigFactory
from pyhocon import ConfigTree
import math
import argparse
import cv2

import sys
from pathlib import Path
current_dir = Path(__file__).resolve().parents[1]
sys.path.append(current_dir)
# from planarsplat
from planarsplat.utils import model_util
from planarsplat.utils.graphics_utils import focal2fov, getProjectionMatrix

def read_files(directory, endtxt):
    file_paths = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(endtxt)]
    file_list = natsorted(file_paths)
    return file_list

# step 1: 读取数据集，我只需要得到PlanarSplat所需的view_info_list即可
class ViewInfo(nn.Module):
    '''
    borrowed from PlanarSplatting
    '''
    def __init__(self, cam_info: Dict, gt_info: Dict):
        super().__init__()
        # get cam info
        self.intrinsic = cam_info['intrinsic'].cuda()
        self.pose = cam_info['pose'].cuda()
        self.raster_cam_w2c = cam_info['raster_cam_w2c'].cuda()
        self.raster_cam_proj = cam_info['raster_cam_proj'].cuda()
        self.raster_cam_fullproj = cam_info['raster_cam_fullproj'].cuda()
        self.raster_cam_center = cam_info['raster_cam_center'].cuda()
        self.raster_cam_FovX = cam_info['raster_cam_FovX'].cpu().item()
        self.raster_cam_FovY = cam_info['raster_cam_FovY'].cpu().item()
        self.tanfovx = math.tan(self.raster_cam_FovX  * 0.5)
        self.tanfovy = math.tan(self.raster_cam_FovY * 0.5)
        self.raster_img_center = cam_info['raster_img_center'].cuda()
        self.cam_loc = cam_info['cam_loc'].cuda()

        # get gt info
        self.rgb = gt_info['rgb'].cuda()
        self.mono_depth = gt_info['mono_depth'].cuda()
        self.mono_normal_local = gt_info['mono_normal_local'].cuda()
        self.mono_normal_global = gt_info['mono_normal_global'].cuda()
        self.index = gt_info['index']
        self.image_path = gt_info['image_path']
        self.mask_mvsa = gt_info['mask_mvsa'].cuda()

        # other info
        self.scale = 1.0
        self.shift = 0.0
        self.plane_depth = None


class ViewInfoBuilder:
    def __init__(self, data_folder, prior_folder, mvsa_folder, device="cuda"):
        self.data_folder = data_folder
        self.prior_folder = prior_folder
        self.device = device
        self.img_res = (480, 640)

        # paths
        self.image_dir = os.path.join(data_folder, 'color')
        self.mono_depth_dir = os.path.join(data_folder, 'depth_m3d')
        self.mono_normal_dir = os.path.join(prior_folder, 'normal_npy_m')
        self.mask_normal_dir = os.path.join(mvsa_folder, 'mask_normal_FFF_fusion')
        self.intrinsic_path = os.path.join(data_folder, 'intrinsic', 'intrinsic_depth.txt')
        self.poses_dir = os.path.join(data_folder, 'pose')
        self.mono_mesh_dest = os.path.join(data_folder, 'mesh', 'tsdf_fusion_SR_m3d.ply')
        self.mvsa_pts_path = os.path.join(data_folder, 'points3d.ply')
        self.color_path = os.path.join(mvsa_folder, 'object_pcd', 'inst_colors.npy')
        self.inst_num = np.loadtxt(self.color_path.replace('inst_colors.npy', 'inst_num.txt')).astype(np.int32).item()

        # file lists
        self.rgb_paths = read_files(self.image_dir, 'jpg')
        self.mono_depth_paths = read_files(self.mono_depth_dir, 'png')
        self.mono_normal_paths = read_files(self.mono_normal_dir, 'npy')
        self.mask_normal_paths = read_files(self.mask_normal_dir, 'npy')
        self.pose_paths = read_files(self.poses_dir, 'txt')

        self.n_images = len(self.rgb_paths)

        # load camera
        self.intrinsics_all, self.poses_all = self.load_cameras()
        # load rgbs
        self.rgbs = self.load_rgbs() # n, hw, 3
        # load mono depths
        self.mono_depths = self.load_mono_depths() # n, hw
        # load mono normals
        # self.mono_normals = self.load_mono_normals() # n, hw, 3

        # load mask normals which replace mono normals
        self.mono_normals, self.mask_mvsa = self.load_mask_normals() # (n, hw, 3) and (n, hw)

        # load mvsa mask rgb
        self.color_lut = torch.from_numpy(np.load(self.color_path)).float().cuda() # (500, 3)
        self.mask_rgbs = self.color_lut[self.mask_mvsa.long()] # (n, hw, 3)
        self.mask_rgbs[self.mask_mvsa == 0] = 0

        # get cam parameters for rasterization
        self.raster_cam_w2c_list, self.raster_cam_proj_list, self.raster_cam_fullproj_list, self.raster_cam_center_list, self.raster_cam_FovX_list, self.raster_cam_FovY_list, self.raster_img_center_list = self.get_raster_cameras(
            self.intrinsics_all, self.poses_all, self.img_res[0], self.img_res[1])
        
        # prepare view list
        self.view_info_list = []
        for idx in tqdm(range(self.n_images), desc="building view int list..."):
            cam_loc = self.poses_all[idx][:3, 3].clone() 
            cam_info = {
                "intrinsic": self.intrinsics_all[idx].clone(),
                "pose": self.poses_all[idx].clone(),  # camera to world
                "raster_cam_w2c": self.raster_cam_w2c_list[idx].clone(),
                "raster_cam_proj": self.raster_cam_proj_list[idx].clone(),
                "raster_cam_fullproj": self.raster_cam_fullproj_list[idx].clone(),
                "raster_cam_center": self.raster_cam_center_list[idx].clone(),
                "raster_cam_FovX": self.raster_cam_FovX_list[idx].clone(),
                "raster_cam_FovY": self.raster_cam_FovY_list[idx].clone(),
                "raster_img_center": self.raster_img_center_list[idx].clone(),
                "cam_loc": cam_loc.squeeze(0),
            }

            normal_local = self.mono_normals[idx].clone().cuda()
            normal_global = normal_local @ self.poses_all[idx][:3, :3].T

            gt_info = {
                # "rgb": self.rgbs[idx],
                "rgb": self.mask_rgbs[idx],
                "image_path": self.rgb_paths[idx],
                "mono_depth": self.mono_depths[idx],
                "mono_normal_global": normal_global,
                "mono_normal_local": normal_local,
                "mask_mvsa": self.mask_mvsa[idx],
                'index': idx
            }
            self.view_info_list.append(ViewInfo(cam_info, gt_info))
        
        logger.info('data loader finished')

    # -------------------------
    # camera
    # -------------------------
    def load_cameras(self):
        intrinsics_all, poses_all = [], []

        pbar = tqdm(range(self.n_images), desc="Loading cameras")
        for i in pbar:
            intrinsic_i = np.loadtxt(self.intrinsic_path)
            pose_i = np.loadtxt(self.pose_paths[i])

            intrinsics_all.append(
                torch.from_numpy(intrinsic_i).float().to(self.device)
            )
            poses_all.append(
                torch.from_numpy(pose_i).float().to(self.device)
            )

            pbar.set_postfix(idx=i)

        return intrinsics_all, poses_all

    # -------------------------
    # rgb
    # -------------------------
    def load_rgbs(self):
        rgbs_all = []

        pbar = tqdm(range(self.n_images), desc="Loading rgbs")
        for i in pbar:
            rgb_i = cv2.imread(self.rgb_paths[i])[:, :, ::-1]
            rgb_i = np.ascontiguousarray(rgb_i.transpose(2, 0, 1))
            rgb_i = torch.from_numpy(rgb_i).float().to(self.device) / 255.0
            rgbs_all.append(rgb_i)

        rgbs = torch.stack(rgbs_all, dim=0)          # n, 3, h, w
        rgbs = rgbs.reshape(self.n_images, 3, -1).permute(0, 2, 1)  # n, hw, 3
        return rgbs

    # -------------------------
    # mono depth
    # -------------------------
    def load_mono_depths(self):
        depths_all = []

        pbar = tqdm(range(self.n_images), desc="Loading mono_depths")
        for i in pbar:
            depth_i = cv2.imread(
                self.mono_depth_paths[i],
                cv2.IMREAD_UNCHANGED
            ).astype(np.float32) / 1000.0

            depths_all.append(
                torch.from_numpy(depth_i).float().to(self.device)
            )

        depths = torch.stack(depths_all, dim=0)      # n, h, w
        mono_depths = depths.reshape(self.n_images, -1)  # n, hw
        return mono_depths

    # -------------------------
    # mono normals
    # -------------------------
    def load_mono_normals(self):
        normals_all = []

        pbar = tqdm(range(self.n_images), desc="Loading mono_normals")
        for i in pbar:
            normal_i = np.load(self.mono_normal_paths[i])[:, :, :3]
            normal_i = np.ascontiguousarray(normal_i.transpose(2, 0, 1))

            normals_all.append(
                torch.from_numpy(normal_i).float().to(self.device)
            )

        normals = torch.stack(normals_all, dim=0)    # n, 3, h, w
        normals = F.normalize(normals, dim=1)

        mono_normals = normals.reshape(
            self.n_images, 3, -1
        ).permute(0, 2, 1)                            # n, hw, 3

        return mono_normals
    
    # -------------------------
    # mask normals
    # -------------------------
    def load_mask_normals(self):
        mask_normals_all = []
        mask_mvsa_all = []
        pbar = tqdm(range(self.n_images), desc="Loading mask_normals")
        for i in pbar:
            mask_normal_i = np.load(self.mask_normal_paths[i])[:, :, :3]
            mask_mvsa_i = np.load(self.mask_normal_paths[i])[:, :, 3]
            mask_normal_i = np.ascontiguousarray(mask_normal_i.transpose(2, 0, 1))

            mask_normals_all.append(
                torch.from_numpy(mask_normal_i).float().to(self.device)
            )
            mask_mvsa_all.append(
                torch.from_numpy(mask_mvsa_i).float().to(self.device)
            )
        mask_normals = torch.stack(mask_normals_all, dim=0) # n, 3, h, w
        mask_mvsa = torch.stack(mask_mvsa_all, dim=0) # n, h, w

        mask_normals = mask_normals.reshape(self.n_images, 3, -1).permute(0, 2, 1) # n, hw, 3
        mask_mvsa = mask_mvsa.reshape(self.n_images, -1) # n, hw

        return mask_normals, mask_mvsa

    # -------------------------
    # get raster cameras
    # -------------------------
    def get_raster_cameras(self, intrinsics_all, poses_all, height, width):
        zfar = 10.
        znear = 0.01
        raster_cam_w2c_list = []
        raster_cam_proj_list = []
        raster_cam_fullproj_list = []
        raster_cam_center_list = []
        raster_cam_FovX_list = []
        raster_cam_FovY_list = []
        raster_img_center_list = []

        for i in range(self.n_images):
            focal_length_x = intrinsics_all[i][0,0]
            focal_length_y = intrinsics_all[i][1,1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)

            cx = intrinsics_all[i][0, 2]
            cy = intrinsics_all[i][1, 2]

            c2w = poses_all[i]  # 4, 4
            w2c = c2w.inverse()  # 4, 4
            w2c_right = w2c.T

            world_view_transform = w2c_right.clone()
            projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=FovX, fovY=FovY).transpose(0,1).cuda()
            full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
            camera_center = world_view_transform.inverse()[3, :3]

            raster_cam_w2c_list.append(world_view_transform)
            raster_cam_proj_list.append(projection_matrix)
            raster_cam_fullproj_list.append(full_proj_transform)
            raster_cam_center_list.append(camera_center)
            raster_cam_FovX_list.append(torch.tensor([FovX]).cuda())
            raster_cam_FovY_list.append(torch.tensor([FovY]).cuda())

            raster_img_center_list.append(torch.tensor([cx, cy]).cuda())
        
        return raster_cam_w2c_list, raster_cam_proj_list, raster_cam_fullproj_list, raster_cam_center_list, raster_cam_FovX_list, raster_cam_FovY_list, raster_img_center_list