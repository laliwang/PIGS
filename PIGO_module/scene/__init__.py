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
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import numpy as np
import torch

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], not_pgsr=False, voxel_size=None, init_w_gaussian=None, hive_dict=None):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.source_path = args.source_path
        self.seg_path = args.seg_path

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.dataset_name = None
        self.colorid_list = None
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            self.dataset_name = "Colmap"
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            self.dataset_name = "Blender"
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "traj.txt")):
            print("Found traj.txt file, assuming replica data set!")
            self.dataset_name = "Replica"
            scene_info = sceneLoadTypeCallbacks["Replica"](args.source_path, False, 2, 0, -1, 0, \
                                        voxel_size=voxel_size, init_w_gaussian=init_w_gaussian)
        elif os.path.exists(os.path.join(args.source_path, "pose")):
            print("Found pose file folder, assuming scannet data set!")
            self.dataset_name = "Ours"
            scene_info = sceneLoadTypeCallbacks["Ours"](args.source_path, args.seg_path, False, 2, \
                                        frame_start=0, frame_num=-1, frame_step=0, \
                                        voxel_size=voxel_size, init_w_gaussian=init_w_gaussian, hive_dict=hive_dict)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            # 没有loaded_iter的话说明该过程是training过程而非infer过程
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling 打乱训练集和测试集的相机排列顺序防止过拟合
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        # print(f"cameras_extent {self.cameras_extent}") # radius其实就是场景范围 高斯初始化的时候就是在一个以translate为中心，半径为radius的球里初始化高斯点云的

        if self.dataset_name not in ['Colmap', 'Blender', 'Dyn']:
            use_depth = True
        else:
            use_depth = False
        
        if hive_dict is not None and hive_dict['use_mask']:
            if self.dataset_name not in ['Colmap', 'Blender', 'Dyn','Replica']:
                use_mask = True
            else:
                use_mask = False
        else:
            use_mask = False

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, use_depth=use_depth,use_mask=use_mask)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, use_depth=use_depth,use_mask=use_mask)
            if not not_pgsr:
                self.multi_view_num = args.multi_view_num # 这应该是只找每一帧附近的前multi_view_num帧作为nearest储备
                print("computing nearest_id") # nearest_id 是直接在所有cameras里面预先计算得到的如果是增量的就得一
                self.world_view_transforms = []
                camera_centers = []
                center_rays = []
                for id, cur_cam in enumerate(self.train_cameras[resolution_scale]):
                    self.world_view_transforms.append(cur_cam.world_view_transform)
                    camera_centers.append(cur_cam.camera_center)
                    R = torch.tensor(cur_cam.R).float().cuda()
                    T = torch.tensor(cur_cam.T).float().cuda()
                    center_ray = torch.tensor([0.0,0.0,1.0]).float().cuda()
                    center_ray = center_ray@R.transpose(-1,-2) # 这一步就是相机坐标系给变回世界坐标系了
                    center_rays.append(center_ray)
                self.world_view_transforms = torch.stack(self.world_view_transforms)
                camera_centers = torch.stack(camera_centers, dim=0)
                center_rays = torch.stack(center_rays, dim=0)
                center_rays = torch.nn.functional.normalize(center_rays, dim=-1)
                diss = torch.norm(camera_centers[:,None] - camera_centers[None], dim=-1).detach().cpu().numpy() # distance 矩阵 需要增量式更新的N*N相机平移距离矩阵
                tmp = torch.sum(center_rays[:,None]*center_rays[None], dim=-1) # 这个则是在计算各个相机旋转矩阵的余弦值邻接矩阵
                angles = torch.arccos(tmp)*180/3.14159 # 从弧度转换为角度？
                angles = angles.detach().cpu().numpy()
                with open(os.path.join(self.model_path, "multi_view.json"), 'w') as file:
                    for id, cur_cam in enumerate(self.train_cameras[resolution_scale]):
                        sorted_indices = np.lexsort((angles[id], diss[id])) # 在进行所谓nearest维护的时候angles的话语权要比diss大 所以进行多级排序 好厉害
                        # sorted_indices = np.lexsort((diss[id], angles[id]))
                        mask = (angles[id][sorted_indices] < args.multi_view_max_angle) & \
                            (diss[id][sorted_indices] > args.multi_view_min_dis) & \
                            (diss[id][sorted_indices] < args.multi_view_max_dis)
                        sorted_indices = sorted_indices[mask]
                        multi_view_num = min(self.multi_view_num, len(sorted_indices))
                        json_d = {'ref_name' : cur_cam.image_name, 'nearest_name': []}
                        for index in sorted_indices[:multi_view_num]:
                            cur_cam.nearest_id.append(index)
                            cur_cam.nearest_names.append(self.train_cameras[resolution_scale][index].image_name)
                            json_d["nearest_name"].append(self.train_cameras[resolution_scale][index].image_name)
                        json_str = json.dumps(json_d, separators=(',', ':'))
                        file.write(json_str)
                        file.write('\n')
                        # print(f"frame {cur_cam.image_name}, neareast {cur_cam.nearest_names}, \
                        #       angle {angles[id][cur_cam.nearest_id]}, diss {diss[id][cur_cam.nearest_id]}")

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            if init_w_gaussian:
                print("initializing gaussians from gaussian_init from SAD-GS ing ...")
                self.gaussians.create_from_gs(scene_info.gaussian_init["mean_xyz"], scene_info.gaussian_init["mean_rgb"], scene_info.gaussian_init["cov"], self.cameras_extent)
            else:
                if 'no_initial' in hive_dict.keys() and hive_dict['no_initial']:
                    self.gaussians.create_from_pcd_non(scene_info.point_cloud, self.cameras_extent, scene_info.hive_dict,scene_info.colorid_list)
                else:
                    self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, scene_info.hive_dict,scene_info.colorid_list)
                self.colorid_list = scene_info.colorid_list

    def save(self, iteration, mask=None):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"), mask) # 所使用的高斯场景表示点云也是可以保存的

    def getTrainCameras(self, scale=1.0):
        # 这里返回的train_cameras应该就已经是Camera对象而非scene目录里的CameraInfo对象了吧
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]