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
import sys
import glob
import cv2
import torch
from natsort import natsorted
from tqdm import trange
import open3d as o3d
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from scene.initialize_utils import precompute_gaussians
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    fx: float
    fy: float

class DepthCameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    mat: np.array
    FovY: np.array
    FovX: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    depth_path: str
    depth_name: str
    mask_xpd_path: str
    mask_xpd_name: str
    pose_gt: np.array = np.eye(4)
    cx: float = -1
    cy: float = -1
    depth_scale: float = 1
    max_depth: float = 10.0
    timestamp: float = -1

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    gaussian_init: dict
    hive_dict: dict
    colorid_list: list

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def load_poses(pose_path, num):
    poses = []
    with open(pose_path, "r") as f:
        lines = f.readlines()
    for i in range(num):
        line = lines[i]
        c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
        c2w[:3,3] = c2w[:3,3] * 10.0
        w2c = np.linalg.inv(c2w)
        w2c = w2c
        poses.append(w2c)
    poses = np.stack(poses, axis=0)
    return poses

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX,
                              image_path=image_path, image_name=image_name, 
                              width=width, height=height, fx=focal_length_x, fy=focal_length_y)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    # cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : int(x.image_name.split('_')[-1]))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    
    js_file = f"{path}/split.json"
    train_list = None
    test_list = None
    if os.path.exists(js_file):
        with open(js_file) as file:
            meta = json.load(file)
            train_list = meta["train"]
            test_list = meta["test"]
            print(f"train_list {len(train_list)}, test_list {len(test_list)}")

    if train_list is not None:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name in train_list]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name in test_list]
        print(f"train_cam_infos {len(train_cam_infos)}, test_cam_infos {len(test_cam_infos)}")
    elif eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/points3D.ply")
    bin_path = os.path.join(path, "sparse/points3D.bin")
    txt_path = os.path.join(path, "sparse/points3D.txt")
    if not os.path.exists(ply_path) or True:
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
            print(f"xyz {xyz.shape}")
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           gaussian_init=None)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           gaussian_init=None,)
    return scene_info

# for depth cameras mesh extraction camera information
def readDepthCameras(
    color_paths,
    depth_paths,
    mask_xpd_paths,
    poses,
    mats,
    intrinsic,
    indices,
    depth_scale,
    max_depth,
    timestamps,
    crop_edge=0,
    eval_=False,
    sad_gs=False,
):
    cam_infos = []
    pose_w_t0 = np.eye(4)
    for idx_ in range(len(indices)):
        idx = indices[idx_]
        sys.stdout.write("\r")
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx_ + 1, len(indices)))
        sys.stdout.flush()
        mat = mats[idx]
        if sad_gs:
            print("SAD-GS")
            c2w = poses[idx]
            R = c2w[:3,:3]
            T = c2w[:3, 3]
            T = -R.T @ T
            # R = np.transpose(R)
        else:
            c2w = poses[idx]
            if idx_ == 0:
                pose_w_t0 = np.linalg.inv(c2w)
            # pass invalid pose
            if np.isinf(c2w).any():
                continue
            # if not eval_:
            #     c2w = pose_w_t0 @ c2w
            poses[idx] = c2w
            # get the world-to-camera transform and set R, T
            # cam_info 里的R和T本来就已经是世界到相机坐标系的坐标变换了
            w2c = np.linalg.inv(c2w)
            R = np.transpose(
                w2c[:3, :3]
            )  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

        image_color = Image.open(color_paths[idx])
        image_depth = (
            np.asarray(Image.open(depth_paths[idx]), dtype=np.float32) / depth_scale
        )
        image_color = np.asarray(
            image_color.resize((image_depth.shape[1], image_depth.shape[0]))
        )
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        if crop_edge > 0:
            image_color = image_color[
                crop_edge:-crop_edge,
                crop_edge:-crop_edge,
                :,
            ]
            image_depth = image_depth[
                crop_edge:-crop_edge,
                crop_edge:-crop_edge,
            ]
            cx -= crop_edge
            cy -= crop_edge

        height, width = image_color.shape[:2]
        # print("image size:", height, width)
        FovX = focal2fov(fx, width)
        FovY = focal2fov(fy, height)
        image_name = os.path.basename(color_paths[idx]).split(".")[0]
        depth_name = os.path.basename(depth_paths[idx]).split(".")[0]
        mask_xpd_name = os.path.basename(mask_xpd_paths[idx]).split(".")[0]
        #print(mask_xpd_paths[idx])
        #print(mask_xpd_name)此处已经确定两个都有效
        cam_info = DepthCameraInfo(
            uid=idx_,
            R=R,
            T=T,
            mat=mat,
            FovY=FovY,
            FovX=FovX,
            image_path=color_paths[idx],
            image_name=image_name,
            width=width,
            height=height,
            depth_path=depth_paths[idx],
            depth_name=depth_name,
            mask_xpd_path=mask_xpd_paths[idx],
            mask_xpd_name=mask_xpd_name,
            cx=cx,
            cy=cy,
            depth_scale=depth_scale,
            max_depth=max_depth,
            timestamp=timestamps[idx],
        )
        #print(cam_info.mask_xpd_path)
        #print(cam_info.mask_xpd_name)此处已经确定两个都有效
        cam_infos.append(cam_info)
    sys.stdout.write("\n")
    return cam_infos

# for dynamic cameras mesh extraction camera information
def readDynCameras(
    color_paths,
    poses,
    intrinsics,
    indices,
    crop_edge=0,
    eval_=False,
):
    cam_infos = []
    pose_w_t0 = np.eye(4)
    for idx_ in range(len(indices)):
        idx = indices[idx_]
        sys.stdout.write("\r")
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx_ + 1, len(indices)))
        sys.stdout.flush()

        w2c = poses[idx]
        K = intrinsics[idx]
        # if idx_ == 0:
        #     pose_w_t0 = np.linalg.inv(c2w)
        # # pass invalid pose
        # if np.isinf(c2w).any():
        #     continue
        # if not eval_:
        #     c2w = pose_w_t0 @ c2w
        # poses[idx] = c2w
        # # get the world-to-camera transform and set R, T
        # # cam_info 里的R和T本来就已经是世界到相机坐标系的坐标变换了
        # w2c = np.linalg.inv(c2w)
        R = np.transpose(
            w2c[:3, :3]
        )  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        image_color = Image.open(color_paths[idx])
        image_color = np.asarray(image_color)
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        if crop_edge > 0:
            image_color = image_color[
                crop_edge:-crop_edge,
                crop_edge:-crop_edge,
                :,
            ]
            cx -= crop_edge
            cy -= crop_edge

        height, width = image_color.shape[:2]
        # print("image size:", height, width)
        FovX = focal2fov(fx, width)
        FovY = focal2fov(fy, height)
        image_name = os.path.basename(color_paths[idx]).split(".")[0]

        cam_info = CameraInfo(
            uid=idx_,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            image_path=color_paths[idx],
            image_name=image_name,
            width=width,
            height=height,
            fx=fx,
            fy=fy,
        )
        cam_infos.append(cam_info)
    sys.stdout.write("\n")
    return cam_infos

def saveCfg(poses, config, save_path):
    cameras = []
    for idx in range(len(poses)):
        width, height = config["w"], config["h"]
        c2w = poses[idx]

        R = c2w[:3, :3]
        T = c2w[:3, 3]
        position = T.tolist()
        rotation = [x.tolist() for x in R]
        id = idx
        img_name = "frame_%04d" % idx
        fx, fy = config["fx"], config["fy"]
        cameras.append(
            {
                "id": id,
                "img_name": img_name,
                "width": width,
                "height": height,
                "position": position,
                "rotation": rotation,
                "fx": fx,
                "fy": fy,
            }
        )
    with open(os.path.join(save_path, "cameras.json"), "w") as file:
        json.dump(cameras, file)

def saveCfg_rep2(poses, config, save_path):
    cameras = []
    for idx in range(len(poses)):
        width, height = config["w"], config["h"]
        c2w = poses[idx]

        R = c2w[:3, :3]
        T = c2w[:3, 3]
        position = T.tolist()
        rotation = [x.tolist() for x in R]
        id = idx
        img_name = "%06d" % (idx*10)
        fx, fy = config["fx"], config["fy"]
        cameras.append(
            {
                "id": id,
                "img_name": img_name,
                "width": width,
                "height": height,
                "position": position,
                "rotation": rotation,
                "fx": fx,
                "fy": fy,
            }
        )
    with open(os.path.join(save_path, "cameras.json"), "w") as file:
        json.dump(cameras, file)

def saveCameraJson(poses, config, save_path):
    # print('save camera json as camears.json?')
    cameras = []
    for idx in range(len(poses)):
        width, height = config["w"], config["h"]
        if 'k' in config.keys():
            c2w = np.linalg.inv(poses[idx])
        else:
            c2w = poses[idx]
        # pass invalid pose
        if np.isinf(c2w).any():
            print("get inf at frame {:d}".format(idx))
            continue
        # get the world-to-camera transform and set R, T
        # w2c = np.linalg.inv(c2w)
        R = c2w[:3, :3]
        T = c2w[:3, 3]
        position = T.tolist()
        rotation = [x.tolist() for x in R]
        id = idx
        if 'k' in config.keys():
            img_name = str(idx)
            fx, fy = config["k"][idx][0,0], config["k"][idx][1,1]
        else:
            img_name = "frame_%04d" % idx
            fx, fy = config["fx"], config["fy"]
        cameras.append(
            {
                "id": id,
                "img_name": img_name,
                "width": width,
                "height": height,
                "position": position,
                "rotation": rotation,
                "fx": fx,
                "fy": fy,
            }
        )
    with open(os.path.join(save_path, "cameras.json"), "w") as file:
        json.dump(cameras, file)

def project_indice_select(depth_list, pose_list, indices):
    depth_indice_list = []
    pose_indice_list = []
    for idx_ in range(len(indices)):
        idx = indices[idx_]
        c2w = pose_list[idx]
        if np.isinf(c2w).any():
            continue
        else:
            depth_indice_list += [depth_list[idx]]
            pose_indice_list += [pose_list[idx]]
    return depth_indice_list, pose_indice_list


def project_init_pointcloud(depth_list, pose_list, K_input, max_depth=10, depth_scale=6553.5, init_w_gaussian=False):
    print('pointcloud initialized by Depth input...')
    depth_list = natsorted(depth_list)
    pcd_all = o3d.geometry.PointCloud()
    if len(depth_list) <= 10:
        skip = 1
    else:
        skip = 10
    for i in trange(0,len(depth_list),skip):
        depth_input = depth_list[i]
        pose_input = pose_list[i]
        depth_array = cv2.imread(depth_input, -1)
        depth_array = depth_array / depth_scale
        max_depth = 10.0
        # print(max_depth)
        if max_depth > 0:
            depth_array[depth_array > max_depth] = 0
        fx, fy, cx, cy = K_input[0,0],K_input[1,1],K_input[0,2],K_input[1,2]
        # Convert to 3D coordinates
        mask_valid = (depth_array > 0)
        v, u = np.where(mask_valid)
        depth_mask = depth_array[mask_valid]
        x = (u - cx) * depth_mask / fx
        y = (v - cy) * depth_mask / fy
        z = depth_mask
        points = np.stack((x, y, z), axis=-1)
        points = points.reshape(-1, 3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd = pcd.transform(pose_input)
        pcd = pcd.voxel_down_sample(voxel_size=0.025)
        if i == 0:
            pcd_all_points = np.array(pcd.points)
        else:
            pcd_all_points = np.concatenate((pcd_all_points, np.array(pcd.points)),axis=0)
    pcd_all.points = o3d.utility.Vector3dVector(pcd_all_points)
    pcd_all = pcd_all.voxel_down_sample(voxel_size=0.05)
    if not init_w_gaussian:
        if len(pcd_all.points) > 100000:
            print(len(pcd_all.points))
            indices = np.random.choice(len(pcd_all.points), 100000, replace=False)
            pcd_all = pcd_all.select_by_index(indices)
    print(f"pointcloud project by depth ok! Generated points num={np.array(pcd_all.points).shape[0]}")
    return np.array(pcd_all.points)          

# for RTG-SLAM replica type mesh extraction
def readReplicaSceneInfo(
    datapath, eval, llffhold, frame_start=0, frame_num=1, frame_step=1, voxel_size=None, init_w_gaussian=False,
):
    def load_poses(path, n_img):
        poses = []
        with open(path, "r") as f:
            lines = f.readlines()
        pose_w_t0 = np.eye(4)
        for i in range(n_img):
            line = lines[i]
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            if i == 0:
                pose_w_t0 = np.linalg.inv(c2w)
            c2w = pose_w_t0 @ c2w
            poses.append(c2w)
        return poses
    
    def load_poses_sad(path, n_img):
        poses = []
        with open(path, "r") as f:
            lines = f.readlines()
        for i in range(n_img):
            line = lines[i]
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            poses.append(c2w)
        return poses

    if os.path.exists(f"{datapath}/results"):
        print("results folder exist! recognized as original Replica!")
        color_paths = sorted(glob.glob(f"{datapath}/results/frame*.jpg"))
        depth_paths = sorted(glob.glob(f"{datapath}/results/depth*.png"))
        sad_flag = False
    elif os.path.exists(f"{datapath}/depths"):
        print("depths folder exist! recognized as sad-gs Replica!")
        color_paths = sorted(glob.glob(f"{datapath}/images/*.jpg"))
        depth_paths = sorted(glob.glob(f"{datapath}/depths/*.png"))
        sad_flag = True

    n_img = len(color_paths)
    timestamps = [i / 30.0 for i in range(n_img)]
    if sad_flag:
        poses = load_poses_sad(f"{datapath}/traj.txt", n_img)
    else:
        poses = load_poses(f"{datapath}/traj.txt", n_img)
    mats = load_poses(f"{datapath}/traj.txt", n_img)

    if frame_num == -1:
        indicies = list(range(n_img))
    else:
        frame_num = min(n_img, frame_num)
        indicies = list(range(frame_num))
    indicies = [frame_start + i * (frame_step + 1) for i in indicies]
    
    with open("/data/xxx/Replica/cam_params.json", "r") as f:
        config = json.load(f)["camera"]
    intrinsic = np.eye(3)
    intrinsic[0, 0] = config["fx"]
    intrinsic[1, 1] = config["fx"]
    intrinsic[0, 2] = config["cx"]
    intrinsic[1, 2] = config["cy"]

    cam_infos = readDepthCameras(
        color_paths=color_paths,
        depth_paths=depth_paths,
        poses=poses,
        mats=mats,
        intrinsic=intrinsic,
        indices=indicies,
        depth_scale=config["scale"],
        max_depth=10.0,
        timestamps=timestamps,
        crop_edge=0,
        sad_gs=sad_flag,
    )
    if os.path.exists(f"{datapath}/results"):
        saveCfg(poses, config, datapath)
    elif os.path.exists(f"{datapath}/depths"):
        saveCfg_rep2(poses, config, datapath)
    if eval:
        train_cam_infos = [
            c for idx, c in enumerate(cam_infos) if (idx + 1) % llffhold != 0
        ]
        test_cam_infos = [
            c for idx, c in enumerate(cam_infos) if (idx + 1) % llffhold == 0
        ]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(datapath, "points3d.ply")
    # if not os.path.exists(ply_path):
    # Since this data set has no colmap data, we start with random points
    xyz = project_init_pointcloud(depth_paths, poses, intrinsic, init_w_gaussian)
    shs = np.random.random((xyz.shape[0],3)) / 255.0
    pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((xyz.shape[0], 3)))

    storePly(ply_path, xyz, SH2RGB(shs) * 255)
    # try:
    #     pcd = fetchPly(ply_path)
    # except:
    #     pcd = None

    # Borrowed from SAD-GS for gaussian initialization
    gaussian_init = None
    # voxel_size 尝试从命令行给定
    if init_w_gaussian:
        mean_xyz, mean_rgb, cov = precompute_gaussians(torch.tensor(pcd.points).to('cuda'), torch.tensor(pcd.colors).to('cuda'), voxel_size)
        gaussian_init={"mean_xyz": mean_xyz, "mean_rgb": mean_rgb, "cov": cov}
        print("gaussian_init generated ok from input_pcd!")
    
    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        gaussian_init=gaussian_init,
    )
    return scene_info

# for RTG-SLAM scannet type mesh extraction
def readOursSceneInfo(
    datapath, segpath, eval_, llffhold, frame_start=0, frame_num=100, frame_step=0, voxel_size=None, init_w_gaussian=False,
    hive_dict=None,
):
    def load_poses(datapaths, n_img):
        poses = []
        for i in range(n_img):
            pose_file = datapaths[i]
            pose = np.loadtxt(pose_file)
            poses.append(pose)
        return poses

    color_path = "color"
    depth_path = "depth_m3d"
    pose_path = "pose"

    mask_xpd_path = f"mask_normal_FFF_{hive_dict['mask_type']}"
    print(f"Mask Normal Supervision from:{mask_xpd_path}")

    if eval_:
        color_path += "_eval"
        depth_path += "_eval"
        pose_path += "_eval"
        mask_xpd_path+= "_eval"
    color_paths = natsorted(
        glob.glob(f"{datapath}/{color_path}/*.jpg"),
        key=lambda x: int(os.path.basename(x).split(".")[0]),
    )
    depth_paths = natsorted(
        glob.glob(f"{datapath}/{depth_path}/*.png"),
        key=lambda x: int(os.path.basename(x).split(".")[0]),
    )
    # only mask normal from seg_folder referred to mvsa_output
    mask_xpd_paths = natsorted(
        glob.glob(f"{segpath}/{mask_xpd_path}/*.npy"),
        key=lambda x: int(os.path.basename(x).split(".")[0]),
    )
    n_img = len(color_paths)
    timestamps = [(i+1) / 30.0 for i in range(n_img)]

    crop_edge = 0

    pose_paths = natsorted(
        glob.glob(f"{datapath}/{pose_path}/*.txt"),
        key=lambda x: int(os.path.basename(x).split(".")[0]),
    )
    
    poses = load_poses(pose_paths, n_img)
    if eval_:
        eval_list = os.path.join(datapath, "eval_list.txt")
        if os.path.exists(eval_list):
            eval_list = list(np.loadtxt(eval_list, dtype=np.int32))
            print("eval_list:", eval_list)
            color_paths = [
                color_paths[i] for i in range(len(color_paths)) if i in eval_list
            ]
            depth_paths = [
                depth_paths[i] for i in range(len(depth_paths)) if i in eval_list
            ]
            mask_xpd_paths = [
                mask_xpd_paths[i] for i in range(len(mask_xpd_paths)) if i in eval_list
            ]
            poses = [poses[i] for i in range(len(poses)) if i in eval_list]
            n_img = len(poses)
    if eval_:
        pose_t0_c2w_fake = load_poses(f"{datapath}/pose", 1)[0]
        pose_t0_w2c = np.linalg.inv(pose_t0_c2w_fake)
        for i in range(len(poses)):
            poses[i] = pose_t0_w2c @ poses[i]
    if frame_num == -1:
        indicies = list(range(n_img))
    else:
        indicies = list(range(frame_num))
    indicies = [frame_start + i * (frame_step + 1) for i in indicies]
    indicies = [i for i in indicies if i < n_img]

    if eval_:
        indicies = list(range(n_img))

    intrinsic = np.loadtxt(os.path.join(datapath, "intrinsic", "intrinsic_depth.txt"))

    cam_infos = readDepthCameras(
        color_paths,
        depth_paths,
        mask_xpd_paths=mask_xpd_paths,
        poses=poses,
        mats=poses,
        intrinsic=intrinsic,
        indices=indicies,
        depth_scale=1000.0,
        max_depth=10.0,
        timestamps=timestamps,
        crop_edge=crop_edge,
        eval_=eval_,
    )
    saveCameraJson(
        poses,
        {
            "h": cam_infos[0].height,
            "w": cam_infos[0].width,
            "fx": intrinsic[0, 0],
            "fy": intrinsic[1, 1],
        },
        datapath,
    )

    train_cam_infos = cam_infos
    test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    # 用来利用颜色将3dgs与mask id绑定起来
    # colorid_path = os.path.join(datapath, "maskcluster", "inst_colors.npy")
    colorid_path = os.path.join(segpath, 'object_pcd', 'inst_colors.npy')
    colorid_list = list(np.load(colorid_path))

    # ply_path = os.path.join(datapath, "points3d.ply")

    plane_path = os.path.join(datapath, "points3d_gs.ply")
    if not os.path.exists(plane_path):
        print('Warning: no points3d_gs.ply found, use points3d.ply instead!')
        plane_path = os.path.join(datapath, "points3d.ply")
        
    if hive_dict['use_mask'] and os.path.exists(plane_path):
        # print("**********use plane3d.ply for initialization**********")
        pcd = fetchPly(plane_path)
        ply_path =plane_path
    else:
        depth_indiced, pose_indiced = project_indice_select(depth_paths, poses, indicies)
        xyz = project_init_pointcloud(depth_indiced, pose_indiced, intrinsic, depth_scale=1000.0, init_w_gaussian=init_w_gaussian)
        shs = np.random.random((xyz.shape[0],3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((xyz.shape[0], 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    
    # Borrowed from SAD-GS for gaussian initialization
    gaussian_init = None
    # voxel_size 尝试从命令行给定
    if init_w_gaussian:
        mean_xyz, mean_rgb, cov = precompute_gaussians(torch.tensor(pcd.points).to('cuda'), torch.tensor(pcd.colors).to('cuda'), voxel_size)
        gaussian_init={"mean_xyz": mean_xyz, "mean_rgb": mean_rgb, "cov": cov}
        print("gaussian_init generated ok from input_pcd!")
    
    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        gaussian_init=gaussian_init,
        hive_dict=hive_dict,
        colorid_list=colorid_list,
    )
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Replica": readReplicaSceneInfo,
    "Ours": readOursSceneInfo,
}