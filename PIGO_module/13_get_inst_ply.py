import open3d as o3d
import numpy as np
import cv2
import os
import copy
from tqdm import trange, tqdm
from natsort import natsorted
import torch
import argparse
import shutil
from plyfile import PlyData, PlyElement

# =========================================================
# Utils
# =========================================================

def rebuild_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def local_to_global(normal_map, pose_input):
    T = np.loadtxt(pose_input)
    if np.isinf(T).any():
        return None
    R = T[:3, :3]
    return normal_map @ R.T

def voxel_down_sample_normal(pcd_input, v_size=0.025, pcd_single=True):
    normal_input = np.asarray(pcd_input.normals)
    
    if pcd_single:
        normal_unique = np.unique(normal_input, axis=0)
        if normal_unique.shape[0] != 1:
            raise ValueError("Multiple normals in single-normal mode")
        normal_use = normal_unique[0]
    else:
        normal_use = normal_input.mean(axis=0)
        normal_use /= np.linalg.norm(normal_use) + 1e-8

    pcd_down = pcd_input.voxel_down_sample(voxel_size=v_size)
    pcd_down.normals = o3d.utility.Vector3dVector(
        np.tile(normal_use, (len(pcd_down.points), 1))
    )
    return pcd_down

def read_files(directory, endtxt):
    return natsorted([
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(endtxt)
    ])

# =========================================================
# 核心：frame-first 投影（新增）
# =========================================================

def project_frame_to_points(
    depth_path,
    pose_path,
    normal_map,
    mask_map,
    K,
    max_depth=10.0
):
    depth = cv2.imread(depth_path, -1).astype(np.float32) / 1000.0
    if max_depth > 0:
        depth[depth > max_depth] = 0

    valid = depth > 0
    v, u = np.where(valid)
    z = depth[v, u]

    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points = np.stack([x, y, z], axis=1)
    normals = normal_map[v, u]
    mask_ids = mask_map[v, u]

    T = np.loadtxt(pose_path)
    R = T[:3, :3]
    t = T[:3, 3]

    points_w = points @ R.T + t
    normals_w = normals @ R.T

    return points_w, normals_w, mask_ids

# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', required=True, help='Path to the data folder.')
    parser.add_argument('--seg_folder', required=True, help='Path to the mvsa_output folder.')
    parser.add_argument('--model_path', required=True, help='Path to PIGO rendered folder')
    parser.add_argument('--mask_type', default='fusion')
    parser.add_argument('--max_depth', type=float, default=10.0)
    parser.add_argument('--use_render', action="store_true", default=False)
    parser.add_argument('--filter', action="store_true", default=False, help='filter or not')
    parser.add_argument('--voxel_size', type=float, default=0.015, help='voxel size for point cloud down sample')
    args = parser.parse_args()

    model_path = os.path.abspath(args.model_path)
    data_folder = args.data_folder
    seg_folder = args.seg_folder
    voxel_size = args.voxel_size

    pose_dir = os.path.join(data_folder, 'pose')
    K_input = np.loadtxt(os.path.join(data_folder, 'intrinsic/intrinsic_depth.txt'))

    mask_path = os.path.join(seg_folder, f'mask_normal_FFF_{args.mask_type}')
    render_folder = os.path.join(model_path, 'train_pigo')
    iteration = os.listdir(render_folder)[0]

    render_depth = os.path.join(render_folder, iteration, 'render_depth16')
    render_normal = os.path.join(render_folder, iteration, 'renders_normal')

    out_path = os.path.join(model_path, 'mesh/filter_non')
    filter_path = os.path.join(model_path, 'mesh/filter')
    rebuild_folder(out_path)
    rebuild_folder(filter_path)

    inst_color_path = os.path.join(seg_folder, 'object_pcd/inst_colors.npy')
    inst_colors = np.load(inst_color_path)

    depth_list = read_files(render_depth, ".png")
    pose_list = read_files(pose_dir, ".txt")
    mask_list = read_files(mask_path, ".npy")
    normal_list = read_files(render_normal, ".npy")

    # print(len(depth_list), len(pose_list), len(mask_list))
    assert len(depth_list) == len(pose_list) == len(mask_list)

    # =====================================================
    # frame-first accumulation
    # =====================================================

    pcd_inst_dict = {}

    for i in trange(len(depth_list)):
        if np.isinf(np.loadtxt(pose_list[i])).any():
            continue

        mask_full = np.load(mask_list[i])
        mask_map = mask_full[..., -1]

        normal_map = (
            np.load(normal_list[i])
            if args.use_render
            else mask_full[..., :-1]
        )
        # normal_map = local_to_global(normal_map, pose_list[i])
        if normal_map is None:
            continue

        points_w, normals_w, mask_ids = project_frame_to_points(
            depth_list[i],
            pose_list[i],
            normal_map,
            mask_map,
            K_input,
            args.max_depth
        )

        for mid in np.unique(mask_ids):
            if mid == 0:
                continue
            sel = mask_ids == mid
            if sel.sum() == 0:
                continue

            # voxel_down_sample_normal for each instance point cloud inner current frame
            pcd_mid = o3d.geometry.PointCloud()
            pcd_mid.points = o3d.utility.Vector3dVector(points_w[sel])
            pcd_mid.normals = o3d.utility.Vector3dVector(normals_w[sel])
            pcd_mid.colors = o3d.utility.Vector3dVector(np.tile(inst_colors[int(mid)], (points_w[sel].shape[0], 1)))
            pcd_mid = voxel_down_sample_normal(pcd_mid, v_size=voxel_size, pcd_single=False)
            if mid not in pcd_inst_dict:
                pcd_inst_dict[mid] = pcd_mid
            else:
                pcd_inst_dict[mid] += pcd_mid

    # =====================================================
    # Build Open3D once per instance
    # =====================================================

    for inst_id, data in tqdm(
            pcd_inst_dict.items(),
            total=len(pcd_inst_dict),
            desc="Saving instance point clouds"
        ):

        pcd = voxel_down_sample_normal(
            data, v_size=voxel_size, pcd_single=False
        )

        o3d.io.write_point_cloud(
            f"{out_path}/{int(inst_id)}.ply", pcd
        )


    # post-processing for a little outlier filter
    def select_down_sample(pcd, mask):
        pcd_filter = o3d.geometry.PointCloud()
        pcd_filter.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[mask])
        pcd_filter.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals)[mask])
        pcd_filter.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[mask])
        return pcd_filter

    if args.filter:
        dist_threshold = 0.05
        plane_list = read_files(out_path, '.ply')
        point_cloud_folder = f'{model_path}/point_cloud/'
        iter_max = natsorted(os.listdir(point_cloud_folder))[-1]
        ref_point_path = f'{point_cloud_folder}/{iter_max}/point_cloud.ply'
        plydata = PlyData.read(ref_point_path)
        points_xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                            np.asarray(plydata.elements[0]["y"]),
                            np.asarray(plydata.elements[0]["z"])),  axis=1)
        points_ids = np.array(plydata.elements[0]["extra_field_62"])

        # filtering each plane instance ply
        print(f"post filtering each plane instance ply vs. PIGS_{iter_max}...")
        for i in trange(len(plane_list)):
            inst_id = int(os.path.basename(plane_list[i]).split('.')[0])
            plane_i = o3d.io.read_point_cloud(plane_list[i])
            ref_i = o3d.geometry.PointCloud()
            ref_i.points = o3d.utility.Vector3dVector(points_xyz[points_ids==inst_id])
            distance_i = plane_i.compute_point_cloud_distance(ref_i)

            mask_i = np.array(distance_i) < dist_threshold
            plane_filter_i = select_down_sample(plane_i, mask_i)
            o3d.io.write_point_cloud(f'{filter_path}/{inst_id}.ply', plane_filter_i)