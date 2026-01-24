# 2025-03-09 用来将scannet++ 解包出来的数据集存储为airplanes所需要的sensor_data格式
# 为了与真值mesh对齐 pose_intrinsic_imu的相机位姿变换到colmap第一帧下 
# 2025-03-17 不用对齐了 pose_intrinsic_imu中的aligned_pose就是对齐的了

import os
import math
import numpy as np
import cv2
import json
import shutil
import open3d as o3d
import argparse
from tqdm import trange
from natsort import natsorted

def rebuild_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    if not os.path.exists(path):
        os.makedirs(path)

def read_files(directory, endtxt):
    file_paths = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(endtxt)]
    file_list = natsorted(file_paths)
    return file_list

def read_txt_to_list(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    # 去掉每行的换行符并返回列表
    return [line.strip() for line in lines]

def qvec2rotmat(qvec):
    # 四元数转旋转矩阵
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])
def read_images_txt(path):
    # 从colmap/images.txt读取相机位姿
    T_list = []
    name_list = []
    with open(path, 'r') as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                if '.' in elems[0]:
                    continue
                image_id = int(elems[0])
                # image_id = int(float(elems[0]))
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                # camera_id = int(float(elems[8]))
                image_name = elems[9]
                R_line = qvec2rotmat(qvec) # (3,3)
                T_line = np.eye(4).astype(np.float32)
                T_line[:3,:3] = R_line
                T_line[:3,3] = tvec
                T_line = np.linalg.inv(T_line)
                T_list.append(T_line)
                name_list.append(image_name)
    T_list = np.stack(T_list, axis=0)
    # print(f'poses from colmap images.txt shape = {T_list.shape}')
    return T_list, name_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process scene name.')
    parser.add_argument('--scene_id', type=str, required=True, help='Name of the scene to process.')
    parser.add_argument('--path_data', type=str, default='', help='Path to scans_data folder.')
    parser.add_argument('--path_test', type=str, default='', help='Path to scans_test folder.')
    parser.add_argument('--path_hive', type=str, default='', help='Path to scans_hive folder.')
    args = parser.parse_args()
    
    scene_id = args.scene_id
    scene_id_rename = scene_id
    scene_id_rename_sensor = scene_id

    scene_root = args.path_data
    out_sensor_root = args.path_test
    out_ours_root = args.path_hive

    device_type = 'iphone'
    ori_w, ori_h = 1920, 1440
    tgt_w, tgt_h = 256, 192
    tgt_w_hive, tgt_h_hive = 640, 480

    rgb_1st_folder = os.path.join(scene_root, scene_id, device_type, 'rgb')
    depth_1st_folder = os.path.join(scene_root, scene_id, device_type, 'depth')
    camera_file = os.path.join(scene_root, scene_id, device_type, 'pose_intrinsic_imu.json')
    colmap_file = os.path.join(scene_root, scene_id, device_type, 'colmap/images.txt')
    scans_folder = os.path.join(scene_root, scene_id, 'scans')

    sensor_folder = os.path.join(out_sensor_root, scene_id_rename_sensor, 'sensor_data')
    rgb_step_folder = os.path.join(out_ours_root, scene_id_rename+"_step", 'color')
    depth_step_folder = os.path.join(out_ours_root, scene_id_rename+"_step", 'depth')
    pose_step_folder = os.path.join(out_ours_root, scene_id_rename+"_step", 'pose')

    rebuild_folder(sensor_folder)
    rebuild_folder(rgb_step_folder)
    rebuild_folder(depth_step_folder)
    rebuild_folder(pose_step_folder)
    
    camera_dict = json.load(open(camera_file, 'r'))
    colmap_list, name_list = read_images_txt(colmap_file)
    rgb_list = read_files(rgb_1st_folder, '.jpg')
    depth_list = read_files(depth_1st_folder, '.png')

    assert len(rgb_list) == len(depth_list), "Length of rgb_list and depth_list must be equal"
    
    # compute imu2colmap transformation
    colmap_name_0 = name_list[0].split('.')[0]
    if '/' in colmap_name_0:
        colmap_name_0 = colmap_name_0.split('/')[-1]
    colmap_pose_0 = colmap_list[0]
    imu_pose = np.array(camera_dict[colmap_name_0]['pose']).reshape(4,4).astype(np.float32)
    imu2colmap = colmap_pose_0 @ np.linalg.inv(imu_pose)

    image_num = len(rgb_list)
    # step = math.floor(image_num/1000) + 1
    step = 10

    print(f'image_num for sensor data: {image_num}')
    print(f'sample step for our data: {step}')

    K_average = np.zeros((3,3))
    pcd_all = o3d.geometry.PointCloud()

    for i in trange(0,len(rgb_list)):
        img_name = rgb_list[i].split('/')[-1].split('.')[0]
        img_id = str(img_name.split('_')[1])
        rgb_i = cv2.imread(rgb_list[i])
        # rgb_i = cv2.cvtColor(rgb_i, cv2.COLOR_BGR2RGB)
        depth_i = cv2.imread(depth_list[i], -1)
        # pose_intrinsic_imu不是每个帧都有哦
        if img_name not in camera_dict.keys():
            continue
        pose_i = np.array(camera_dict[img_name]['aligned_pose']).reshape(4,4).astype(np.float32)
        # if i == 0:
        #     imu2colmap = colmap_list[0] @ np.linalg.inv(pose_i)
        # pose_i = imu2colmap @ pose_i # imu相机位姿变换到colmap坐标系下

        shutil.copy(rgb_list[i], os.path.join(sensor_folder, 'frame-'+img_id+'.color.jpg'))
        shutil.copy(depth_list[i], os.path.join(sensor_folder, 'frame-'+img_id+'.depth.png'))
        np.savetxt(os.path.join(sensor_folder, 'frame-'+img_id+'.pose.txt'), pose_i)

        if i % step == 0:
            if np.isinf(pose_i).any():
                continue
            rgb_i = cv2.resize(rgb_i, (tgt_w_hive, tgt_h_hive), cv2.INTER_AREA)
            depth_i = cv2.resize(depth_i, (tgt_w_hive, tgt_h_hive), cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(rgb_step_folder, f'{i:06}.jpg'), rgb_i)
            cv2.imwrite(os.path.join(depth_step_folder, f'{i:06}.png'), depth_i)
            np.savetxt(os.path.join(pose_step_folder, f'{i:06}.txt'), pose_i)
        
        K_i = np.array(camera_dict[img_name]['intrinsic']).reshape(3,3).astype(np.float32)
        K_average += K_i

    K_average /= image_num
    print(f'K_average: {K_average}')

    intrinsic_sensor_path = os.path.join(out_sensor_root, scene_id_rename_sensor, 'intrinsic')
    intrinsic_ours_path = os.path.join(out_ours_root, scene_id_rename+"_step", 'intrinsic')
    rebuild_folder(intrinsic_sensor_path)
    rebuild_folder(intrinsic_ours_path)

    def resize_K(K_input, scale_w, scale_h):
        K_output = np.eye(4)
        K_output[0,0] = K_input[0,0]*scale_w
        K_output[1,1] = K_input[1,1]*scale_h
        K_output[0,2] = K_input[0,2]*scale_w
        K_output[1,2] = K_input[1,2]*scale_h
        return K_output
    
    K_sensor = resize_K(K_average, tgt_w/ori_w, tgt_h/ori_h)
    K_ours = resize_K(K_average, tgt_w_hive/ori_w, tgt_h_hive/ori_h)
    
    np.savetxt(os.path.join(intrinsic_sensor_path, 'intrinsic_depth.txt'), K_sensor)
    np.savetxt(os.path.join(intrinsic_ours_path, 'intrinsic_depth.txt'), K_ours)

    mesh_list = read_files(scans_folder, '.ply')
    json_list = read_files(scans_folder, '.json')

    print('meshes and annotations copied ing...')
    for i in trange(0,len(mesh_list)):
        mesh_i = mesh_list[i]
        json_i = json_list[i]
        shutil.copy(mesh_i, os.path.join(out_ours_root, scene_id_rename+"_step", mesh_i.split('/')[-1]))
        shutil.copy(mesh_i, os.path.join(out_sensor_root, scene_id_rename_sensor, mesh_i.split('/')[-1]))
        shutil.copy(json_i, os.path.join(out_sensor_root, scene_id_rename_sensor, json_i.split('/')[-1]))