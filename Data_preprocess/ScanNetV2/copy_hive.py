import shutil
import os
import cv2
from natsort import natsorted
from tqdm import trange
import argparse
import numpy as np


def check_dir_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)
def read_files(directory, endtxt):
    file_paths = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(endtxt)]
    file_list = natsorted(file_paths)
    return file_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process scene name.')
    parser.add_argument('--scene_name', type=str, required=True, help='Name of the scene to process.')
    parser.add_argument('--path_ori', type=str, required=True, help='Path to the original scene.')
    parser.add_argument('--path_dest', type=str, required=True, help='Path to the destination scene.')
    args = parser.parse_args()
    scene_name = args.scene_name

    path_ori = args.path_ori
    path_dest = args.path_dest
    sensor_ori = os.path.join(path_ori, scene_name, 'sensor_data')
    intrinsic_ori = os.path.join(path_ori, scene_name, 'intrinsic')
    color_ori_list = read_files(sensor_ori, '.jpg')
    depth_ori_list = read_files(sensor_ori, '.png')
    pose_ori_list = read_files(sensor_ori, '.txt')
    intrinsic_list = natsorted(os.listdir(intrinsic_ori))

    color_dest = os.path.join(path_dest, scene_name+'_step', 'color')
    depth_dest = os.path.join(path_dest, scene_name+'_step', 'depth')
    intrinsic_dest = os.path.join(path_dest, scene_name+'_step', 'intrinsic')
    pose_dest = os.path.join(path_dest, scene_name+'_step', 'pose')

    check_dir_exist(color_dest)
    check_dir_exist(depth_dest)
    check_dir_exist(intrinsic_dest)
    check_dir_exist(pose_dest)

    image_num = len(color_ori_list)
    if image_num < 1000:
        step = 1
    elif image_num < 2000:
        step = 2
    elif image_num < 3000:
        step = 3
    elif image_num < 4000:
        step = 4
    else:
        step = 5

    print(f"selected evenly sample step = {step}")
    img_ori_0 = cv2.imread(color_ori_list[0])
    height_ori, width_ori = img_ori_0.shape[:2]
    print(f"original image size: {width_ori} x {height_ori}")
    print(f"PIGS required image size: 640 x 480")

    for i in trange(0, image_num, step):
        img_ori = cv2.imread(color_ori_list[i])
        img_idx = int(color_ori_list[i].split('.')[0].split('/')[-1].split('-')[1])
        pose_i = np.loadtxt(pose_ori_list[i])
        if np.isinf(pose_i).any():
            print(f'pose {img_idx}.txt has inf values, skip it!')
            continue
        img_dest = cv2.resize(img_ori, (640, 480))
        cv2.imwrite(os.path.join(color_dest, f'{img_idx:06}.jpg'), img_dest)
        shutil.copy(depth_ori_list[i], os.path.join(depth_dest, f'{img_idx:06}.png'))
        shutil.copy(pose_ori_list[i], os.path.join(pose_dest, f'{img_idx:06}.txt'))
    
    for i in trange(len(intrinsic_list)):
        shutil.copy(os.path.join(intrinsic_ori, intrinsic_list[i]), os.path.join(intrinsic_dest, intrinsic_list[i]))
    
    intrinsic_color = np.loadtxt(os.path.join(intrinsic_dest, 'intrinsic_color.txt'))
    intrinsic_color[0, :] = intrinsic_color[0, :] / (width_ori / 640)
    intrinsic_color[1, :] = intrinsic_color[1, :] / (height_ori / 480)
    np.savetxt(os.path.join(intrinsic_dest, 'intrinsic_color.txt'), intrinsic_color)
    
    print(f'scannet dataset for hive gaussian copy done!')
