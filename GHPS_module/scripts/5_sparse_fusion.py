# 2025-04-28 将mask_xpd可信度较高的mask与sam提取的额外补充mask相互补充融合得到最终结果
import numpy as np
import os
import cv2
import argparse
from natsort import natsorted
from tqdm import trange
import copy
import distinctipy
import matplotlib.pyplot as plt

def read_files(directory, endtxt):
    file_paths = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(endtxt)]
    file_list = natsorted(file_paths)
    return file_list

def filter_small_and_long(mask, min_size=1000, large_size=6000, obb_size=5):
    if np.sum(mask) < min_size:
        return False
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rect = cv2.minAreaRect(contours[0])
    cnt_w, cnt_h = rect[1]
    if ((cnt_w / (cnt_h+1e-10)) > obb_size or (cnt_h / (cnt_w+1e-10)) > obb_size) and (np.sum(mask) < large_size):
        return False
    else:
        return True

def obb_occupancy_ratio(mask, ratio_thresh=0.5, min_size=200, large_size=3000, obb_size=5):
    """
    compute the occupancy ratio of the mask in its minimum bounding rectangle
    """
    mask_u8 = mask.astype(np.uint8)

    contours, _ = cv2.findContours(
        mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours) == 0:
        return 0.0, 0.0, 0

    all_pts = np.vstack(contours)
    rect = cv2.minAreaRect(all_pts)
    w, h = rect[1]
    obb_area = max(w * h, 1.0)
    mask_area = int(np.sum(mask_u8))
    occupancy = mask_area / obb_area
    
    long_flag = ((w / (h+1e-10)) > obb_size or (h / (w+1e-10)) > obb_size) or (mask_area < min_size)
    ratio_flag = (occupancy < ratio_thresh) and (mask_area < large_size)
    valid_flag = ~(long_flag or ratio_flag) # not too long or too small and occupancy not too low
    return valid_flag

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process scene name.')
    parser.add_argument('--data_folder', type=str, required=True, help='Path to the data folder.')
    parser.add_argument('--seg_folder', type=str, required=True, help='Path to the output folder.')
    parser.add_argument('--crop', type=int, default=0, help='Crop the image for scannet v2 data.')
    args = parser.parse_args()
    data_folder = args.data_folder
    seg_folder = args.seg_folder
    crop = args.crop

    mask_xpd_folder = os.path.join(seg_folder, 'mask_xpd')
    mask_sam_folder = os.path.join(seg_folder, 'planesam/mask_npy')
    # Each module final output saved at hive_pigs not hive_2d
    mask_fusion_folder = os.path.join(seg_folder.replace('hive_2d', 'hive_pigs'), 'ghps_output/mask_fusion')

    if not os.path.exists(mask_fusion_folder):
        os.makedirs(mask_fusion_folder)

    mask_xpd_list = read_files(mask_xpd_folder, '.npy')
    mask_sam_list = read_files(mask_sam_folder, '.npy')
    mask_fusion_list = []

    for i in trange(len(mask_xpd_list)):
        mask_xpd_i = np.load(mask_xpd_list[i])
        mask_sam_i = np.load(mask_sam_list[i])
        mask_fusion_i = copy.deepcopy(mask_xpd_i)
        xpd_non_mask = (mask_xpd_i == 0)
        id_start = np.max(mask_xpd_i)
        mask_sam_i[~xpd_non_mask] = 0
        sam_unique_ids = np.unique(mask_sam_i)
        for id in sam_unique_ids:
            if id == 0:
                continue
            mask_id = (mask_sam_i == id)
            # if filter_small_and_long(mask_id):
            if obb_occupancy_ratio(mask_id):
                mask_fusion_i[mask_id] = id_start + 1
                id_start += 1
        
        if crop > 0:
            mask_fusion_i = mask_fusion_i[crop:-crop, crop:-crop].astype(np.uint8)
            mask_fusion_i = cv2.copyMakeBorder(mask_fusion_i, crop, crop, crop, crop, cv2.BORDER_CONSTANT, value=0).astype(np.float16) 

        mask_fusion_list.append(mask_fusion_i)
        np.save(os.path.join(mask_fusion_folder, mask_xpd_list[i].split('/')[-1]), mask_fusion_i)
    
    print(f'Sparse Planar Fusion mask saved at shape: {mask_fusion_i.shape}')