# 2025-04-07 对法向量聚类的粗略结果进行sam内部细分使用mask进行指导即可
# sam分割后最重要的一个问题就是对于过细的粒度该如何是好
# 对于每一张图像，分别在其对应的每个normal聚类mask上进行sam密集分割，最后叠加起来就好
import os
import numpy as np
from scipy import stats
import cv2
import torch
import copy
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import distinctipy
import argparse
from natsort import natsorted
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

np.random.seed(325)
colorlist = distinctipy.get_colors(100)
colorlist = np.array(colorlist)

def makedir_list(path_list):
    for path in path_list:
        if not os.path.exists(path):
            os.makedirs(path)
def read_files(directory, endtxt):
    file_paths = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(endtxt)]
    file_list = natsorted(file_paths)
    return file_list

def show_anns(anns, invalid, kernel_size=(9,9), open_flag=True, coarse_flag=False):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    anns_color = np.zeros((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 3))
    anns_npy = np.zeros((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1]))
    idx = 1
    for ann in sorted_anns:
        m = ann['segmentation']
        if np.sum(anns_color[m]) > 100 and coarse_flag:
            continue
        if open_flag:
            m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(np.bool_)
            if np.sum(m) < 500:
                continue
        color_idx = np.random.random(3)
        anns_npy[m] = idx
        anns_color[m] = color_idx
        idx += 1
    # print(f'contains {idx} sam masks')
    anns_color[invalid] = np.array([0,0,0])
    anns_npy[invalid] = 0
    anns_color = (anns_color * 255).astype(np.uint8)
    return anns_color, anns_npy

# 将整张图像sam分割结果由 kmeans聚类法相mask分割成对应平面的mask（甚至可以用聚类sam mask来进行细分）
def decompose_sam_by_normal(mask_n_list, mask_sam_input, min_size=1000, post_dict=None, dist_input=None):
    mask_sam_output = np.zeros_like(mask_sam_input)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
    for i in range(len(mask_n_list)):
        mask_valid = mask_n_list[i] & (mask_sam_input > 0)
        mask_sam_output[mask_valid] = mask_sam_input[mask_valid] + i*100
    unique_ids = np.unique(mask_sam_output)

    mask_sam_all = np.zeros_like(mask_sam_output)
    id_all = 0
    for id in unique_ids:
        m = cv2.morphologyEx((mask_sam_output == id).astype(np.uint8), cv2.MORPH_OPEN, kernel)
        if id == 0 or np.sum(m) < min_size:
            continue
        mask_sam_all[m.astype(np.bool_)] = id_all + 1
        id_all += 1
    
    post_dict = {'post_flag': False} if post_dict is None else post_dict
    mask_sam_all = post_process(mask_sam_all, dist_input=dist_input, post_dict=post_dict) if post_dict['post_flag'] else mask_sam_all
    mask_sam_all_color = (colorlist[mask_sam_all.astype(np.uint8)]* 255).astype(np.uint8)
    mask_sam_all_color[mask_sam_all == 0] = np.array([0,0,0])
    return mask_sam_all_color, mask_sam_all

# 将每个kmeans聚类法相mask上分别分割的sam mask叠加得到最终的sam分割结果
def compose_sam_from_normal():
    pass

# 最后一步的后处理，从mask_sam_all中 1.拆分id相同但是并不连同的mask 2.根据mask obb bbox长宽比 滤除长条状mask
def post_process(mask_sam_all, dist_input=None, post_dict=None):
    obb_size = post_dict['obb_size']
    large_size = post_dict['large_size']
    dist_thresh = post_dict['dist_thresh']
    dist_var = post_dict['dist_var']
    # obb_size: mask包围框宽高比阈值，滤除细长条； large_size: 防止把大面积的细长条也滤除了; dist_thresh: 分割联通区域的distance额外辅助
    unique_ids_pre = np.unique(mask_sam_all) # 后处理之前的unique_mask_ids
    mask_post_1 = np.zeros_like(mask_sam_all) # 存储后处理之后的新mask
    id_post_1 = 0 # 后处理之后的mask id计数
    for id in unique_ids_pre:
        if id == 0:
            continue
        mask_binary = (mask_sam_all == id)
        # if np.sum(mask_binary) > large_size:
        # 腐蚀一下万一可以分解出整片mask中的小平面呢
        # kernel = np.ones((5,5), np.uint8)
        # mask_binary = cv2.erode(mask_binary.astype(np.uint8), kernel, iterations=1).astype(np.bool_)

        # 先得mask 再画轮廓
        num_labels, labels_im = cv2.connectedComponents(mask_binary.astype(np.uint8), connectivity=8)

        # 记录当前平面id所有连通区域内的distance平均值到list中
        if dist_input is not None:
            dist_mask = dist_input > 0    
        else:
            dist_mask = np.ones_like(mask_sam_all).astype(np.bool_)
        dist_list = None
        dist_id_list = []

        for j in range(1,num_labels):
            maskij = ((labels_im == j) & dist_mask) if num_labels > 2 else (labels_im == j)
            if np.sum(maskij) > 1000:
                contours, _ = cv2.findContours(maskij.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                rect = cv2.minAreaRect(contours[0])
                cnt_w, cnt_h = rect[1]
                # if ((cnt_w / (cnt_h+1e-10)) > obb_size or (cnt_h / (cnt_w+1e-10)) > obb_size) or (cnt_w == 0) and (np.sum(maskij) < large_size):
                # if ((cnt_w / (cnt_h+1e-10)) > 2*obb_size or (cnt_h / (cnt_w+1e-10)) > 2*obb_size):
                #     continue
                if ((cnt_w / (cnt_h+1e-10)) > obb_size or (cnt_h / (cnt_w+1e-10)) > obb_size) and (np.sum(maskij) < large_size):
                    continue
                if dist_input is not None:
                    if np.var(dist_input[maskij]) > dist_var and dist_var < 10:
                        continue
                    dist_ij = np.mean(dist_input[maskij])
                    # dist_ij = np.median(dist_input[maskij]) # 用中位数是不是可以排除异常值对平均值的影响？
                    # dist_ij = stats.mode(np.round(dist_input[maskij],1))[0][0]
                    if dist_list is None:
                        dist_list = np.array([dist_ij])
                        mask_post_1[maskij] = id_post_1 + 1
                        dist_id_list.append(id_post_1 + 1)
                        id_post_1 += 1
                    else:
                        dist_diff = np.abs(dist_list - dist_ij)
                        if dist_diff[np.argmax(dist_diff)] > (dist_thresh*dist_ij):
                            mask_post_1[maskij] = max(dist_id_list)+1
                            dist_id_list.append(max(dist_id_list)+1)
                        else:
                            mask_post_1[maskij] = dist_id_list[np.argmin(dist_diff)]
                            dist_id_list.append(dist_id_list[np.argmin(dist_diff)])
                        dist_list = np.concatenate((dist_list, np.array([dist_ij])))
                else:
                    mask_post_1[maskij] = id_post_1 + 1
                    id_post_1 += 1
        if dist_list is not None:
            id_post_1 = max(dist_id_list)

    return mask_post_1

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='sam extractor params')
    parser.add_argument('--data_folder', type=str, required=True, help='Path to the data folder.')
    parser.add_argument('--seg_folder', type=str, required=True, help='Path to the output folder.')
    parser.add_argument('--open_flag', action="store_true", default=False)
    parser.add_argument('--coarse_flag', action="store_true", default=False)
    parser.add_argument('--use_ori', action="store_true", default=False) # if use original image for sam
    parser.add_argument('--debug_id', type=int, default=-1, help='single view debug id input')
    parser.add_argument('--post_flag', action="store_true", default=False)
    parser.add_argument('--dist_thresh', type=float, default=0.2, help='distance diff ratio threshold for post_process')
    parser.add_argument('--dist_var', type=float, default=0.6, help='distance diff ratio threshold for post_process')
    parser.add_argument('--large_size', type=int, default=6000, help='large size for post_process')
    parser.add_argument('--obb_size', type=float, default=5, help='obb size for post_process')
    parser.add_argument('--weight_pth', type=str, required=True, help='path to the weight file of sam weight.')
    args = parser.parse_args()

    # sam mask post process params dict
    post_params = {}
    post_params['dist_thresh'] = args.dist_thresh
    post_params['large_size'] = args.large_size
    post_params['post_flag'] = args.post_flag
    post_params['obb_size'] = args.obb_size
    post_params['dist_var'] = args.dist_var

    device = "cuda"
    sam = sam_model_registry["vit_h"](checkpoint=args.weight_pth)
    sam.to(device=device)
    print('sam model loaded')

    mask_generator_2 = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=16,
    pred_iou_thresh=0.88,
    stability_score_thresh=0.92,
    crop_n_layers=0,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=500,  # Requires open-cv to run post-processing
)
    data_folder = args.data_folder
    seg_folder = args.seg_folder

    img_folder = os.path.join(seg_folder, 'planesam/for_sam/image')
    dist_folder = os.path.join(seg_folder, 'planesam/for_sam/distance')
    img_ori_folder = os.path.join(data_folder, 'color')
    mask_n_folder = os.path.join(seg_folder, 'planesam/for_sam/normal_mask')
    output_folder = os.path.join(seg_folder, 'planesam/mask_npy')
    output_folder_vis = os.path.join(seg_folder, 'planesam/for_vis/sam_mask')
    makedir_list([output_folder, output_folder_vis])

    img_pth_list = read_files(img_folder, '.png')
    dist_pth_list = read_files(dist_folder, '.npy')
    img_ori_pth_list = read_files(img_ori_folder, '.jpg')
    mask_pth_list = read_files(mask_n_folder, '.npy')

    for idx in trange(len(img_pth_list)):
        if args.debug_id != -1:
            idx = args.debug_id
        img_id = img_pth_list[idx].split('/')[-1].split('.')[0]
        img_pth = img_pth_list[idx]
        dist_pth = dist_pth_list[idx]
        img_ori_pth = img_ori_pth_list[idx]
        mask_n_pth = mask_pth_list[idx]
        mask_n = np.load(mask_n_pth)
        n_cluster = np.unique(mask_n)

        img = cv2.imread(img_pth)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        dist = np.load(dist_pth)

        img_ori = cv2.imread(img_ori_pth)
        img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)

        mask_n_list = [] # from normal_cluster.py

        for id in n_cluster:
            if id == -1:
                continue
            mask_n_list.append(mask_n == id) # maintain mask_n_list
        
        mask_invalid = mask_n == -1
        with torch.cuda.amp.autocast():
            masks_sam = mask_generator_2.generate(img_ori if args.use_ori else img)
        
        mask_color, mask_npy = show_anns(masks_sam, mask_invalid, open_flag=args.open_flag, coarse_flag=args.coarse_flag)
        mask_sam_all_color_p, mask_sam_all_p = decompose_sam_by_normal(mask_n_list, mask_npy, post_dict=post_params, dist_input=dist)

        # save results for method and viz
        cv2.imwrite(f'{output_folder_vis}/{img_id}.png', mask_sam_all_color_p)
        np.save(f'{output_folder}/{img_id}.npy', mask_sam_all_p.astype(np.float16))

        if args.debug_id != -1: # debug mode only one pic
            print(f'debug mode on {img_id}.png, exit')
            break
    
    print(f'Distance refined mask saved at shape: {mask_sam_all_p.shape}')