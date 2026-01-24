# 2025-04-10 before sam-based plane segmentation we detect plane regions using normal clustering
# and we compute distance map from depth map and normal map for sam-based plane segmentation
import os
import cv2
import numpy as np
import torch
import copy
from tqdm import trange
from natsort import natsorted
import argparse
from cuml.cluster import KMeans as cuKMeans
import distinctipy
from sklearn.cluster import AgglomerativeClustering
np.random.seed(325)
colors = np.array(distinctipy.get_colors(10))
colors[-1] = np.array([0,0,0])

def makedir_list(path_list):
    for path in path_list:
        if not os.path.exists(path):
            os.makedirs(path)

def read_files(directory, endtxt):
    file_paths = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(endtxt)]
    file_list = natsorted(file_paths)
    return file_list

def compute_distance(depth_input, K_input, normal_input, normal_conf, max_depth=10):
    distance_output = np.zeros_like(depth_input)
    distance_color_output = np.zeros_like(normal_input).astype(np.uint8)
    if max_depth > 0:
        depth_input[depth_input > max_depth] = 0
    depth_input = depth_input[6:-6, 6:-6]
    depth_input = cv2.copyMakeBorder(depth_input, 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=0)
    
    fx, fy, cx, cy = K_input[0,0],K_input[1,1],K_input[0,2],K_input[1,2]
    # Convert to 3D coordinates
    mask_valid = (depth_input > 0) & (~normal_conf)
    v, u = np.where(mask_valid)
    depth_mask = depth_input[mask_valid]
    normal_mask = (normal_input[mask_valid]).reshape(-1,3)
    x = (u - cx) * depth_mask / fx
    y = (v - cy) * depth_mask / fy
    z = depth_mask
    points = np.stack((x, y, z), axis=-1)
    points = points.reshape(-1, 3)
    distance = -np.sum(points * normal_mask, axis=-1)
    distance_color = (distance - distance.min()) / (distance.max() - distance.min() + 1e-20)
    distance_color = (distance_color * 255).clip(0, 255).astype(np.uint8)
    distance_color = cv2.applyColorMap(distance_color, cv2.COLORMAP_JET)
    distance_output[mask_valid] = distance
    distance_color_output[mask_valid] = np.squeeze(distance_color).astype(np.uint8)
    return distance_output, distance_color_output

def cluster_normals(normals, num_clusters=5):
    kmeans = cuKMeans(n_clusters=num_clusters)
    kmeans.fit(normals)
    return kmeans.labels_, kmeans.cluster_centers_

def smooth_normals(normals, conf, normal_mask):
    normals_smooth = np.zeros_like(normals)
    normals_smooth_viz = np.zeros_like(normals)
    unique_ids = np.unique(normal_mask)
    for id in unique_ids:
        if id == -1:
            continue
        region_idx = np.where(normal_mask == id)
        region_normals = normals[region_idx]
        region_conf = ~conf[region_idx]
        region_normals = region_normals[region_conf]

        if len(region_normals) > 3000:
            region_normals = region_normals[np.random.choice(len(region_normals), 3000, replace=False)]

        agglomerative = AgglomerativeClustering(n_clusters=None, distance_threshold=1, linkage='ward')
        labels = agglomerative.fit_predict(region_normals)
        unique_labels, counts = np.unique(labels, return_counts=True)
        dominant_label = unique_labels[np.argmax(counts)]
        dominant_cluster = region_normals[labels == dominant_label]
        dominant_normal = np.mean(dominant_cluster, axis=0)
        dominant_normal = dominant_normal / np.linalg.norm(dominant_normal)

        normals_smooth[region_idx] = dominant_normal
        normals_smooth_viz[region_idx] = ((dominant_normal+1)*127.5).astype(np.uint8).clip(0, 255)
    normals_smooth_viz = normals_smooth_viz.astype(np.uint8)
    return normals_smooth, normals_smooth_viz

def draw_mask(labels, rgb, conf):
    mask = labels.reshape(rgb.shape[0], rgb.shape[1])
    mask = filter_small_and_long(mask)
    mask = filter_small_and_long(mask, min_size=2000, kernel_size=(9,9))
    mask_color = (colors[mask] * 255).astype(np.uint8)
    # mask_color[conf] = np.zeros(3)
    combined = cv2.addWeighted(rgb, 0.5, mask_color, 0.5, 0)
    mask_valid = mask != -1
    rgb_sam = copy.deepcopy(rgb)
    rgb_sam[~mask_valid] = np.array([0,0,0])
    return combined, rgb_sam, mask
    # cv2.imwrite(f"{output_path}/for_sam/combined/{idx}.png", combined)
    # cv2.imwrite(f"{output_path}/for_sam/image/{idx}.png", rgb_sam)
    # np.save(f'{output_path}/for_sam/mask/{idx}.npy', mask)

def filter_small_and_long(mask, min_size=1000, kernel_size=(9,9)):
    unique_labels = np.unique(mask)
    new_mask = -np.ones_like(mask)
    for label in unique_labels:
        if label == -1:
            continue
        # 遍历mask=label所有的联通区域
        mask_i = mask == label
        num_labels, labels_im = cv2.connectedComponents(mask_i.astype(np.uint8))
        # print(num_labels)
        for j in range(1,num_labels):
            # 计算每个连通区域的面积
            maskij = (labels_im == j)
            region_size = np.sum(maskij)
            if region_size < min_size:
                new_mask[maskij] = -1
            else:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
                maskij_open = (cv2.morphologyEx(maskij.astype(np.uint8), cv2.MORPH_OPEN, kernel)).astype(np.bool_)
                region_size = np.sum(maskij_open)
                if region_size < min_size:
                    new_mask[maskij] = -1
                else:
                    new_mask[maskij_open] = label
    return new_mask

if __name__ == '__main__':
    # params
    n_cluster = 4

    parser = argparse.ArgumentParser(description='normal cluster params')
    parser.add_argument('--data_folder', type=str, required=True, help='Path to the input folder.')
    parser.add_argument('--seg_folder', type=str, required=True, help='Path to the output folder.')
    args = parser.parse_args()
    data_folder = args.data_folder
    seg_folder = args.seg_folder

    normal_input_folder = os.path.join(seg_folder, 'normal_npy_m')
    color_input_folder = os.path.join(data_folder, 'color')
    depth_m_input_folder = os.path.join(data_folder, 'depth_m3d')
    K_path = os.path.join(data_folder, 'intrinsic', 'intrinsic_depth.txt')

    output_folder = os.path.join(seg_folder, 'planesam')
    image_for_sam_path = os.path.join(output_folder, 'for_sam/image')
    dist_for_sam_path = os.path.join(output_folder, 'for_sam/distance')
    normal_mask_for_sam_path = os.path.join(output_folder, 'for_sam/normal_mask')
    combined_for_vis_path = os.path.join(output_folder, 'for_vis/normal_mask')
    makedir_list([output_folder, image_for_sam_path, dist_for_sam_path, normal_mask_for_sam_path, combined_for_vis_path])

    normal_list = read_files(normal_input_folder, '.npy')
    color_list = read_files(color_input_folder, '.jpg')
    depth_list = read_files(depth_m_input_folder, '.png')
    K_input = np.loadtxt(K_path)

    for i in trange(len(normal_list)):
        idx = os.path.basename(normal_list[i]).split('.')[0]
        color_i = cv2.imread(color_list[i])
        normal_i = np.load(normal_list[i]).astype(np.float32)[:,:,:3]
        normal_i_flatten = normal_i.reshape(-1,3)
        depth_i = cv2.imread(depth_list[i], -1) / 1000.0
        conf_i = np.load(normal_list[i]).astype(np.float32)[:,:,3] < 3
        normal_i_flatten[conf_i.reshape(-1)] = np.array([-1,-1,-1]).astype(np.float32)
        labels, centers = cluster_normals(normal_i_flatten, num_clusters=n_cluster)
        labels[conf_i.reshape(-1)] = -1
        combined_i, rgb_sam_i, normal_mask_i = draw_mask(labels, color_i, conf_i)
        dist_i, dist_color_i = compute_distance(depth_i, K_input, normal_i, conf_i)

        # save results for sam and viz
        cv2.imwrite(f'{image_for_sam_path}/{idx}.png', rgb_sam_i)
        cv2.imwrite(f'{combined_for_vis_path}/{idx}.png', combined_i)
        np.save(f'{normal_mask_for_sam_path}/{idx}.npy', normal_mask_i.astype(np.float16))
        np.save(f'{dist_for_sam_path}/{idx}.npy', dist_i.astype(np.float16))
