import os
import cv2
import shutil
import argparse
import warnings
import numpy as np
from tqdm import tqdm
from natsort import natsorted
from sklearn.cluster import AgglomerativeClustering
import multiprocessing as mp

warnings.filterwarnings("ignore")

# =========================================================
# Utils
# =========================================================

def rebuild_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def read_list(folder, ext):
    files = [x for x in os.listdir(folder) if x.endswith(ext)]
    files = natsorted(files)
    return [os.path.join(folder, x) for x in files]

# =========================================================
# Core logic (PURE function, multiprocessing-safe)
# =========================================================

def find_dominant_single_frame(
    normal_conf,
    mask,
    conf_threshold=3,
    max_sample=3000,
):
    """
    normal_conf: (H, W, 4) -> xyz + confidence
    mask:        (H, W)    -> instance id
    """
    normals = normal_conf[..., :3]
    confs = normal_conf[..., 3]

    H, W = mask.shape
    mask_normal = np.zeros((H, W, 3), dtype=np.float32)
    mask_normal_vis = np.zeros((H, W, 3), dtype=np.uint8)
    filter_mask = np.zeros((H, W), dtype=mask.dtype)

    for mask_id in np.unique(mask):
        if mask_id == 0:
            continue

        region = (mask == mask_id)
        if region.sum() < 10:
            continue

        region_normals = normals[region]
        region_confs = confs[region]

        valid = region_confs > conf_threshold
        region_normals = region_normals[valid]

        if region_normals.shape[0] < 5:
            continue

        if region_normals.shape[0] > max_sample:
            idx = np.random.choice(
                region_normals.shape[0],
                max_sample,
                replace=False
            )
            region_normals = region_normals[idx]

        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1,
            linkage="ward"
        )
        cluster_labels = clustering.fit_predict(region_normals)

        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        if unique_labels.size == 0:
            continue

        dominant_label = unique_labels[np.argmax(counts)]
        dominant_cluster = region_normals[cluster_labels == dominant_label]

        dominant_normal = dominant_cluster.mean(axis=0)
        norm = np.linalg.norm(dominant_normal)
        if norm < 1e-6:
            continue
        dominant_normal /= norm

        mask_normal[region] = dominant_normal
        mask_normal_vis[region] = ((dominant_normal + 1) * 127.5).astype(np.uint8)
        filter_mask[region] = mask_id

    return mask_normal, mask_normal_vis, filter_mask

# =========================================================
# Multiprocessing task
# =========================================================

def process_one_frame(args):
    normal_path, mask_path, out_vis, out_npy, conf_threshold = args

    normal_conf = np.load(normal_path)
    mask = np.load(mask_path)

    if np.sum(mask > 0) == 0:
        H, W = mask.shape
        zero_vis = np.zeros((H, W, 3), dtype=np.uint8)
        zero_out = np.zeros((H, W, 4), dtype=np.float16)
        cv2.imwrite(out_vis, zero_vis)
        np.save(out_npy, zero_out)
        return

    mask_normal, mask_normal_vis, filter_mask = find_dominant_single_frame(
        normal_conf,
        mask,
        conf_threshold=conf_threshold,
    )

    mask_normal_combine = np.concatenate(
        [mask_normal, filter_mask[..., None]],
        axis=2
    )

    cv2.imwrite(out_vis, mask_normal_vis)
    np.save(out_npy, mask_normal_combine.astype(np.float16))

# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    # 防止 sklearn / numpy 线程互相抢
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, required=True)
    parser.add_argument('--seg_folder', type=str, required=True)
    parser.add_argument('--mask_type', type=str, required=True,
                        choices=['xpd', 'sam', 'fusion'])
    parser.add_argument('--conf_threshold', type=float, default=3)
    parser.add_argument('--num_workers', type=int, default=32)
    args = parser.parse_args()

    seg_folder = args.seg_folder

    normal_path = os.path.join(seg_folder, 'normal_npy_m')

    if args.mask_type == 'xpd':
        mask_path = os.path.join(seg_folder, 'mask_xpd')
    elif args.mask_type == 'sam':
        mask_path = os.path.join(seg_folder, 'planesam/mask_npy')
    else:
        mask_path = os.path.join(seg_folder.replace('hive_2d', 'hive_pigs'), 'ghps_output/mask_fusion')

    out_vis_path = os.path.join(
        seg_folder.replace('hive_2d', 'hive_pigs'), f'ghps_output/mask_normal_vis_{args.mask_type}'
    )
    out_npy_path = os.path.join(
        seg_folder.replace('hive_2d', 'hive_pigs'), f'ghps_output/mask_normal_com_{args.mask_type}'
    )

    rebuild_folder(out_vis_path)
    rebuild_folder(out_npy_path)

    normal_list = read_list(normal_path, '.npy')
    mask_list = read_list(mask_path, '.npy')

    assert len(normal_list) == len(mask_list)

    tasks = []
    for i in range(len(normal_list)):
        frame_id = os.path.basename(normal_list[i]).split('.')[0]
        tasks.append((
            normal_list[i],
            mask_list[i],
            f"{out_vis_path}/{frame_id}.png",
            f"{out_npy_path}/{frame_id}.npy",
            args.conf_threshold
        ))

    num_workers = min(args.num_workers, mp.cpu_count())

    with mp.Pool(num_workers) as pool:
        list(tqdm(
            pool.imap_unordered(process_one_frame, tasks),
            total=len(tasks)
        ))
