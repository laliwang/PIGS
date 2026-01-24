import os
import shutil
import time
import argparse
import numpy as np
import open3d as o3d
import cv2
from natsort import natsorted
from tqdm import trange
from contextlib import contextmanager
from cuml.cluster import DBSCAN as cuDBSCAN

# ---------------------------------------------------------
# Log Manager
# ---------------------------------------------------------
class LogManager:
    @staticmethod
    @contextmanager
    def task(task_name):
        print(f"[-] Starting: {task_name}...")
        start_time = time.perf_counter()
        try:
            yield
            end_time = time.perf_counter()
            print(f"[√] Completed: {task_name} | Elapsed: {end_time - start_time:.4f}s\n")
        except Exception as e:
            end_time = time.perf_counter()
            print(f"[X] Failed: {task_name} | Error: {e} | Elapsed: {end_time - start_time:.4f}s\n")
            raise e

# ---------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------
def rebuild_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    if not os.path.exists(path):
        os.makedirs(path)

def copy_folder(source_dir, destination_dir):
    if os.path.exists(destination_dir):
        shutil.rmtree(destination_dir)
    shutil.copytree(source_dir, destination_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process scene name.')
    parser.add_argument('--data_folder', type=str, required=True, help='Path to the data folder.')
    parser.add_argument('--seg_folder', type=str, required=True, help='Path to the output folder.')
    parser.add_argument('--model', type=str, required=True, help='Name of the model.')
    parser.add_argument('--mask', type=str, required=True, help='Name of the mask.')
    parser.add_argument('--render', action="store_true", default=False, help='Whether to use reproj mesh.')
    parser.add_argument('--debug', action="store_true", default=False, help='Whether to render mvsa mask in video.')
    args = parser.parse_args()

    # Path Setup
    data_folder = args.data_folder
    seg_folder = args.seg_folder
    depth_model = args.model
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    pcd_path = os.path.join(data_folder, f'mesh/points3d_{depth_model}_proj.ply') if args.render else os.path.join(data_folder, f'mesh/points3d_{depth_model}.ply')
    cluster_path = os.path.join(seg_folder, 'mvsa_output/object/object_dict.npy')
    color_folder = os.path.join(data_folder, 'color')
    pose_folder = os.path.join(data_folder, 'pose')
    intrinsic_path = os.path.join(data_folder, 'intrinsic/intrinsic_depth.txt')

    mask_normal_com_folder = os.path.join(seg_folder, f'ghps_output/mask_normal_com_{args.mask}')
    mask_normal_final_folder = os.path.join(seg_folder, f'mvsa_output/mask_normal_final_{args.mask}')
    mask_normal_FFF_folder = os.path.join(seg_folder, f'mvsa_output/mask_normal_FFF_{args.mask}')
    mask_normal_vis_folder = os.path.join(seg_folder, f'mvsa_output/mask_normal_vis_{args.mask}')
    output_video_path = os.path.join(seg_folder, f'mvsa_output/output_video.mp4')
    output_ply_path = os.path.join(seg_folder, f'mvsa_output/object_pcd')

    # Global Config
    frame_rate = 30
    np.random.seed(325)
    
    with LogManager.task("Initialization and Resource Loading"):
        rebuild_folder(mask_normal_final_folder)
        rebuild_folder(mask_normal_FFF_folder)
        rebuild_folder(mask_normal_vis_folder)
        rebuild_folder(output_ply_path)
        
        object_dict = (np.load(cluster_path, allow_pickle=True)).item()
        object_num = len(object_dict.keys())
        colors_np = np.load(os.path.join(current_dir, 'inst_colors.npy'))
        shutil.copy(os.path.join(current_dir, 'inst_colors.npy'), os.path.join(output_ply_path, 'inst_colors.npy'))
        
        mask_normal_com_list = natsorted(os.listdir(mask_normal_com_folder))
        mask_normal_name_list = mask_normal_com_list
        mask_normal_final_list = [os.path.join(mask_normal_final_folder, m) for m in mask_normal_com_list]
        mask_normal_com_paths = [os.path.join(mask_normal_com_folder, m) for m in mask_normal_com_list]
        
        # Pre-allocate final arrays
        sample_npy = np.load(mask_normal_com_paths[0])
        final_npy_list = [np.zeros_like(sample_npy[..., -1]) for _ in range(len(mask_normal_com_paths))]
        
        pcd_ply = o3d.io.read_point_cloud(pcd_path)
        points_np = np.asarray(pcd_ply.points)
        print(f"[*] Loaded point cloud: {pcd_path}")

    # 1. Mask Correlation
    with LogManager.task("Cross-frame Mask Correlation (Assigning Instance IDs)"):
        for inst_idx, inst_id in enumerate(object_dict.keys(), start=1):
            mask_list = object_dict[inst_id]['mask_list']
            for mask_info in mask_list:
                frame_id = mask_info[0]
                mask_id = mask_info[1]
                mask_idx = mask_normal_name_list.index(f'{frame_id:06}.npy')

                mask_normal_com = np.load(mask_normal_com_paths[mask_idx])
                final_npy_list[mask_idx][mask_normal_com[..., -1] == mask_id] = (inst_id + 1)

            print(f"--- Correlating Instance: {inst_idx}/{object_num} (ID: {inst_id})", end='\r', flush=True)
        print("") # Clear line after CR

    with LogManager.task("Saving Correlated Masks to Disk"):
        for i in trange(len(final_npy_list), desc="Saving progress", leave=False):
            mask_normal_com = np.load(mask_normal_com_paths[i])
            mask_normal_final = np.concatenate([mask_normal_com[..., :-1], final_npy_list[i][..., np.newaxis]], axis=-1)
            np.save(mask_normal_final_list[i], mask_normal_final)

    # 2. Debug Visualization
    if args.debug:
        with LogManager.task("Generating Mask Correlation Visualization Video"):
            def mask_visualize(mask):
                mask_vis = np.zeros((*mask.shape, 3), dtype=np.uint8)
                unique_mask_ids = (np.unique(mask)).astype(np.int32)
                for m_id in unique_mask_ids:
                    if m_id == 0: continue
                    color = (colors_np[m_id] * 255).astype(np.uint8)
                    mask_vis[mask == m_id] = [color[2], color[1], color[0]] # BGR
                return mask_vis

            video_writer = None
            color_files = natsorted(os.listdir(color_folder))
            color_paths = [os.path.join(color_folder, c) for c in color_files]

            for i in trange(len(mask_normal_final_list), desc="Rendering video", leave=False):
                mask_data = np.load(mask_normal_final_list[i])[..., -1]
                img_color = cv2.imread(color_paths[i])
                mask_color_img = mask_visualize(mask_data)
                mask_vis_overlay = cv2.addWeighted(img_color, 0.5, mask_color_img, 0.5, 0)

                if video_writer is None:
                    h, w = mask_vis_overlay.shape[:2]
                    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (w, h))
                video_writer.write(mask_vis_overlay)
            
            if video_writer: video_writer.release()
            print(f"[*] Video saved at: {output_video_path}")

    # 3. Global Normal Consistency
    pose_files = natsorted(os.listdir(pose_folder))
    pose_paths = [os.path.join(pose_folder, p) for p in pose_files]

    def local_to_global(normal_map, pose_path, idx):
        T = np.loadtxt(pose_path)
        if np.isinf(T).any(): return None
        return np.dot(normal_map, T[:3, :3].T)

    def global_to_local(normal_map, pose_path, idx):
        T = np.loadtxt(pose_path)
        if np.isinf(T).any(): return None
        return np.dot(normal_map, T[:3, :3])

    normal_dict = {i+1: [] for i in range(object_num)}

    with LogManager.task("Computing Global Dominant Normals"):
        for i in trange(len(mask_normal_final_list), desc="Processing frames", leave=False):
            mask_normal_final = np.load(mask_normal_final_list[i])
            mask_layer = mask_normal_final[..., -1]
            norm_layer = mask_normal_final[..., :-1]
            
            unique_ids = np.unique(mask_layer)
            for m_i in unique_ids:
                if m_i == 0: continue
                indices = np.where(mask_layer == m_i)
                global_norms = local_to_global(norm_layer[indices], pose_paths[i], i)
                if global_norms is not None:
                    mean_norm = global_norms.mean(axis=0)
                    normal_dict[m_i].append(mean_norm / np.linalg.norm(mean_norm))

    with LogManager.task("Normal Aware DBSCAN Clustering"):
        dominant_normal_list = []
        other_list = []
        potential_split_num = 0
        
        for i in normal_dict.keys():
            norms_i = np.vstack(normal_dict[i])
            clustering = cuDBSCAN(eps=0.05, min_samples=10)
            labels = clustering.fit(norms_i).labels_

            u_labels, counts = np.unique(labels, return_counts=True)
            dom_label = u_labels[np.argmax(counts)]
            
            dom_cluster = norms_i[labels == dom_label]
            dom_norm = np.mean(dom_cluster, axis=0)
            dom_norm /= np.linalg.norm(dom_norm)
            dominant_normal_list.append(dom_norm)

            for label, count in zip(u_labels, counts):
                if label == dom_label: continue
                other_cluster = norms_i[labels == label]
                other_norm = np.mean(other_cluster, axis=0)
                other_norm /= np.linalg.norm(other_norm)
                if np.dot(dom_norm, other_norm) < 0.2:
                    potential_split_num += 1
                    other_list.append(other_norm)
        
        dominant_normal_list.extend(other_list)
        print(f"[*] Potential splits identified: {potential_split_num}")

    # 4. Final Export
    with LogManager.task("Exporting Smooth Point Clouds"):
        np.savetxt(os.path.join(output_ply_path, 'inst_num.txt'), np.array([object_num], dtype=np.float16))
        for inst_id in object_dict.keys():
            p_idx = object_dict[inst_id]['point_ids']
            pcd_inst = o3d.geometry.PointCloud()
            pcd_inst.points = o3d.utility.Vector3dVector(points_np[p_idx])
            pcd_inst.colors = o3d.utility.Vector3dVector(np.tile(colors_np[inst_id+1], (p_idx.shape[0], 1)))
            pcd_inst.normals = o3d.utility.Vector3dVector(np.tile(dominant_normal_list[inst_id], (p_idx.shape[0], 1)))
            o3d.io.write_point_cloud(os.path.join(output_ply_path, f'plane_inst_{inst_id+1}.ply'), pcd_inst)

    with LogManager.task("Exporting Global Consistent Normal Maps"):
        for i in trange(len(mask_normal_final_list), desc="Saving normal maps", leave=False):
            data = np.load(mask_normal_final_list[i])
            # mask_f, norm_f = data[..., -1], data[..., :-1]
            mask_f = data[..., -1]
            norm_f = np.zeros_like(data[..., :-1])
            
            norm_vis = np.zeros_like(norm_f)
            u_ids = np.unique(mask_f)
            for m_i in u_ids:
                if m_i == 0: continue
                smooth_i = global_to_local(dominant_normal_list[int(m_i-1)], pose_paths[i], i)
                smooth_i /= np.linalg.norm(smooth_i)
                norm_f[mask_f == m_i] = smooth_i
                norm_vis[mask_f == m_i] = ((smooth_i + 1) * 127.5).clip(0, 255)
            
            out_fff = np.concatenate((norm_f, mask_f[..., np.newaxis]), axis=-1)
            np.save(mask_normal_final_list[i].replace('mask_normal_final', 'mask_normal_FFF'), out_fff)
            cv2.imwrite(os.path.join(mask_normal_vis_folder, f"{i}.png"), norm_vis.astype(np.uint8))

    with LogManager.task("Cleanup Temporary Files"):
        if os.path.exists(mask_normal_final_folder):
            shutil.rmtree(mask_normal_final_folder)

    print("======================================================")
    print(f"  All processes finished successfully. Check outputs at {os.path.join(seg_folder, 'mvsa_output')}. ")
    print("======================================================")