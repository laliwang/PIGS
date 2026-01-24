import collections
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import struct
import click
import numba
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
import trimesh
from loguru import logger
from numba import cuda
from skimage.measure import marching_cubes
from tqdm import tqdm, trange
import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "airplanes_part/seq_ransac")
    )
)

from ransac_kernels import (
    assign_nearest_label,
    compute_d,
    compute_d_optimized,
    connected_components_labels,
    count_occurrences,
    fill_inlier_matrix_dot,
    fill_inlier_matrix_embeddings_optimised_shared_mem,
    filter_small_planes,
)


@dataclass
class StoppingCriteria:
    # if we start finding planes smaller than this then stop
    min_planar_size: int = 30
    # let's not find any more planes than this
    max_planes: int = 1


@dataclass
class RansacOptions:
    normal_inlier_threshold: float = 0.8
    distance_inlier_threshold: float = 0.1## r_d in the paper
    # distance_inlier_threshold: float = 0.05 # modified in 2025-05-10
    embeddings_inlier_threshold: float = 0.8
    num_iterations: int = 1000
    # if this is true, we guarantee that all points will end up assigned to a plane
    force_assign_points_to_planes: bool = False


class CustomSequentialRansac:
    def __init__(
        self,
        stopping_criteria: StoppingCriteria,
        ransac_options: RansacOptions,
        # embeddings_usage: str,
        merge_planes_with_similar_embeddings: bool = False,
    ):
        self.stopping_criteria = stopping_criteria
        self.ransac_options = ransac_options
        # self.embeddings_usage = embeddings_usage
        self.merge_planes_with_similar_embeddings = merge_planes_with_similar_embeddings

        self.inlier_mat = None
        self.d_inlier_mat = None

        self.timings = collections.defaultdict(list)

    def allocate_inlier_matrix(self, num_iterations: int, num_points: int):
        """
        Allocate memory for the inlier matrix.
        params:
            num_iterations: number of iterations to allocate for
            num_points: number of points to allocate for
        """

        self.inlier_mat = torch.zeros(
            (num_iterations, num_points), dtype=torch.int8, requires_grad=False, device="cuda"
        )
        self.d_inlier_mat = cuda.as_cuda_array(self.inlier_mat)

    def __call__(
        self,
        pcd: o3d.geometry.PointCloud,
        #mesh_edges: np.ndarray,
        embeddings: Optional[np.ndarray] = None,
        estimate_normals=False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Runs sequential ransac on the given point cloud and returns a per-point label array
            and a boolean array indicating which points were assigned labels by ransac.

        Args:
            pcd (o3d.geometry.PointCloud): The point cloud to run ransac on.
            embeddings: if set, consider predicted embeddings when fitting planes

        Returns:
            np.ndarray: A per-point label array. Each point is labelled with the plane it has been
                assigned to
            np.ndarray: A boolean array indicating which points were assigned labels by ransac,
                i.e. which points are 'core points'.
        """
        # We seed numpy before each run, rather than at the start of the script.
        # This ensures results are deterministic regardless of the order we process the scenes.
        np.random.seed(10)
        torch.manual_seed(10)
        torch.cuda.manual_seed(10)
        if estimate_normals:
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=15))

        per_point_plane_assignment = np.zeros(len(pcd.points), np.int32)
        original_points_array = np.asarray(pcd.points)
        original_normals_array = np.asarray(pcd.normals)
        original_embeddings = embeddings.copy() if embeddings is not None else None

        # We use original_idx_for_each_point to keep track of the idx each point corresponds to in
        # the original (full) point cloud. As we remove points from pcd.points, we also remove
        # points from this array. This ensures we maintain a 1:1 mapping between the values in this
        # array and the points in pcd.
        original_idx_for_each_point = np.arange(len(pcd.points))
        original_idx_for_each_point = torch.tensor(
            original_idx_for_each_point, device="cuda", dtype=torch.int32
        )

        plane_idx = 0

        points_tensor = torch.tensor(
            np.array(pcd.points), requires_grad=False, device="cuda", dtype=torch.float32
        )
        normals_tensor = torch.tensor(
            np.array(pcd.normals), requires_grad=False, device="cuda", dtype=torch.float32
        )
        if embeddings is not None:
            assert embeddings.shape[1] == 3, "embeddings must be Nx3"
            embeddings_tensor = torch.tensor(
                embeddings, requires_grad=False, device="cuda", dtype=torch.float32
            )
        else:
            embeddings_tensor = None

        #mesh_edges = torch.tensor(mesh_edges, device="cuda")

        per_point_plane_assignment = torch.tensor(per_point_plane_assignment).cuda().long()
        if embeddings is not None:
            original_embeddings = torch.tensor(original_embeddings).cuda().float()
        original_normals_array = torch.tensor(original_normals_array).cuda().float()
        original_points_array = torch.tensor(original_points_array).cuda().float()

        num_points = len(points_tensor)
        num_iterations = self.ransac_options.num_iterations

        # we have a limited amount of shared memory we can use for keeping track of the
        # inlier sums. We have set it to 1024 32-bit ints in shared memory, so we need to make sure
        # we don't exceed this.
        assert num_iterations < 1024, f"num_iterations must be less than 1024, got {num_iterations}"

        self.allocate_inlier_matrix(num_iterations=num_iterations, num_points=num_points)
        # for _ in range(self.stopping_criteria.max_planes, unit="plane"):
        for _ in range(self.stopping_criteria.max_planes):
            # Find a single plane with ransac
            
            inliers, inlier_sum = self.find_single_largest_plane_dot_torch_cuda(
                    points=points_tensor, normals=normals_tensor)

            # for debugging
            assert inliers.sum() == inlier_sum, f"{inliers.sum()} != {inlier_sum}"

            # See if we want to early exit.
            if inlier_sum < self.stopping_criteria.min_planar_size:
                break

            # Update the global assignment array (we use plane_idx + 1 so that the first plane
            # starts from index 1. This means unassigned points will end up as 0.)
            this_plane_original_idxs = original_idx_for_each_point[inliers]
            per_point_plane_assignment[this_plane_original_idxs] = plane_idx + 1
            plane_idx += 1

            # Remove the points from the pcd and the original point array
            points_tensor = points_tensor[~inliers]
            normals_tensor = normals_tensor[~inliers]

            original_idx_for_each_point = original_idx_for_each_point[~inliers]

            if embeddings is not None:
                embeddings_tensor = embeddings_tensor[~inliers]

            assert len(original_idx_for_each_point) == len(
                points_tensor
            ), f"{len(original_idx_for_each_point)} != {len(points_tensor)}"

        # core plane points are the points which were given a label by ransac. We compute this
        # from the per_point_plane_assignment array now, before we potentially give all unlabelled
        # points a label (which we do if force_assign_points_to_planes is True)
        core_plane_points = (per_point_plane_assignment != 0).cpu().numpy()

        per_point_plane_assignment = self.remove_small_planes_cuda(per_point_plane_assignment)

        per_point_plane_assignment = per_point_plane_assignment.cpu().numpy()

        return per_point_plane_assignment, core_plane_points

    def find_single_largest_plane_dot_torch_cuda(
        self,
        points: torch.tensor,
        normals: torch.tensor,
    ) -> torch.tensor:
        """
        run ransac on the given point cloud and return the inliers for the largest plane found
        params:
            points: Nx3 tensor of points
            normals: Nx3 tensor of normals
            embeddings: Nx3 tensor of embeddings
        """
        num_points = len(points)

        num_iterations = self.ransac_options.num_iterations

        distance_inlier_threshold = self.ransac_options.distance_inlier_threshold
        normal_inlier_threshold = self.ransac_options.normal_inlier_threshold

        # Allocate memory for the result
        # we use inlier_sums to keep track of the best hypothesis
        ds = torch.zeros((num_points), dtype=torch.float32, requires_grad=False, device="cuda")
        sample_idxs = torch.randint(
            size=(num_iterations,),
            low=0,
            high=num_points - 1,
            requires_grad=False,
            device="cuda",
            dtype=torch.int32,
        )
        inlier_sums = torch.zeros(
            (num_iterations), dtype=torch.int32, requires_grad=False, device="cuda"
        )

        # Define block and grid dimensions for compute_d
        threads_per_block_1d = 1024
        blocks_per_grid_1d = (num_points + threads_per_block_1d - 1) // threads_per_block_1d

        # Define block and grid dimensions for fill_inlier_matrix
        threads_per_block = (8, 32)
        blocks_per_grid_x = (num_iterations + threads_per_block[0] - 1) // threads_per_block[0]
        blocks_per_grid_y = (num_points + threads_per_block[1] - 1) // threads_per_block[1]
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        # Copy data to device
        d_points = cuda.as_cuda_array(points)
        d_normals = cuda.as_cuda_array(normals)

        d_ds = cuda.as_cuda_array(ds)
        d_sample_idxs = cuda.as_cuda_array(sample_idxs)
        d_inlier_sums = cuda.as_cuda_array(inlier_sums)

        cuda.synchronize()

        # Launch the kernel
        compute_d[blocks_per_grid_1d, threads_per_block_1d](d_points, d_normals, d_ds)

        fill_inlier_matrix_dot[blocks_per_grid, threads_per_block](
            d_sample_idxs,
            d_points,
            d_normals,
            d_ds,
            self.d_inlier_mat,
            numba.float32(distance_inlier_threshold),
            numba.float32(normal_inlier_threshold),
            d_inlier_sums,
            numba.int32(num_iterations),  # n_rows
            numba.int32(num_points),  # n_cols
        )

        # Synchronize to ensure all GPU work is finished
        cuda.synchronize()

        # get the best iteration, we use the last column as it is the sum of inliers
        best_iter = inlier_sums.argmax()
        torch.cuda.synchronize()

        # get the inliers for the best iteration, we ignore the last column as it is the sum of inliers
        best_plane_inliers = self.inlier_mat[best_iter, :num_points]

        return best_plane_inliers.bool(), inlier_sums[best_iter]

    def merge_planes(
        self,
        per_point_plane_assignment: torch.Tensor,
        original_points_array: torch.Tensor,
        original_normals_array: torch.Tensor,
        original_embeddings: Optional[torch.Tensor],
    ):
        # if self.embeddings_usage == "none":
        #     original_embeddings = torch.zeros((len(original_points_array), 3), device="cuda")
        original_embeddings = torch.zeros((len(original_points_array), 3), device="cuda")

        # get mean embeddings per plane
        max_label = per_point_plane_assignment.max() + 1
        mean_normal = torch.zeros((max_label, 3), device="cuda")
        mean_offset = torch.zeros((max_label, 1), device="cuda")
        mean_embedding = torch.zeros((max_label, original_embeddings.shape[-1]), device="cuda")

        for label in range(max_label):
            if label > 0:
                mask = per_point_plane_assignment == label
                # compute plane parameters and move all points onto the plane
                plane_points = original_points_array[mask]
                plane_normal = torch.median(original_normals_array[mask], dim=0)[0]
                normal = plane_normal / torch.norm(plane_normal)

                offset = -torch.median((plane_points * normal[None, :]).sum(-1), dim=0)[0]

                mean_normal[label] = normal
                mean_offset[label] = offset
                mean_embedding[label] = torch.median(original_embeddings[mask], dim=0)[0]

        # create final inlier matrix
        embeddings_diff = torch.sqrt(
            ((mean_embedding[None, :] - mean_embedding[:, None]) ** 2).sum(2)
        )
        normal_diff = (mean_normal[None, :] * mean_normal[:, None]).sum(2)
        offset_diff = torch.abs(mean_offset[None, :] - mean_offset[:, None])[..., 0]

        # if self.embeddings_usage == "ransac":
        #     inlier_matrix = ((embeddings_diff < 0.2) * (normal_diff > 0.6)).long()
        # else:
        inlier_matrix = ((normal_diff > 0.8) * (offset_diff < 0.3)).long()

        for label in range(max_label):
            if label > 0:
                mask = per_point_plane_assignment == label
                per_point_plane_assignment[mask] = inlier_matrix[label].argmax()

        return per_point_plane_assignment

    def remove_small_planes_cuda(self, labels):
        counts = torch.zeros((labels.shape[0]), device="cuda", dtype=torch.int32)
        threads_per_block = 1024
        blocks_per_grid = (labels.shape[0] + threads_per_block - 1) // threads_per_block
        count_occurrences[blocks_per_grid, threads_per_block](labels, counts)
        filter_small_planes[blocks_per_grid, threads_per_block](
            labels, counts, self.stopping_criteria.min_planar_size
        )
        return labels


def run_ransac_on_mesh(
    out_ply_file: Path,
    embeddings: Optional[np.ndarray],
    # embeddings_usage: str,
    force_assign_points_to_planes: bool,
    normals_inlier_threshold: float,
    distance_inlier_threshold: float,
    embeddings_inlier_threshold: float,
    merge_planes_with_similar_embeddings: bool,
    point_cloud:o3d.geometry.PointCloud
) -> None:
    # Run ransac
    ransac_plane_finder = CustomSequentialRansac(
        stopping_criteria=StoppingCriteria(),
        ransac_options=RansacOptions(
            force_assign_points_to_planes=force_assign_points_to_planes,
            normal_inlier_threshold=normals_inlier_threshold,
            distance_inlier_threshold=distance_inlier_threshold,
            embeddings_inlier_threshold=embeddings_inlier_threshold,
        ),
        # embeddings_usage=embeddings_usage,
        merge_planes_with_similar_embeddings=merge_planes_with_similar_embeddings,
    )
    #per_point_labels是一个和点云数量维度一样的平面标签列表，值为平面标签，0表示非平面点
    #core_plane_points实际上是一个平面标签索引啊，我直接给索引外的点删了试试

    pcd=point_cloud
    colors = np.asarray(pcd.colors)  # Nx3 数组
    if len(colors) == 0:
        raise ValueError("Point cloud has no color information.")

    unique_colors = np.unique(colors, axis=0)  # 获取唯一的颜色

    segments = []

    # 对于每个唯一的颜色值，提取所有属于该颜色的点
    for color in unique_colors:
        # 找到所有颜色相同的点的索引
        color_mask = np.all(colors == color, axis=1)
    
        # 提取这些点
        cluster_points = np.asarray(pcd.points)[color_mask]
        cluster_colors = np.asarray(pcd.colors)[color_mask]
        cluster_normals = np.asarray(pcd.normals)[color_mask] if len(pcd.normals) > 0 else []

        # 创建新的点云对象
        cluster_pcd = o3d.geometry.PointCloud()
        cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points)
        cluster_pcd.colors = o3d.utility.Vector3dVector(cluster_colors)
        if len(cluster_normals) > 0:
            cluster_pcd.normals = o3d.utility.Vector3dVector(cluster_normals)
        if len(cluster_points) > 30:
            segments.append(cluster_pcd)
    results=[]
    for segment in segments:
        per_point_labels_segment, core_plane_points = ransac_plane_finder(
        pcd=segment, embeddings=embeddings,estimate_normals=False)

        filtered_points = np.asarray(segment.points)[core_plane_points]
        filtered_normals = np.asarray(segment.normals)[core_plane_points]
        filtered_colors = np.asarray(segment.colors)[core_plane_points]  # If color information is available

        #开始统一法向轴距
        average_normal = np.mean(filtered_normals, axis=0)
        average_normal=average_normal / np.linalg.norm(average_normal) 
        filtered_normals=np.tile(average_normal, (len(filtered_points), 1))
        distances = np.dot(filtered_points, average_normal)
        mean_distance = np.mean(distances)
        filtered_points= filtered_points - np.outer(distances - mean_distance, average_normal)
        
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
        filtered_pcd.normals = o3d.utility.Vector3dVector(filtered_normals)
        filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

        results.append(filtered_pcd)

    combined_pcd = o3d.geometry.PointCloud()
    for resulti in results:
        combined_pcd += resulti
    result = combined_pcd

    out_ply_file.parent.mkdir(exist_ok=True)
    o3d.io.write_point_cloud(str(out_ply_file), result)
    if 'points3d_pigo' in str(out_ply_file):
        print(f"PIGO point With Ransac regularized saved at {str(out_ply_file)}")
    else:
        print(f"MVSA point With Ransac regularized saved at {str(out_ply_file)}")


def run_ransac_on_scenes(
    num_harmonics: int,
    embedding_dim: int,
    force_assign_points: bool,
    embeddings_scale_factor: float,
    embeddings_inlier_threshold: float,
    normal_inlier_threshold: float,
    distance_inlier_threshold: float,
    merge_planes_with_similar_embeddings: bool,
    point_cloud:o3d.geometry.PointCloud,
    points3dpath:Path
) -> None:
    # for scene in tqdm(scenes, unit="scan", desc="All scans"):
    out_ply_file=points3dpath
    run_ransac_on_mesh(
        out_ply_file=out_ply_file,
        # embeddings_usage=embeddings_usage,
        embeddings=None,
        force_assign_points_to_planes=force_assign_points,
        normals_inlier_threshold=normal_inlier_threshold,
        distance_inlier_threshold=distance_inlier_threshold,
        embeddings_inlier_threshold=embeddings_inlier_threshold,
        merge_planes_with_similar_embeddings=merge_planes_with_similar_embeddings,
        point_cloud=point_cloud
    )

def merge_point_clouds(input_folder,down_sample=False):
    # 获取输入文件夹中的所有文件路径
    file_paths = [os.path.join(input_folder, file_name) for file_name in os.listdir(input_folder) if file_name.endswith('.ply')]

    # 初始化空的点云对象
    all_points = []
    all_colors = []
    all_normals = []

    # 遍历所有文件并读取点云
    for file_path in file_paths:
        # 读取点云文件
        pcd = o3d.io.read_point_cloud(file_path)

        # 获取点云的位置和颜色
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        normals= np.asarray(pcd.normals)


        if down_sample:    
        
            original_color = colors[0]
            original_normal = normals[0]

            points_ori = np.array(pcd.points)
            sample_rate = 0.2
            sample_num = int(len(points) * sample_rate)
            indices = np.random.choice(len(points), sample_num, replace=False)
            down_pcd = o3d.geometry.PointCloud()
            down_pcd.points = o3d.utility.Vector3dVector(points_ori[indices])

            num_points = np.asarray(down_pcd.points).shape[0]
            new_colors = np.tile(original_color, (num_points, 1))
            new_normals = np.tile(original_normal, (num_points, 1))
    
            # 重新赋值颜色和法线
            down_pcd.colors = o3d.utility.Vector3dVector(new_colors)
            down_pcd.normals = o3d.utility.Vector3dVector(new_normals)

        mean_value = np.mean(colors, axis=0)
        colors[:] = mean_value

        normals= np.asarray(pcd.normals)
        # 将位置和颜色添加到合并列表中
        if down_sample:
            all_points.append(np.asarray(down_pcd.points))
            all_colors.append(np.asarray(down_pcd.colors))
            all_normals.append(np.asarray(down_pcd.normals))
        else:
            all_points.append(points)
            all_colors.append(colors)
            all_normals.append(normals)

    # 合并所有点云
    all_points = np.concatenate(all_points, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)
    all_normals = np.concatenate(all_normals, axis=0)

    merged_pcd_1 = o3d.geometry.PointCloud()
    merged_pcd_1.points = o3d.utility.Vector3dVector(all_points)
    merged_pcd_1.colors = o3d.utility.Vector3dVector(all_colors)
    merged_pcd_1.normals = o3d.utility.Vector3dVector(all_normals)

    return merged_pcd_1


@click.command()
@click.option(
    "--num-harmonics",
    type=int,
    default=24,
)
@click.option(
    "--embedding-dim",
    type=int,
    default=3,
)
@click.option(
    "--force-assign-points",
    is_flag=True,
    help=(
        "If set, this will force all points to be assigned to a plane, even if they are not "
        "Assigned to a plane in the RANSAC step."
    ),
)
@click.option(
    "--embeddings-scale-factor",
    type=float,
    default=1.0,
)
@click.option(
    "--embeddings-inlier-threshold",
    type=float,
    default=0.0,
)
@click.option(
    "--normal-inlier-threshold",
    type=float,
    default=0.8,
)
@click.option(
    "--distance-inlier-threshold",
    type=float,
    default=0.1,
)
@click.option(
    "--merge-planes-with-similar-embeddings",
    is_flag=True,
)
@click.option(
    "--scene-name",
    type=str,
    is_flag=True,
)
@click.option(
    "--clustering-path",
    type=Path,
    required=True,

)
@click.option(
    "--points3d-path",
    type=Path,
    required=True,

)
@click.option(
    "--down-sample",
    is_flag=True,
)
def cli(
    num_harmonics: int,
    embedding_dim: int,
    force_assign_points: bool,
    embeddings_scale_factor: float,
    embeddings_inlier_threshold: float,
    normal_inlier_threshold: float,
    distance_inlier_threshold: float,
    merge_planes_with_similar_embeddings: bool,
    scene_name:str,
    clustering_path:Path,
    points3d_path:Path,
    down_sample:bool,
):
    """
    get ransac filtered point cloud from mvsa points
    """

    pcd_merged=merge_point_clouds(str(clustering_path),down_sample)
    if 'points3d_pigo' not in str(points3d_path):
        pcd_for_gs_path = str(points3d_path).replace('points3d', 'points3d_gs')
        o3d.io.write_point_cloud(pcd_for_gs_path, pcd_merged)
        print(f"MVSA point without Ransac regularized saved at {pcd_for_gs_path}")

    run_ransac_on_scenes(
        num_harmonics=num_harmonics,
        embedding_dim=embedding_dim,
        force_assign_points=force_assign_points,
        embeddings_scale_factor=embeddings_scale_factor,
        embeddings_inlier_threshold=embeddings_inlier_threshold,
        normal_inlier_threshold=normal_inlier_threshold,
        distance_inlier_threshold=distance_inlier_threshold,
        merge_planes_with_similar_embeddings=merge_planes_with_similar_embeddings,
        point_cloud=pcd_merged,
        points3dpath=points3d_path
    )


if __name__ == "__main__":
    cli()
