from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

import pandas as pd
import click
import numpy as np
import open3d as o3d
import ray
import torch
from loguru import logger
from tqdm import tqdm
import sys
import os


sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

from airplanes_part.meshing.meshing_tools import (
    compute_point_cloud_metrics,
    find_nearest_neighbors,
    squash_gt_vertices_onto_planes,
    subsample_mesh,
)
# from airplanes_part.utils.generic_utils import read_scannetv2_filename
from airplanes_part.utils.io_utils import (
    AssetFileNames,
    MeshingBundle,
    load_gt_bundle,
    load_pred_bundle,
)
from airplanes_part.utils.metrics_utils import ResultsAverager


@dataclass
class MetricParams:
    # function that is used to compute metrics. It depends on the type of metric we want to compute
    metric_computer_fnc: Callable

    # num_samples: number of points to sample from the mesh
    num_samples: int

    # max_dist: maximum distance to clip distances to in meter
    max_dist: float

    # dist_threshold: distance threshold to use for precision and recall in meters
    dist_threshold: float

    # k: number of ground truth planes to use for the top planes metrics
    k: int


def compute_planar_mesh_metrics_for_scene(
    gt_bundle: MeshingBundle, pred_bundle: MeshingBundle, params: MetricParams,Scene_name:str,save_excel:bool,mesh_name:str,pred_root:str,
):
    """
    Compute metrics for a single scene on planar meshes.
    Params:
        gt_bundle: MeshingBundle for the gt mesh
        pred_bundle: MeshingBundle for the predicted mesh
        params: various parameters for the metrics computations
    """
    # double check that the scene is the same
    if gt_bundle.scene != pred_bundle.scene:
        raise ValueError(
            f"Error in the benchmark. "
            f"Pred scene ({pred_bundle.scene}) is different than gt scene ({gt_bundle.scene})"
        )

    # squash the ground-truth mesh
    gt_mesh = gt_bundle.mesh
    gt_planes = gt_bundle.planes
    #gt_planes是以numpy数组形式存储的理想平面

    if gt_planes is None:
        raise ValueError(
            f"Cannot find the plane npy file for scene {gt_bundle.scene}."
            f"Planes are required when -squash-gt is set"
            f"Expecting it at {gt_bundle.scene_root / AssetFileNames.get_planes_filename(gt_bundle.scene)}"
        )
    gt_mesh = squash_gt_vertices_onto_planes(mesh=gt_mesh, planes=gt_planes)
    gt_bundle.planar_mesh = gt_mesh
    gt_mesh_to_use = gt_bundle.planar_mesh

    # sample the meshes to get point clouds
    assert gt_mesh_to_use is not None, "Cannot find gt mesh"
    assert pred_bundle.planar_mesh is not None, "Cannot find planar mesh"

    gt_mesh_o3d = gt_mesh_to_use.as_open3d
    pred_mesh_o3d = pred_bundle.planar_mesh.as_open3d
    
    # explicitly update colours as well
    pred_colours = pred_bundle.planar_mesh.visual.vertex_colors[:, :3].astype("float64")
    pred_mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(pred_colours / 255.0)

    # we want to mask out points from faces which are invalid (i.e. span 2 planes).
    # we store this info in the red colour channel since open3d doesn't accept an alpha channel
    gt_colours = gt_mesh_to_use.visual.vertex_colors
    gt_colours[:, 0] = gt_colours[:, 3]
    gt_mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(
        gt_colours[:, :3].astype("float64") / 255.0
    )

    if mesh_name == 'final' or mesh_name == 'pigs':
        o3d.io.write_triangle_mesh(os.path.join(pred_root, 'mesh_gt.ply'), gt_mesh_o3d )

    gt_pcd = subsample_mesh(mesh=gt_mesh_o3d, num_samples=params.num_samples)
    pred_pcd = subsample_mesh(mesh=pred_mesh_o3d, num_samples=params.num_samples)

    # trim the prediction using a visibility volume
    visible_pred_indices = None

    # # try to do not load the visibility volume
    # assert gt_bundle.visibility_volume is not None, "Cannot find visibility volume"

    if gt_bundle.visibility_volume is not None:
        print(f"Visibility_volumes Loaded from {str(gt_bundle.visibility_path)}")
        # move the volume to gpu
        gt_bundle.visibility_volume.cuda()
        # get the raw points from the pred pcd
        pcd_points_N3 = torch.tensor(np.array(pred_pcd.points)).float().cuda()
        # sample the volume at those pred points to figure out if they're visible
        vis_samples_N = gt_bundle.visibility_volume.sample_volume(world_points_N3=pcd_points_N3)
        valid_mask_N = vis_samples_N > 0.5
        # get visible indices
        visible_pred_indices = valid_mask_N.nonzero().squeeze().cpu().numpy().tolist()

        # compute visible_gt_indices is necessary
        pcd_points_N3_gt = torch.tensor(np.array(gt_pcd.points)).float().cuda()
        vis_samples_N_gt = gt_bundle.visibility_volume.sample_volume(world_points_N3=pcd_points_N3_gt)
        valid_mask_N_gt = vis_samples_N_gt > 0.5
        visible_gt_indices = valid_mask_N_gt.nonzero().squeeze().cpu().numpy().tolist()

        # move the volume back to cpu to save memory
        gt_bundle.visibility_volume.cpu()


        

    # compute scores
    metrics = params.metric_computer_fnc(
        gt_pcd=gt_pcd,
        pred_pcd=pred_pcd,
        params=params,
        visible_pred_indices=visible_pred_indices,
        visible_gt_indices=visible_gt_indices,
        Scene_name=Scene_name,
        save_excel=save_excel,
        mesh_name=mesh_name
    )
    #       metrics["acc↓"] = max_dist
    #metrics["compl↓"] = max_dist
    #metrics["chamfer↓"] = max_dist
    #metrics["precision↑"] = 0.0
    #metrics["recall↑"] = 0.0
    #metrics["f1_score↑"] = 0.0
    return metrics


def compute_point_cloud_metrics_aux(
    gt_pcd: o3d.geometry.PointCloud,
    pred_pcd: o3d.geometry.PointCloud,
    params: MetricParams,
    Scene_name:str,
    save_excel:bool,
    mesh_name:str,
    visible_pred_indices: Optional[list[int]] = None,
    visible_gt_indices: Optional[list[int]] = None,
) -> Dict[str, float]:
    metrics, _, _ = compute_point_cloud_metrics(
        gt_pcd=gt_pcd,
        pred_pcd=pred_pcd,
        max_dist=params.max_dist,
        dist_threshold=params.dist_threshold,
        visible_pred_indices=visible_pred_indices,
    )
    
    return metrics


def compute_planar_metrics(
    gt_pcd: o3d.geometry.PointCloud,
    pred_pcd: o3d.geometry.PointCloud,
    params: MetricParams,
    Scene_name:str,
    save_excel:bool,
    mesh_name:str,
    visible_pred_indices: Optional[list[int]] = None,
    visible_gt_indices: Optional[list[int]] = None,
) -> Dict[str, float]:
    """Compute metrics for a predicted and gt point cloud.

    If the predicted point cloud is empty, all the lower-is-better metrics will be set to max_dist
    and all the higher-is-better metrics to 0.

    Args:
        gt_pcd (o3d.geometry.PointCloud): gt point cloud.
            Note that the red channel of gt_pcd.colors contains information about points that
            span more than one plane
        pred_pcd (o3d.geometry.PointCloud): predicted point cloud, will be compared to gt_pcd.
        params: various param for metric computation
        visible_pred_indices (list[int], optional): Indices of the predicted points that are
            visible in the scene. Defaults to None. When not None will be used to filter out
            predicted points when computing pred to gt.
            Note: these isn't being used in this function, but is part of the input, so that
            both functions have consistent signatures

    Returns:
        dict[str, float]: Metrics for this point cloud comparison.
    """

    metrics: Dict[str, float] = {}

    if len(pred_pcd.points) == 0:
        metrics["top_planes_compl↓"] = params.max_dist
        metrics["top_planes_acc↓"] = params.max_dist

        return metrics

    pred_points = np.array(pred_pcd.points)
    pred_colors = np.array(pred_pcd.colors)
    pred_colors = np.rint(pred_colors * 255)

    if visible_pred_indices is not None:
        pred_points = pred_points[visible_pred_indices]
        pred_colors = pred_colors[visible_pred_indices]

    pred_unique_colors, pred_counts = np.unique(pred_colors, axis=0, return_counts=True)
    pred_planes_pts = []
    for color, count in zip(pred_unique_colors, pred_counts):
        # removing small predicted planes, for speed only.
        #多于20个点的平面才被保留
        if count > 20:
            pred_planes_pts.append(pred_points[(pred_colors == color).all(axis=1)])

    gt_colors = np.array(gt_pcd.colors)
    gt_points = np.array(gt_pcd.points)

    # the red channel stores points that span 2 or more planes
    # these are now removed from the ground truth
    #R通道为0的点要被删
    valid_gt = gt_colors[:, 0] > 0.01
    gt_colors = gt_colors[valid_gt]
    gt_points = gt_points[valid_gt]
    # discard the first channel since it only encoded visibility
    gt_colors = np.rint(gt_colors[:, 1:] * 255)
    #rint是一种取整，不过这里把R通道直接去掉不影响吗

    gt_unique_colors, gt_counts = np.unique(gt_colors, axis=0, return_counts=True)

    # sort gt planes by number of sampled points
    #从大到小排序gt的平面点云
    sorted_gt_planes = sorted(
        zip(gt_counts, gt_unique_colors), reverse=True, key=lambda tup: tup[0]
    )
    #只选取前20名的gt点云
    sorted_gt_planes = sorted_gt_planes[: params.k]

    final_accuracy = 0
    final_completion = 0
    final_count = 0

    for count, color in sorted_gt_planes:
        gt_plane_pts = gt_points[(gt_colors == color).all(axis=1)]

        best_completion = 1000 * params.max_dist
        best_matched_plane = -1

        # find closest predicted plane
        #先寻找到了和当前gt平面距离最近的pred平面
        for idx, pred_plane_pts in enumerate(pred_planes_pts):
            distances_gt2pred, _ = find_nearest_neighbors(
                gt_plane_pts, pred_plane_pts, params.max_dist
            )

            completion = float(np.mean(distances_gt2pred))
            if completion < best_completion:
                best_completion = completion
                best_matched_plane = idx

        distances_pred2gt, _ = find_nearest_neighbors(
            pred_planes_pts[best_matched_plane], gt_plane_pts, params.max_dist
        )
        accuracy = float(np.mean(distances_pred2gt))
        if accuracy > params.max_dist or best_completion > params.max_dist:
            raise ValueError("Accuracy or completion are larger than max_dist")
        final_accuracy += accuracy * count
        final_completion += best_completion * count
        final_count += count

    metrics["top_planes_compl↓"] = final_completion / final_count
    metrics["top_planes_acc↓"] = final_accuracy / final_count

    return metrics


def compute_scale_aware_planar_metrics(
    gt_pcd: o3d.geometry.PointCloud,
    pred_pcd: o3d.geometry.PointCloud,
    params: MetricParams,
    Scene_name: str,
    save_excel: bool,
    mesh_name: str,
    visible_pred_indices: Optional[list[int]] = None,
    visible_gt_indices: Optional[list[int]] = None,
) -> Dict[str, float]:
    """
    以预测平面(pred)为基准，计算相对于真值(gt)的几何指标。
    """

    metrics: Dict[str, float] = {}

    if len(pred_pcd.points) == 0:
        metrics["large_planes_compl↓"] = params.max_dist * 100
        metrics["large_planes_acc↓"] = params.max_dist * 100
        metrics["medium_planes_compl↓"] = params.max_dist * 100
        metrics["medium_planes_acc↓"] = params.max_dist * 100
        return metrics

    # --- 1. 数据预处理 ---
    pred_points = np.array(pred_pcd.points)
    pred_colors = np.array(pred_pcd.colors)
    pred_colors = np.rint(pred_colors * 255)

    if visible_pred_indices is not None:
        pred_points = pred_points[visible_pred_indices]
        pred_colors = pred_colors[visible_pred_indices]

    gt_points = np.array(gt_pcd.points)
    gt_colors = np.array(gt_pcd.colors)

    if visible_gt_indices is not None:
        gt_points = gt_points[visible_gt_indices]
        gt_colors = gt_colors[visible_gt_indices]
    
    # 按照原逻辑处理 GT 的 red 通道（可见性/重叠过滤）
    valid_gt = gt_colors[:, 0] > 0.01
    gt_points = gt_points[valid_gt]
    gt_colors = np.rint(gt_colors[valid_gt][:, 1:] * 255) 

    # --- 2. 对 PRED 平面进行规模排序 ---
    pred_unique_colors, pred_counts = np.unique(pred_colors, axis=0, return_counts=True)
    
    # 过滤掉点数过少的噪声平面
    min_pts_pred = 200 
    sorted_pred_planes = sorted(
        [(count, color) for count, color in zip(pred_counts, pred_unique_colors) if count >= min_pts_pred],
        reverse=True, key=lambda tup: tup[0]
    )

    if not sorted_pred_planes:
        return metrics

    # 定义 Pred 平面的 Large 和 Medium 组
    num_k = params.k
    pred_groups = {
        # "large": sorted_pred_planes[:num_k],
        # "medium": sorted_pred_planes[len(sorted_pred_planes)//2 - num_k//2 : len(sorted_pred_planes)//2 + num_k//2]
        "medium": sorted_pred_planes[len(sorted_pred_planes)//2 -num_k//2 : len(sorted_pred_planes)//2 + num_k//2],
        "small": sorted_pred_planes[-num_k:]
    }

    # 获取 GT 的所有平面点云备用
    gt_unique_colors = np.unique(gt_colors, axis=0)
    all_gt_planes_pts = [gt_points[(gt_colors == color).all(axis=1)] for color in gt_unique_colors]
    all_gt_planes_pts = [pts for pts in all_gt_planes_pts if len(pts) > 20] # 过滤掉过小的GT

    # --- 3. 核心计算：以 Pred 查找 GT ---
    for group_name, pred_list in pred_groups.items():
        group_accuracy_sum = 0
        group_completion_sum = 0
        group_total_weight = 0

        for count, color in pred_list:
            # 以当前 Pred 平面的点数作为权重
            weight = count 
            curr_pred_pts = pred_points[(pred_colors == color).all(axis=1)]
            
            best_accuracy = 1000 * params.max_dist
            best_matched_gt_idx = -1

            # A. 寻找与当前 Pred 平面几何最接近的 GT 平面 (计算 Accuracy)
            for idx, gt_pts in enumerate(all_gt_planes_pts):
                # 计算 Pred 到 GT 的平均距离
                dist_pred2gt, _ = find_nearest_neighbors(curr_pred_pts, gt_pts, params.max_dist)
                acc = float(np.mean(dist_pred2gt))
                
                if acc < best_accuracy:
                    best_accuracy = acc
                    best_matched_gt_idx = idx

            if best_matched_gt_idx == -1:
                continue

            # B. 计算该匹配对的 Completion (GT 到 Pred 的距离)
            matched_gt_pts = all_gt_planes_pts[best_matched_gt_idx]
            dist_gt2pred, _ = find_nearest_neighbors(matched_gt_pts, curr_pred_pts, params.max_dist)
            completion = float(np.mean(dist_gt2pred))

            # # 规模匹配约束：如果匹配到的 GT 太大，而在计算中型平面，可能需要跳过
            # gt_size = matched_gt_pts.shape[0]
            # if (group_name == 'medium') and (gt_size > 5 * count):
            #     continue

            if best_accuracy > params.max_dist or completion > params.max_dist:
                # 若误差过大，直接跳过更合适一些
                # best_accuracy = params.max_dist
                # completion = params.max_dist
                continue

            group_accuracy_sum += best_accuracy * weight
            group_completion_sum += completion * weight
            group_total_weight += weight

        # --- 4. 汇总结果 ---
        if group_total_weight > 0:
            metrics[f'{group_name}_planes_compl↓'] = (group_completion_sum / group_total_weight) * 100
            metrics[f'{group_name}_planes_acc↓'] = (group_accuracy_sum / group_total_weight) * 100

    return metrics


def process_scene(
    scene_name: str,
    gt_root: Path,
    pred_root: Path,
    params: MetricParams,
    mesh_name: str,
    Scene_name:str,
    save_excel:bool
):
    """
    Process a single scene
    Params:
        scene_name: name of the scene to process
        gt_root: root directory for gt meshes
        pred_root: root directory for pred meshes
        params: various parameters for the metrics computations
    """

    gt_bundle = load_gt_bundle(
        scene_path=gt_root / scene_name,
        scene_name=scene_name,
        load_visibility_volumes=True,
    )
    pred_bundle = load_pred_bundle(scene_path=pred_root / scene_name, scene_name=scene_name, mesh_name=mesh_name)

    # compute metrics
    metrics = compute_planar_mesh_metrics_for_scene(
        gt_bundle=gt_bundle,
        pred_bundle=pred_bundle,
        params=params,
        Scene_name=Scene_name,
        save_excel=save_excel,
        mesh_name=mesh_name,
        pred_root=pred_root,
    )
    return metrics


# dataclass that contains all the args for process_scene
@dataclass
class ProcessSceneArgs:
    scene_name: str
    gt_root: Path
    pred_root: Path
    params: MetricParams


def multigpu_scenes(
    scenes: list[str],
    gt_root: Path,
    pred_root: Path,
    params: MetricParams,
    benchmark_meter: ResultsAverager,
    num_cpus: Optional[int] = None,
    num_gpus: Optional[int] = None,
    num_workers_per_gpu: int = 8,
    mesh_name: str = "Air",
    Scene_name: str ="Scene_0653_00",
    save_excel: bool =False,
    ray_tmp_dir: str = ""
):
    """
    Process all the scenes using ray
    Params:
        scenes: list of scene names to process
        gt_root: root path where gt scene folders live
        pred_root: root path where pred scene folders live
        params: various parameters for the metrics computations
        benchmark_meter: ResultsAverager to store the results
        num_cpus: number of cpus to use for ray. will default to using all available CPUs if not specified.
        num_gpus: number of gpus to use for ray. will default to using all available GPUs if not specified.
        num_workers_per_gpu: number of workers to use per GPU. This is used to set the num_gpus argument for ray.remote
    """

    # Initialize ray if not already initialized
    ray_already_initialized = ray.is_initialized()
    if not ray_already_initialized:
        # Initialize ray with GPUs
        ray.init(ignore_reinit_error=True, num_cpus=num_cpus, num_gpus=num_gpus, _temp_dir=ray_tmp_dir)

    # Define a remote function
    # NOTE: fractional num_gpus allow for multiple tasks to run on a single GPU
    # num_gpus=0.125 means that upto 8 tasks can run on a single GPU simultaneously
    @ray.remote(num_gpus=1 / num_workers_per_gpu)
    def gpu_worker_function(args: ProcessSceneArgs):
        metrics = process_scene(
            scene_name=args.scene_name,
            gt_root=args.gt_root,
            pred_root=args.pred_root,
            params=params,
            mesh_name=mesh_name,
            Scene_name=Scene_name,
            save_excel=save_excel
        )
        return metrics

    # Create a list of futures
    gpu_futures = []
    for scene_name in scenes:
        process_scene_args = ProcessSceneArgs(
            scene_name=scene_name,
            gt_root=gt_root,
            pred_root=pred_root,
            params=params,
        )
        gpu_futures.append(gpu_worker_function.remote(process_scene_args))

    # Progress bar with tqdm
    pbar = tqdm(total=len(gpu_futures), desc="Processing", dynamic_ncols=True)

    # Loop until all futures are done
    while len(gpu_futures) > 0:
        # Check if any future is ready
        ready_futures, remaining_futures = ray.wait(gpu_futures, timeout=0.1)

        # Get the results from the ready futures
        gpu_results = ray.get(ready_futures)
        # Update results
        for metrics in gpu_results:
            benchmark_meter.update_results(elem_metrics=metrics)

        # Update progress bar
        pbar.update(len(ready_futures))

        # Update the list of futures
        gpu_futures = remaining_futures

    # Close progress bar
    pbar.close()

    # Shutdown ray if it was not already initialized
    if not ray_already_initialized:
        ray.shutdown()


def check_meshing_benchmark_files(
    scenes: list[str],
    gt_root: Path,
    pred_root: Path,
    mesh_name: str,
):
    """
    Check if files exist before running the benchmark. This is a good check
    in case a file is missing and it kills a long benchmark run.
    """
    for scene_name in scenes:
        scene_path = gt_root / scene_name

        # check gt mesh data
        gt_mesh_path = scene_path / AssetFileNames.get_gt_mesh_filename(scene_name)
        if not gt_mesh_path.exists():
            raise ValueError(f"Cannot find gt mesh for scene {scene_name}. Path: {gt_mesh_path}")

        gt_planes_path = scene_path / AssetFileNames.get_planes_filename(scene_name)
        if not gt_planes_path.exists():
            raise ValueError(
                f"Cannot find gt planes for scene {scene_name}. Path: {gt_planes_path}"
            )

        # new check predicted mesh data, assume we're only interested in a planar mesh
        scene_path = pred_root / scene_name
        pred_planar_mesh_path = scene_path / (AssetFileNames.get_planar_mesh_filename(scene_name)).replace('.ply', f'_{mesh_name}.ply')
        if not pred_planar_mesh_path.exists():
            raise ValueError(
                f"Cannot find pred planar mesh for scene {scene_name}. Path: {pred_planar_mesh_path}"
            )


@click.command()
@click.option(
    "--pred-root",
    type=Path,
    required=True,
    help="Path to predicted meshes",
)
@click.option(
    "--gt-root",
    type=Path,
    required=True,
    help="Path to ground-truth meshes",
)
@click.option("--n", type=int, default=200_000, help="number of points to sample from the mesh")
@click.option(
    "--max-dist", type=float, default=1, help="Maximum distance to clip distances to in meter"
)
@click.option(
    "--dist-threshold",
    type=float,
    default=0.05,
    help="Distance threshold to use for precision and recall in meters",
)
@click.option(
    "--output-score-dir",
    type=Path,
    default=None,
    help="Where do you want to save final scores?",
)
@click.option(
    "--validation-file",
    type=Path,
    default=Path("src/airplanes/data_splits/ScanNetv2/standard_split/scannetv2_xxx.txt"),
    help="Path to the file that contains test scenes",
)
@click.option(
    "--use-planar-metrics",
    type=bool,
    is_flag=True,
    help="Wether or not to compute metrics for the top planes, as opposed to the normal mesh metrics",
)
@click.option(
    "--scale-aware-metrics",
    type=bool,
    is_flag=True,
    help="whether or not to compute metrics for top, medium, and small planes, as described in PIGS.",
)
@click.option(
    "--save-excel",
    type=bool,
    is_flag=True,
    help="Wether or not to save excel",
)
@click.option(
    "--k",
    type=int,
    default=20,
    help="Number of planes to consider per scene. Only used if top_plane_metrics flag is also used",
)
@click.option(
    '--mesh-name',
    type=str,
    default='Air',
    help="Specific mesh name for evaluation and metric computation"
)
@click.option(
    '--scene-name',
    help="Name of the planar mesh to be evaluated",
    type=str,
    default='Scene0653_00'
)
@click.option(
    '--ray-tmp-dir',
    help="path-to-ray-tmp-dir",
    type=str,
    default='/tmp/ray'
)
def cli(
    pred_root: Path,
    gt_root: Path,
    n: int,
    max_dist: float,
    dist_threshold: float,
    output_score_dir: Optional[Path],
    validation_file: Path,
    use_planar_metrics: bool = False,
    scale_aware_metrics: bool = False,
    k: int = 20,
    mesh_name: str = 'Air',
    save_excel: bool = False,
    scene_name:str='Scene0653_00',
    ray_tmp_dir: str = '/tmp/ray'
):
    # select only eval scenes
    # scenes = read_scannetv2_filename(validation_file)
    # eval_scenes = list(set(scenes))
    eval_scenes = [scene_name]

    if len(eval_scenes) == 0:
        raise ValueError("Cannot find any scene. Please check the validation file")

    # run a quick check to make sure we have the files we need.
    check_meshing_benchmark_files(scenes=eval_scenes, gt_root=gt_root, pred_root=pred_root, mesh_name=mesh_name)

    benchmark_meter = ResultsAverager(exp_name="meshing benchmark", metrics_name="scene metrics")

    metric_computer_fnc: Callable
    if scale_aware_metrics:
        metric_computer_fnc = compute_scale_aware_planar_metrics
    elif use_planar_metrics:
        metric_computer_fnc = compute_planar_metrics
    else:
        metric_computer_fnc = compute_point_cloud_metrics_aux

    params = MetricParams(
        metric_computer_fnc=metric_computer_fnc,
        num_samples=n,
        max_dist=max_dist,
        dist_threshold=dist_threshold,
        k=k,
    )

    multigpu_scenes(
        scenes=eval_scenes,
        gt_root=gt_root,
        pred_root=pred_root,
        params=params,
        benchmark_meter=benchmark_meter,
        num_cpus=None,  # use all CPUs
        num_gpus=None,  # use all GPUs
        mesh_name=mesh_name,
        Scene_name=scene_name,
        save_excel=save_excel,
        ray_tmp_dir=ray_tmp_dir
    )

    # printing out results
    benchmark_meter.compute_final_average()
    benchmark_meter.print_sheets_friendly(
        include_metrics_names=True,
        print_running_metrics=False,
    )
    if output_score_dir is not None:
        # logger.info(f"Saving results to: {output_score_dir}")
        output_score_dir.mkdir(parents=True, exist_ok=True)
        if scale_aware_metrics:
            benchmark_meter.output_json(str(output_score_dir / f"Scale_{scene_name}.json"), mesh_name)
        elif use_planar_metrics:
            benchmark_meter.output_json(str(output_score_dir / f"Planar_{scene_name}.json"), mesh_name)
        else:
            benchmark_meter.output_json(str(output_score_dir / f"Meshing_{scene_name}.json"), mesh_name)


if __name__ == "__main__":
    cli()  # type: ignore
