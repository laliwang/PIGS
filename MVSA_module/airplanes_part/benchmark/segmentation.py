"""
Instance evaluation benchmark.
This code is mostly based on PlanarRecon
https://github.com/neu-vi/PlanarRecon

Released under Apache-2.0 License
"""

import multiprocessing
from functools import partial
from pathlib import Path
from typing import Optional

import click
import numpy as np
from loguru import logger
from skimage.metrics import variation_of_information
from sklearn.metrics import rand_score
from tqdm import tqdm
import pandas as pd
import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

from airplanes_part.meshing.meshing_tools import squash_gt_vertices_onto_planes
from airplanes_part.segmentation.instance_projector import project_to_mesh
from airplanes_part.segmentation.segmentation_tools import compute_my_sc
from airplanes_part.utils.generic_utils import read_scannetv2_filename
from airplanes_part.utils.io_utils import AssetFileNames, load_gt_bundle, load_pred_bundle
from airplanes_part.utils.metrics_utils import ResultsAverager


@click.group()
def run():
    pass


def generate_scene_pred_plane_instances(scene_name: str, gt_root: Path, pred_root: Path, mesh_name: str) -> None:
    """
    This function prepares instance IDs for the predictions.
    params:
        scene_name: name of the scene to process
        gt_root: path to the root folder with ground meshes
        pred_root: path to root folder with all the prediction, scene by scene
    """
    gt_bundle = load_gt_bundle(
        scene_path=gt_root / scene_name,
        scene_name=scene_name,
    )
    pred_bundle = load_pred_bundle(scene_path=pred_root / scene_name, scene_name=scene_name, mesh_name=mesh_name)

    # compute pred indices from mesh colours
    pred_mesh = pred_bundle.planar_mesh
    pred_instance_ids = -np.ones((pred_mesh.vertices.shape[0],))

    for idx, col in enumerate(np.unique(pred_mesh.visual.vertex_colors, axis=0)):
        mask = (pred_mesh.visual.vertex_colors == col).all(axis=1)
        pred_instance_ids[mask] = idx

    if pred_instance_ids.min() < 0:
        raise ValueError(
            f"This is a bug! For scene {pred_bundle.scene}, there are vertices without a colour"
        )

    # double check that the scene is the same
    if pred_bundle.scene != gt_bundle.scene:
        raise ValueError(
            f"Error in the instance benchmark. "
            f"Pred scene ({pred_bundle.scene}) is different than gt scene ({gt_bundle.scene})"
        )

    # squash the ground truth mesh into its planar representation
    if gt_bundle.planes is None:
        raise ValueError(f"Cannot find planes.npy for scene {gt_bundle.path}.")
    squashed_mesh = squash_gt_vertices_onto_planes(mesh=gt_bundle.mesh, planes=gt_bundle.planes)
    gt_bundle.planar_mesh = squashed_mesh

    # project plane instances IDs
    mesh, plane_instances = project_to_mesh(
        from_mesh=pred_bundle.planar_mesh,
        to_mesh=gt_bundle.planar_mesh,
        attribute=pred_instance_ids,
        attr_name="plane_ins",
        color_mesh=pred_bundle.planar_mesh
    )

    #这里是让gt对齐predmesh
    # saving plane instances IDs
    np.savetxt(
        pred_root / scene_name / AssetFileNames.get_plane_instances_filename(scene_name),
        plane_instances,
        fmt="%d",
    )


def generate_scene_gt_plane_instances(scene_name: str, gt_root: Path) -> None:
    """
    This function prepares instance IDs for the ground truth.
    params:
        scene_name: name of the scene to process
        gt_root: path to the root folder with ground meshes
    """
    gt_bundle = load_gt_bundle(
        scene_path=gt_root / scene_name,
        scene_name=scene_name,
    )

    # squash the ground truth mesh into its planar representation
    if gt_bundle.planes is None:
        raise ValueError(f"Cannot find planes.npy for scene {gt_bundle.path}.")
    squashed_mesh = squash_gt_vertices_onto_planes(mesh=gt_bundle.mesh, planes=gt_bundle.planes)
    gt_bundle.planar_mesh = squashed_mesh

    # decode plane IDs that are encoded in colour channels
    plane_ids = gt_bundle.planar_mesh.visual.vertex_colors.copy().astype("int32")
    plane_ids = (plane_ids[:, 0] * 256 * 256 + plane_ids[:, 1] * 256 + plane_ids[:, 2]) // 100 - 1

    unique_id = np.unique(plane_ids)  # ascending order

    plane_instances: np.ndarray = np.zeros_like(plane_ids, dtype="uint32")  # type: ignore
    invalid_plane_id = 16777216 // 100 - 1  # invalid plane id #(255,255,255)

    # map all the plane ids in a range [1, N+1], where N is the number of unique plane IDs
    for k, id in enumerate(unique_id):
        if id != invalid_plane_id:
            plane_instances[plane_ids == id] = k + 1

    if plane_instances.max() < 1:
        raise ValueError(f"Cannot find any plane ID for scene {gt_bundle.scene}")

    plane_instances_label = np.zeros(plane_instances.shape, dtype="int32")
    sem_id = 1  # planes don't have a real semantic meaning, so this is set to a constant value

    # map each plane ID into a new ID and skip non-planar objects
    for instance_id in np.unique(plane_instances):
        if instance_id == 0:
            # this is not a plane, skip it
            #非平面区域对应的是之前的255，255，255颜色点云
            continue
        instance_mask = plane_instances == instance_id
        plane_instances_label[instance_mask] = sem_id * 1000 + instance_id

    np.savetxt(
        gt_root / scene_name / AssetFileNames.get_plane_instances_filename(scene_name),
        plane_instances_label,
        fmt="%d",
    )


def generate_instances(
    pred_root: Path, gt_root: Path, eval_scenes: list[str], num_processes: int, mesh_name: str
) -> None:
    """This function prepares instance IDs for the predictions.
    This will squash the gt mesh.

    Params:
        pred: path to root folder with all the prediction, scene by scene
        gt: path to the root folder with ground meshes
        eval_scenes: scenes to process.
    Returns:
        this function creates a new txt file called as the scene. The file contains the ID of
        the plane of each vertex in the mesh.
    """
    if len(eval_scenes) == 0:
        raise ValueError("Cannot find any eval scene. Please check the validation file")
   # logger.info(f"Running on {len(eval_scenes)} eval scenes")

    mp_ctx = multiprocessing.get_context(method="spawn")
    with mp_ctx.Pool(processes=num_processes) as pool:
        # Create a partial function with the arguments that will be the same for all processes
        partial_worker = partial(
            generate_scene_pred_plane_instances,
            gt_root=gt_root,
            pred_root=pred_root,
            mesh_name=mesh_name,
        )

        for _ in tqdm(pool.imap_unordered(partial_worker, eval_scenes), total=len(eval_scenes)):
            continue


def generate_gt_instances(gt_root: Path, eval_scenes: list[str], num_processes: int) -> None:
    """Extract plane IDs from the ground truth mesh."""
    if len(eval_scenes) == 0:
        raise ValueError("Cannot find any eval scene. Please check the validation file")

   # logger.info(f"Running on {len(eval_scenes)} eval scenes")

    mp_ctx = multiprocessing.get_context(method="spawn")
    with mp_ctx.Pool(processes=num_processes) as pool:
        # Create a partial function with the arguments that will be the same for all processes
        partial_worker = partial(
            generate_scene_gt_plane_instances,
            gt_root=gt_root,
        )

        for _ in tqdm(pool.imap_unordered(partial_worker, eval_scenes), total=len(eval_scenes)):
            continue


def check_scene_instance_benchmark_files(scene_name: str, gt_root: Path, pred_root: Path):
    """
    Check if files exist before running the benchmark for a specific scene. This is a good check
    in case a file is missing and it kills a long benchmark run.
    """
    scene_path = gt_root / scene_name
    # check gt mesh data
    gt_instances_path = scene_path / AssetFileNames.get_plane_instances_filename(scene_name)
    if not gt_instances_path.exists():
        raise ValueError(
            f"Cannot find gt instances for scene {scene_name}. Path: {gt_instances_path}."
            f"Have you generated instances?"
        )

    # new check predicted mesh data, assume we're only interested in a planar mesh
    scene_path = pred_root / scene_name
    pred_instances_path = scene_path / AssetFileNames.get_plane_instances_filename(scene_name)
    if not pred_instances_path.exists():
        raise ValueError(
            f"Cannot find pred instances for scene {scene_name}. Path: {pred_instances_path}"
        )


def check_instance_benchmark_files(
    scenes: list[str],
    gt_root: Path,
    pred_root: Path,
):
    """
    Check if files exist before running the benchmark. This is a good check
    in case a file is missing and it kills a long benchmark run.
    """
    for scene_name in scenes:
        check_scene_instance_benchmark_files(
            scene_name=scene_name, gt_root=gt_root, pred_root=pred_root
        )


def process_scene_instance_metrics(
    scene_name: str,
    gt_root: Path,
    pred_root: Path,
    mesh_name: str,
) -> dict[str, float]:
    """
    This function computes the instance metrics for a single scene.
    params:
        scene_name: name of the scene to process
        gt_root: path to the root folder with ground truth meshes
        pred_root: path to root folder with all the prediction, scene by scene
    """

    # load data
    gt_bundle = load_gt_bundle(
        scene_path=gt_root / scene_name,
        scene_name=scene_name,
    )
    pred_bundle = load_pred_bundle(scene_path=pred_root / scene_name, scene_name=scene_name, mesh_name=mesh_name)

    # double check that the scene is the same
    if gt_bundle.scene != pred_bundle.scene:
        raise ValueError(
            f"Error in the instance benchmark. "
            f"Pred scene ({pred_bundle.scene}) is different than gt scene ({gt_bundle.scene})"
        )

    # compute scores
    unique_gt, count_gt = np.unique(gt_bundle.instances, return_counts=True)
    unique_pred, count_pred = np.unique(pred_bundle.instances, return_counts=True)
    inst_num_gt = unique_gt.shape[0]
    inst_num_pred = unique_pred.shape[0]
    # print(f'gt_instances length: {inst_num_gt}, less than 200 count: {np.sum(count_gt < 200)}')
    # print(f'pred_instances length: {inst_num_pred}, less than 200 count: {np.sum(count_pred < 200)}')
    icr = (inst_num_pred) / (inst_num_gt-np.sum(count_gt < 200))

    ri = rand_score(labels_true=gt_bundle.instances, labels_pred=pred_bundle.instances)
    h1, h2 = variation_of_information(image0=gt_bundle.instances, image1=pred_bundle.instances)
    voi = h1 + h2
    sc = compute_my_sc(gt_in=gt_bundle.instances, pred_in=pred_bundle.instances)
    metrics = {"voi↓": voi, "ri↑": ri, "sc↑": sc, "icr↓": icr}

    return metrics


def multiprocess_instance_metrics(
    eval_scenes: list[str],
    gt_root: Path,
    pred_root: Path,
    benchmark_meter: ResultsAverager,
    num_processes: int = 8,
    mesh_name: str = "Air",
    save_excel:bool=False,
    Scene_name:str="scene0645_00"
) -> None:
    """
    This function computes the instance metrics for all the scenes.
    params:
        eval_scenes: list of scene names to process
        gt_root: path to the root folder with ground truth meshes
        pred_root: path to root folder with all the prediction, scene by scene
        benchmark_meter: ResultsAverager object to store results
        num_processes: number of processes to use for multiprocessing
    """

    mp_ctx = multiprocessing.get_context(method="spawn")
    with mp_ctx.Pool(processes=num_processes) as pool:
        # Create a partial function with the arguments that will be the same for all processes
        partial_worker = partial(
            process_scene_instance_metrics, gt_root=gt_root, pred_root=pred_root, mesh_name=mesh_name,
        )
        results = []
        for metrics in tqdm(
            pool.imap_unordered(partial_worker, eval_scenes), total=len(eval_scenes)
        ):
            results.append(metrics)
            benchmark_meter.update_results(elem_metrics=metrics)

    return metrics
    


def instance_benchmark_multiprocess(
    eval_scenes: list[str],
    gt_root: Path,
    pred_root: Path,
    benchmark_meter: ResultsAverager,
    num_processes: int = 32,
    mesh_name: str = "Air",
    save_excel:bool=False,
    Scene_name:str="scene0645_00"
):
    """
    This function computes the instance metrics for all the scenes using multiprocessing.
    params:
        eval_scenes: list of scene names to process
        gt_root: path to the root folder with ground truth meshes
        pred_root: path to root folder with all the prediction, scene by scene
        benchmark_meter: ResultsAverager object to store results
    """

   # logger.info("Generating gt instances")
    generate_gt_instances(eval_scenes=eval_scenes, gt_root=gt_root, num_processes=num_processes)

   # logger.info("Generating pred instances")
    generate_instances(
        eval_scenes=eval_scenes, gt_root=gt_root, pred_root=pred_root, num_processes=num_processes, mesh_name=mesh_name,
    )

    # check if instances files exist
    check_instance_benchmark_files(
        scenes=eval_scenes,
        gt_root=gt_root,
        pred_root=pred_root,
    )

   # logger.info("Computing metrics")
    multiprocess_instance_metrics(
        eval_scenes=eval_scenes,
        gt_root=gt_root,
        pred_root=pred_root,
        benchmark_meter=benchmark_meter,
        num_processes=num_processes,
        mesh_name=mesh_name,
        save_excel=save_excel,
        Scene_name=Scene_name
    )


@run.command()
@click.option(
    "--pred-root",
    type=Path,
    help="Path to predicted plane IDs",
)
@click.option(
    "--gt-root",
    type=Path,
    help="Path to ground-truth plane IDs",
)
@click.option(
    "--validation-file",
    type=Path,
    default=Path("src/airplanes/data_splits/ScanNetv2/standard_split/scannetv2_xxx.txt"),
    help="Path to the file that contains the test scenes",
)
@click.option(
    "--output-score-dir",
    type=Path,
    default=None,
    help="Where do you want to save final scores?",
)
@click.option(
    '--mesh-name',
    type=str,
    default='Air',
    help="Specific mesh name for evaluation and metric computation"
)
@click.option(
    "--save-excel",
    type=bool,
    is_flag=True,
    help="Wether or not to save excel",
)
@click.option(
    '--scene-name',
    help="Name of the planar mesh to be evaluated",
    type=str,
    default='Scene0653_00'
)
def run_instance_benchmark(
    pred_root: Path,
    gt_root: Path,
    validation_file: Path,
    output_score_dir: Optional[Path],
    mesh_name: str,
    save_excel: bool=False,
    scene_name:str='Scene0653_00'
):
    """
    This function computes the instance metrics for all the scenes.
    params:
        pred_root: path to root folder with all the prediction, scene by scene
        gt_root: path to the root folder with ground truth meshes
        validation_file: path to the file that contains the list of sequences to use. Only the
            sequences included in this list will be processed.
        output_score_dir: where to save the final scores
        num_processes: number of processes to use for multiprocessing
    """

    # select only eval scenes
    # eval_scenes = read_scannetv2_filename(filepath=validation_file)
    eval_scenes = [scene_name]

    if len(eval_scenes) == 0:
        raise ValueError("Cannot find any eval scene. Please check the validation file")
   # logger.info(f"Running on {len(eval_scenes)} eval scenes")

    benchmark_meter = ResultsAverager(
        exp_name="instance benchmark", metrics_name="segmentation metrics"
    )

    instance_benchmark_multiprocess(
        eval_scenes=eval_scenes,
        gt_root=gt_root,
        pred_root=pred_root,
        benchmark_meter=benchmark_meter,
        mesh_name=mesh_name,
        save_excel=save_excel,
        Scene_name=scene_name
    )

    # printing out results
    benchmark_meter.compute_final_average()
    benchmark_meter.print_sheets_friendly(
        include_metrics_names=True,
        print_running_metrics=False,
    )
    if output_score_dir is not None:
        output_score_dir.mkdir(parents=True, exist_ok=True)
        benchmark_meter.output_json(str(output_score_dir / f"Seg_{scene_name}.json"), mesh_name)


if __name__ == "__main__":
    # run()  # type: ignore
    run_instance_benchmark()
