# Sequential Pipeline of PIGS on ScanNetV2

Here is the detailed sequential pipeline of PIGS on a specific scene from ScanNetV2.

## 📋 Table of Contents

* [0. Data Acquisition and Configuration](#0-data-acquisition-and-configuration)
* [1. GHPS Module: Geometry-Hinted Planar Segmentor](#1-ghps-module-geometry-hinted-planar-segmentor)
* [2. MVSA Module: Multi-View Segments Aggregator](#2-mvsa-module-multi-view-segments-aggregator)
* [3. PIGO Module: Planar-Instance Gaussian Optimizer](#3-planar-instance-gaussian-optimizer)
* [4. Ground Truth Preparation](#4-ground-truth-preparation)
* [5. Evaluation Metrics](#5-evaluation-metrics)

---

## (0) Data Acquisition and Configuration

### Proxy (Optional) and Environment Setup

Configure the network proxy for dataset downloading and model execution, then activate the environment.

```bash
export http_proxy="http://127.0.0.1:7898" && export https_proxy="http://127.0.0.1:7898" # change to your port
conda activate pigs325
```

### Path and Variable Definitions

Please change the following variables to your local directory.

```bash
scene="scene0575_00"                               # Scene ID
cudaid="4"                                         # GPU ID
Code_path="/code3/wjh/2025-PIGS/PIGS"              # Path to PIGS
Data_path="/data/wjh/PIGS_data"                    # Path to save data
Data_type="ScanNetV2"                              # Data type
Weights_path="/code3/wjh/2025-PIGS/PIGS/weights"   # Path to weights
Ray_tmp=/code3/wjh/tmp                             # dir for ray
cd ${Code_path}
```

### Dataset Processing
Please apply for access to and download the ScanNet dataset by following the official instructions provided in the [ScanNet repository](https://github.com/ScanNet/ScanNet).

Download ScanNet `.sens` data → unpack the data → download meshes and annotations → sample and reorganize the dataset.


```bash
cd Data_preprocess/${Data_type}
mkdir -p ${Data_path}/${Data_type}/scans_sens
wget -P ${Data_path}/${Data_type}/scans_sens http://kaldir.vc.in.tum.de/scannet/v1/scans/${scene}/${scene}.sens

python reader_test.py \
    --scans_folder ${Data_path}/${Data_type}/scans_sens/ \
    --single_debug_scan_id "${scene}" \
    --output_path ${Data_path}/${Data_type}/scans_test/scans/scans/ \
    --export_depth_images \
    --export_color_images \
    --export_poses \
    --export_intrinsics

base_url=http://kaldir.vc.in.tum.de/scannet/v2/scans/${scene}
wget -P ${Data_path}/${Data_type}/scans_test/scans/scans/${scene} "$base_url"/{"${scene}.txt","${scene}_vh_clean_2.ply","${scene}_vh_clean_2.labels.ply","${scene}_vh_clean_2.0.010000.segs.json","${scene}.aggregation.json"}

python copy_hive.py --scene_name ${scene} --path_ori ${Data_path}/${Data_type}/scans_test/scans/scans --path_dest ${Data_path}/${Data_type}/scans_hive/scans
```

---

## (1) GHPS Module: Geometry-Hinted Planar Segmentor

**Extract GHPS planar segments from monocular priors estimated by [Metric3D](https://github.com/YvanYin/Metric3D) and [SAM]().**

```bash
Data_folder="${Data_path}/${Data_type}/scans_hive/scans/${scene}_step"
Seg_folder="${Data_path}/${Data_type}/planeseg/result_seg/hive_2d/${scene}_step"
cd ${Code_path}/GHPS_module/scripts

# 1. Metric3D estimate monocular normal and depth priors
CUDA_VISIBLE_DEVICES=${cudaid} python 1_mono_estimator.py \
    --data_folder ${Data_folder} \
    --seg_folder ${Seg_folder} \
    --depth --crop 6 --sdf_trunc 0.1 --voxel_size 0.025 --model m3d

# 2. GHPS: Normal Guided Planar Segmentor
CUDA_VISIBLE_DEVICES=${cudaid} python 2_normal_cluster.py \
    --data_folder ${Data_folder} \
    --seg_folder ${Seg_folder}

# 3. GHPS: Distance Guided Planar Refinement
CUDA_VISIBLE_DEVICES=${cudaid} python 3_distance_refine.py \
    --data_folder ${Data_folder} \
    --seg_folder ${Seg_folder} \
    --weight_pth ${Weights_path}/sam_vit_h_4b8939.pth \
    --open_flag --coarse_flag --post_flag --dist_var 10

# GHPS: Sparse Planar Label Fusion
# 4. X-PDNet Sparse Planar Label Extraction
CUDA_VISIBLE_DEVICES=${cudaid} python 4_xpd_net.py \
    --config=XPDNet_101_config \
    --trained_model=${Weights_path}/XPDNet_101_9_125000.pth \
    --images=${Data_folder}/color/:${Seg_folder}/mask_xpd \
    --crop 6 --small

# 5. Distance Refined Mask + X-PDNet Mask Fusion
CUDA_VISIBLE_DEVICES=${cudaid} python 5_sparse_fusion.py \
    --data_folder ${Data_folder} \
    --seg_folder ${Seg_folder} \
    --crop 6

# 6. GHPS: Planar Structural Outputs
CUDA_VISIBLE_DEVICES=${cudaid} python 6_smooth_parallel.py \
    --data_folder ${Data_folder} \
    --seg_folder ${Seg_folder} \
    --mask_type fusion

# 7. GHPS: Meshing-Rendering-Projection
python 7_render_proj.py \
    --data_folder ${Data_folder} \
    --depth_model m3d
```

---

## (2) MVSA Module: Multi-View Segments Aggregator

**Perform multi-view mask planar instance association and mvsa point cloud aggregation.**

```bash
Data_folder="${Data_path}/${Data_type}/scans_hive/scans/${scene}_step"
Seg_folder="${Data_path}/${Data_type}/planeseg/result_seg"
cd ${Code_path}/MVSA_module/scripts

# 8. MVSA: Planar Normal guided Mask Clustering stage I
CUDA_VISIBLE_DEVICES=${cudaid} python 8_MC_stage1.py \
    --config scannet --debug \
    --data_folder ${Data_folder} \
    --seg_folder ${Seg_folder}/hive_pigs/${scene}_step \
    --model m3d --mask fusion --render

# 9. MVSA: Planar Normal guided Mask Clustering stage II
CUDA_VISIBLE_DEVICES=${cudaid} python 9_MC_stage2.py \
    --data_folder ${Data_folder} \
    --seg_folder ${Seg_folder}/hive_pigs/${scene}_step \
    --model m3d --mask fusion --render

# 10. MVSA: Planar Distance Normalized Ransac for MVSA point
python 10_ransac_3d.py \
    --clustering-path ${Seg_folder}/hive_pigs/${scene}_step/mvsa_output/object_pcd \
    --points3d-path ${Data_folder}/points3d.ply \
    --normal-inlier-threshold 0.85 \
    --distance-inlier-threshold 0.1
```

---

## (3) Planar-Instance Gaussian Optimizer

**Planar-Instance Gaussian Optimization and final mesh extraction.**

```bash
Data_folder="${Data_path}/${Data_type}/scans_hive/scans/${scene}_step"
Seg_folder="${Data_path}/${Data_type}/planeseg/result_seg"
PIGO_folder="${Code_path}/PIGO_module/output_${Data_type}/${scene}"
PIGS_folder="${Data_path}/${Data_type}/scans_pigs"
cd ${Code_path}/PIGO_module

# 11. PIGO: Planar-Instance Gaussian Optimizer: Training
# Ablation settings:
# 1. --no_initial     PIGS w/o Geometric Initialization
# 2. --use_temp       PIGS with instance-level splatting
# 3. --no_normal_p    PIGS w/o Planar Normal Loss
# 4. --no_distance    PIGS w/o Planar Distance Loss
# 5. --increment      PIGS with incremental distance loss
CUDA_VISIBLE_DEVICES=${cudaid} python 11_train_scannet_ablation.py \
    -s ${Data_folder} \
    -m ${PIGO_folder} \
    --seg_path ${Seg_folder}/hive_pigs/${scene}_step/mvsa_output \
    --iterations 10000 \
    --mask_type fusion \
    --use_temp --increment

# 12. PIGO: Planar-Instance Gaussian Optimizer: Rendering
CUDA_VISIBLE_DEVICES=${cudaid} python 12_render.py \
    -m ${PIGO_folder} \
    --max_depth 10.0 \
    --voxel_size 0.01 \
    --mask_type fusion

# 13. Extracting PIGO optimized planar instance points
CUDA_VISIBLE_DEVICES=${cudaid} python 13_get_inst_ply.py \
    --data_folder ${Data_folder} \
    --seg_folder ${Seg_folder}/hive_pigs/${scene}_step/mvsa_output \
    --model_path ${PIGO_folder} \
    --mask_type fusion \
    --use_render --filter

# 14. Planar Distance Normalized Ransac for PIGO point
cd ${Code_path}/MVSA_module/scripts
python 10_ransac_3d.py \
    --clustering-path ${PIGO_folder}/mesh/filter \
    --points3d-path ${PIGO_folder}/mesh/points3d_pigo.ply \
    --down-sample \
    --distance-inlier-threshold 0.075

# 15. Planar-Aware Ball-Pivoting Meshing: Extracting Mesh
python 15_ball_pivoting.py \
    --input_pcd_path ${PIGO_folder}/mesh/points3d_pigo.ply \
    --output_mesh_path ${PIGS_folder}/${scene}/${scene}_planar_mesh_pigs.ply

# EXTRA: Replace 2DGS in PIGS by rectangular GS in PlanarSplatting
cd ${Code_path}/PIGO_module
CUDA_VISIBLE_DEVICES=${cudaid} python 11_train_scannet_planarsplat.py \
    --data_folder ${Data_folder} \
    --prior_folder ${Seg_folder}/hive_2d/${scene}_step \
    --mvsa_folder ${Seg_folder}/hive_pigs/${scene}_step/mvsa_output \
    --base_conf planarsplat/confs/base_conf_planarSplatCuda.conf \
    --scene_conf planarsplat/confs/scannetv2_train.conf
```

---

## (4) Ground Truth Preparation

**Generate ground truth meshes, renders, and visibility volumes for evaluation.**

```bash
cd ${Code_path}/MVSA_module/airplanes_part/benchmark
export PYTHONPATH=${Code_path}/MVSA_module:$PYTHONPATH

# 0. Generate Ground Truth Meshes
python 0_generate_ground_truth.py \
    --scannet ${Data_path}/${Data_type}/scans_test/scans/scans \
    --output ${Data_path}/${Data_type}/gt_plane_meshes \
    --tsv_path scannetv2-labels.combined.tsv \
    --scene_name ${scene}

# 1. Rendering Ground Truth
python 1_rendering.py \
    --data-dir ${Data_path}/${Data_type}/scans_test/scans/scans \
    --planes-dir ${Data_path}/${Data_type}/gt_plane_meshes \
    --output-dir ${Data_path}/${Data_type}/gt_plane_renders/${scene} \
    --height 192 \
    --width 256 \
    --render-depth

# 2. Generate Visibility Volume
python 2_generate_visibility_volumes.py \
    --scan_data_root ${Data_path}/${Data_type}/scans_test/scans/scans \
    --rendered_depths_root ${Data_path}/${Data_type}/gt_plane_renders \
    --output_dir ${Data_path}/${Data_type}/gt_visibility_volumes \
    --scene_name ${scene}
```

---

## (5) Evaluation Metrics

**Compute Metrics on Segmentation and Geometry (Overall, topk, and medium & small).**

```bash
# 1. Compute Segmentation Metrics
python segmentation.py \
    --pred-root ${PIGS_folder} \
    --gt-root ${Data_path}/${Data_type}/gt_plane_meshes \
    --output-score-dir ${PIGS_folder}/${scene}/scores \
    --scene-name ${scene} \
    --mesh-name pigs

# 2. Compute Mesh Metrics (Overall)
python meshing.py \
    --pred-root ${PIGS_folder} \
    --gt-root ${Data_path}/${Data_type}/gt_plane_meshes \
    --output-score-dir ${PIGS_folder}/${scene}/scores \
    --ray-tmp-dir ${Ray_tmp} \
    --scene-name ${scene} \
    --mesh-name pigs

# 3. Compute Planar Metrics (Top k)
python meshing.py \
    --pred-root ${PIGS_folder} \
    --gt-root ${Data_path}/${Data_type}/gt_plane_meshes \
    --output-score-dir ${PIGS_folder}/${scene}/scores \
    --ray-tmp-dir ${Ray_tmp} \
    --scene-name ${scene} \
    --mesh-name pigs \
    --use-planar-metrics \
    --k 20

# 4. Compute Planar Metrics (Medium & Small Scale)
python meshing.py \
    --pred-root ${PIGS_folder} \
    --gt-root ${Data_path}/${Data_type}/gt_plane_meshes \
    --output-score-dir ${PIGS_folder}/${scene}/scores \
    --ray-tmp-dir ${Ray_tmp} \
    --scene-name ${scene} \
    --mesh-name pigs \
    --scale-aware-metrics \
    --k 20
```
