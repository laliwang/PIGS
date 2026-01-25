<p align="center">
<h1 align="center"><strong> PIGS: Planar-Instance Gaussian Splatting
Leveraging Planar Structural Priors</strong></h1>
</p>

<p align="center">
  <a href="https://pigs325.github.io/" target='_blank'>
    <img src="https://img.shields.io/badge/Project-👔-green?">
  </a>
</p>

## 🏠 Abstract

Planar structures, ubiquitous in man-made indoor environments, enable compact and accurate scene decomposition. Recent methods distill planar features into learning-based MVS geometries for multi-view 3D plane estimation. However, the implicit nature of planar features hinders the semantic–geometry alignment within individual planar instances, resulting in distorted geometry and fragmented semantics. To address this, we propose PIGS, the first Planar-Instance Gaussian Splatting framework that explicitly models planar instances and enforces planar semantic–geometry alignment through progressively strengthened structural priors, free from feature distillation. Starting from planar-unaware geometry, our method first employs the GHPS module to achieve annotation-free single-view planar segmentation, then utilizes the MVSA module to perform multi-view instance association and planarity aggregation, and finally leverages the PIGO module to initialize and regularize Planar-Instance Gaussian primitives (PIGs) with planar-aware loss functions during instance-level splatting. Followed by a planar-aware ball-pivoting strategy for final plane mesh extraction, PIGS produces semantically and geometrically aligned 3D planes, closely matching RGB references while enforcing 3D planarity. Extensive experiments on hundreds of indoor scenes, along with comprehensive ablation studies, demonstrate the superior performance of our method.

<img src="assets/Fig1.png">

## 🛠 Install

Tested on Ubuntu 20.04/24.04 with CUDA 11.8.

### Clone this repo

```bash
git clone --recursive https://github.com/laliwang/PIGS.git
cd PIGS
Code_path="$PWD"
```

### Create a conda environment

```bash
conda create -n pigs325 python=3.9
conda activate pigs325
```

### Install torch and basic requirements

```bash
# cuda 11.8 tested
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
# basic requirements
pip install -r requirements_basic.txt
# install pytorch3d offline for stability
wget https://api.anaconda.org/download/pytorch3d/pytorch3d/0.7.4/linux-64/pytorch3d-0.7.4-py39_cu118_pyt201.tar.bz2
conda install pytorch3d-0.7.4-py39_cu118_pyt201.tar.bz2
# no build isolation
pip install https://github.com/JamieWatson683/scikit-image/archive/single_mesh.zip --no-build-isolation
# then for full requirements
pip install -r requirements_then.txt
```
**Notice:** if you encounter "network connectivity" issues during installation, you can try to install them one by one.

### Install thirdparty submodules
```bash
cd ${Code_path}/PIGO_module/submodules/diff-plane-rasterization
pip install . --no-build-isolation

cd ../simple-knn
pip install . --no-build-isolation

cd ${Code_path}/GHPS_module/renderpy
pip install .
```

### (Optional) Install thirdparty for planarsplatting
```bash
cd ${Code_path}/PIGO_module/planarsplat/submodules/diff-rect-rasterization
pip install .

cd ../quaternion-utils
pip install .
```

### Complie Segmentator for ScanNetpp
```bash
cd ${Code_path}/Data_preprocess/ScanNet++/Segmentator
cmake . && make
```

## 📦 Pretrained Weights
```bash
cd ${Code_path}
mkdir weights && cd weights

# X-PDNet pretrained weights
python -m gdown https://drive.google.com/uc?id=1ChJiTemWxG-3oTIbvOFLTo7PJsVxnZ-d
python -m gdown https://drive.google.com/uc?id=1rkPiWZ_313GGFMW1KLpyzM7a-nynchPV

# sam pretrained weights
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

## 📊 Prepare dataset

PIGS has completed validation on ScanNetV2, ScanNet++ and Replica. Please download the following datasets.

* [Replica](https://huggingface.co/datasets/kxic/vMAP/resolve/main/vmap.zip) - Pre-generated Replica sequences from [vMAP](https://github.com/kxhit/vMAP), along with our processed annotation files aligned to the ScanNetV2 format.
* [ScanNet](https://github.com/ScanNet/ScanNet) - Official ScanNet sequences.
* [ScanNet++](https://scannetpp.mlsg.cit.tum.de/scannetpp/) Official ScanNet++ sequences.

### Data & Results Structure
After running the data preprocessing scripts and the full pipeline of our code, the folder structure of the data and results is organized as follows.


<details>
<summary><strong>📁 Click to expand the directory tree</strong></summary>

```text
your_data_path  
└── ScanNetV2
    ├── gt_plane_meshes                   # ground-truth plane meshes
    │   └── scene_name
    ├── gt_plane_renders                  # ground-truth plane renders
    │   └── scene_name
    ├── gt_visibility_volumes             # ground-truth visibility volumes
    │   └── scene_name
    ├── planeseg                          # PIGS Intermediate Results
    │   └── result_seg
    │       ├── hive_2d                   # 2D structural priors
    │       │   └── scene_name_step
    │       │       ├── mask_xpd          # sparse label like xpdnet
    │       │       ├── normal_npy_m      # monocular normals
    │       │       └── planesam          # distance refined segments
    │       └── hive_pigs                 # MVSA Results
    │           └── scene_name_step
    │               ├── ghps_output       # GHPS output
    │               └── mvsa_output       # MVSA output
    ├── scans_hive                        # downsampled dataset in PIGS
    │   └── scans
    │       └── scene_name_step           # (640 x 480)
    │           ├── color
    │           ├── depth
    │           ├── depth_m3d             # monocular depths
    │           ├── intrinsic
    │           ├── mesh
    │           ├── pose
    │           └── pyrender_debug        # pyrender depths
    ├── scans_pigs                        # PIGS final output
    │   └── scene_name
    │       ├── scores                    # Evaluated Scores
    │       └── *_planar_mesh_pigs.ply    # ✦ PIGS planar mesh
    ├── scans_sens                        # original sens files
    └── scans_test                        # sensor_data format
        └── scans
            └── scans
                └── scene_name
```
</details> 

## 🏃 Run

### Bash Script

For efficiency considerations, we reorganize the PIGS pipeline into 6 bash scripts per scene, and expose a minimal and flexible interface that allows users to easily adapt the pipeline to their own directory structures.

**1.ScanNetV2: PIGS/bashes/ScanNetV2/scannet.json**
```json
{   
    "scene": "scene_name",    # "like scene0678_00"
    "cudaid": "0",
    "Code_path": "/path/to/PIGS",
    "Data_path": "/path/to/PIGS_data",
    "Data_type": "ScanNetV2",
    "conda_sh": "/path/to/anaconda3/etc/profile.d/conda.sh",
    "ray_tmp": "/path/to/tmp",
    "Weights_path": "/path/to/PIGS/weights",
    "use_proxy": true
  }
```

**2.ScanNet++: PIGS/bashes/ScanNet++/scannetpp.json**
```json
{   
    "scene": "scene_name",    # "like 785e7504b9"
    "cudaid": "0",
    "Code_path": "/path/to/PIGS",
    "Data_path": "/path/to/PIGS_data",
    "Data_type": "ScanNet++",
    "conda_sh": "/path/to/anaconda3/etc/profile.d/conda.sh",
    "ray_tmp": "/path/to/tmp",
    "Weights_path": "/path/to/PIGS/weights",
    "Token": "your_token (obtained via application)",
    "use_proxy": true
  }
```

**Notice:** the `use_proxy` flag is used if you are using a proxy for network access. If `true`, then you need to change the port in each `*.sh` file to your proxy port. And to load path parameters from json, `jq` is required on your machine.

```bash
sudo apt-get install jq
```

### Code Run

**1. ScanNetV2**
```bash
# for ScanNetV2
cd bashes/ScanNetV2

# prepare the dataset
bash run_0_prepare_scannet.sh scannet.json

# run the ghps stage
bash run_1_ghps.sh scannet.json

# run the mvsa stage
bash run_2_mvsa.sh scannet.json

# run the pigo stage
bash run_3_pigo.sh scannet.json

# generate the gt files
bash eval_compute_gt.sh scannet.json

# evaluate the metrics
bash eval_compute_metrics.sh scannet.json
```

**2. ScanNet++**
```bash
# for ScanNet++
cd bashes/ScanNet++

# prepare the dataset
bash run_0_prepare_scannetpp.sh scannetpp.json

# run the ghps stage
bash run_1_ghps.sh scannetpp.json

# run the mvsa stage
bash run_2_mvsa.sh scannetpp.json

# run the pigo stage
bash run_3_pigo.sh scannetpp.json

# generate the gt files
bash eval_compute_gt.sh scannetpp.json

# evaluate the metrics
bash eval_compute_metrics.sh scannetpp.json
```

**Notice:** For a detailed walkthrough of the full pipeline, please refer to [sequential.md](sequential.md), which provides a step-by-step example on `scene0575_00` from ScanNetV2.

## ❓ FAQ / Common Questions
### Q-1: Installing `renderpy` fails due to missing system `LibGL`?

If the system `LibGL` cannot be found when installing `renderpy`, please declare the following environment variables **before** running `pip install .`:

```bash
export CMAKE_PREFIX_PATH=/usr
export CMAKE_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
export CMAKE_INCLUDE_PATH=/usr/include
```

### Q2: Installing `diff-plane-rasterization` fails due to missing `crypt.h`?

When installing the third-party library `diff-plane-rasterization` in the PIGO module, the build may fail because `crypt.h` cannot be found.  
In this case, please run the following commands:

```bash
cp /usr/include/crypt.h /path/to/pigs_envs/include/crypt.h
export CPATH=/path/to/pigs_envs/include
pip install -e .
```

## 🔗 Citation

If you find our work helpful, please cite:

```bibtex
To be updated.
```

## 👏 Acknowledgements

We would like to express our gratitude to the open-source projects and their contributors [Airplanes](https://github.com/nianticlabs/airplanes), [Maskclustering](https://github.com/PKU-EPIC/MaskClustering), [PGSR](https://github.com/zju3dv/PGSR), [ScanNet](https://github.com/ScanNet/ScanNet) and [ScanNet++](https://github.com/scannetpp/scannetpp). Their valuable work has greatly contributed to the development of our codebase.