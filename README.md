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
git clone https://github.com/BIT-DYN/omnimap.git
cd omnimap
```

### Install the required libraries

```bash
conda env create -f environment.yaml
conda activate omnimap
```

### Install torch-scatter

```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.2+cu118.html
```

### Set CUDA environment

Run this every time before using the environment, or add to conda activation script:

```bash
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
```

To make it permanent, add to conda activate script:
```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH' > $CONDA_PREFIX/etc/conda/activate.d/cuda_env.sh
```

### Install thirdparty components

```bash
pip install --no-build-isolation thirdparty/simple-knn
pip install --no-build-isolation thirdparty/diff-gaussian-rasterization
pip install --no-build-isolation thirdparty/lietorch
```

**Note:** The `mmyolo` package has been copied from YOLO-World repository into `thirdparty/mmyolo/` to resolve a dependency conflict. The original YOLO-World had a version constraint that prevented using mmcv versions newer than 2.0.0, but this project requires mmcv 2.1.0. This issue has been fixed in the local copy.

### Install YOLO-World Model

```bash
cd ..
git clone --recursive https://github.com/AILab-CVC/YOLO-World.git
cd YOLO-World
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
pip install -r <(grep -v "opencv-python" requirements/basic_requirements.txt)
pip install -e . --no-build-isolation
cd ../omnimap
```

**Fix YOLO-World syntax error:** In `YOLO-World/yolo_world/models/detectors/yolo_world.py` line 61, replace:
```python
self.text_feats, None = self.backbone.forward_text(texts)
```
with:
```python
self.text_feats, _ = self.backbone.forward_text(texts)
```

Download pretrained weights [YOLO-Worldv2-L (CLIP-Large)](https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_v2_l_clip_large_o365v1_goldg_pretrain_800ft-9df82e55.pth) to `weights/yolo-world/`.

### Install TAP Model

```bash
pip install flash-attn==2.5.8 --no-build-isolation
pip install git+https://github.com/baaivision/tokenize-anything.git
```

Download pretrained weights to `weights/tokenize-anything/`:
- [tap_vit_l_v1_1.pkl](https://huggingface.co/BAAI/tokenize-anything/resolve/main/models/tap_vit_l_v1_1.pkl)
- [merged_2560.pkl](https://huggingface.co/BAAI/tokenize-anything/resolve/main/models/merged_2560.pkl)

### Install SBERT Model

```bash
pip install -U sentence-transformers
pip install transformers==4.36.2
```

**Note:** If you see `sentence-transformers 5.2.0 has requirement transformers<6.0.0,>=4.41.0, but you have transformers 4.36.2.` just skip it - it's okay.

Download pretrained weights to `weights/sbert/`:
```bash
cd weights/sbert
git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
```

### Install additional dependencies

```bash
pip install --no-build-isolation git+https://github.com/lvis-dataset/lvis-api.git
python -m spacy download en_core_web_sm
```

### Download YOLO-World data files

(This part is unnecessary because data folder already exists with all required scripts)

```bash
mkdir -p data/coco/lvis && cd data/coco/lvis
wget https://huggingface.co/GLIPModel/GLIP/resolve/main/lvis_v1_minival_inserted_image_name.json
cd ../../..
cp -r ../YOLO-World/data/texts data/
```

### Modify the model path

Change the address of the above models in the configuration file in `config/`.

### Reinstall mmcv:

(some packages may change your mmcv version, please reinstall mmcv and check if it's version is 2.1.0)

```bash
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
```

### Fix transformers version compatibility:

If you encounter `AttributeError: module 'torch.utils._pytree' has no attribute 'register_pytree_node'`, install the compatible version of transformers:

```bash
pip install transformers==4.36.2
```

This version is compatible with PyTorch 2.1.2. Newer versions of transformers require PyTorch 2.2+.

### Verify installation

```bash
python -c "import torch; import mmcv; import mmdet; from tokenize_anything import model_registry; print('Setup complete')"
```

**Note:**: You may get the ERROR: `AssertionError: MMCV==2.2.0 is used but incompatible. Please install mmcv>=2.0.0rc4, <2.1.0.`. If so - just go to `__init__.py` and change `mmcv_maximum_version` to `2.2.0`.

## 📊 Prepare dataset

OmniMap has completed validation on Replica (as same with [vMap](https://github.com/kxhit/vMAP)) and ScanNet. Please download the following datasets.

* [Replica Demo](https://huggingface.co/datasets/kxic/vMAP/resolve/main/demo_replica_room_0.zip) - Replica Room 0 only for faster experimentation.
* [Replica](https://huggingface.co/datasets/kxic/vMAP/resolve/main/vmap.zip) - All Pre-generated Replica sequences.
* [ScanNet](https://github.com/ScanNet/ScanNet) - Official ScanNet sequences.

Update the dataset path in `config/replica_config.yaml` or `config/scannet_config.yaml`:
```yaml
path:
  data_path: /path/to/your/dataset
```

## 🏃 Run

### Main Code

Run the following command to start the formal execution of the incremental mapping.

```bash
# for replica
python demo.py --dataset replica --scene {scene} --vis_gui
# for scannet
python demo.py --dataset scannet --scene {scene} --vis_gui
```

You can use `--start {start_id}` and `--length {length}` to specify the starting frame ID and the mapping duration, respectively. The `--vis_gui` flag controls online visualization; disabling it may improve processing speed.

### Examples:

```bash
# Replica
python demo.py --dataset replica --scene room_0

# ScanNet
python main.py --dataset scannet --scene scene0000_00
```

After building the map, the results will be saved in folder `outputs/{scene}`, which contains the rendered outputs and evaluation metrics.

### Gen 3D Mesh

We use the rendered depth and color images to generate the color mesh. You can run the following code to perform this operation.

```bash
# for replica
python tsdf_integrate.py --dataset replica --scene {scene}
# for scannet
python tsdf_integrate.py --dataset scannet --scene {scene}
```

## Project Structure

```
omnimap/
├── config/
│   ├── replica_config.yaml
│   ├── scannet_config.yaml
│   └── yolo-world/
├── data/
│   ├── coco/lvis/
│   └── texts/
├── weights/
│   ├── yolo-world/
│   ├── tokenize-anything/
│   └── sbert/
├── thirdparty/
│   ├── simple-knn/
│   ├── diff-gaussian-rasterization/
│   ├── lietorch/
│   └── mmyolo/
└── demo.py
```

## 🔗 Citation

If you find our work helpful, please cite:

```bibtex
@article{omnimap,
  title={OmniMap: A Comprehensive Mapping Framework Integrating Optics, Geometry, and Semantics},
  author={Deng, Yinan and Yue, Yufeng and Dou, Jianyu and Zhao, Jingyu and Wang, Jiahui and Tang, Yujie and Yang, Yi and Fu, Mengyin},
  journal={IEEE Transactions on Robotics},
  year={2025}
}
```

## 👏 Acknowledgements

We would like to express our gratitude to the open-source projects and their contributors [HI-SLAM2](https://github.com/Willyzw/HI-SLAM2), [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting), [YOLO-World](https://github.com/AILab-CVC/YOLO-World), and [TAP](https://github.com/baaivision/tokenize-anything). Their valuable work has greatly contributed to the development of our codebase.