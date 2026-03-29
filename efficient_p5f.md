# PIGS_p5f Efficient Variant – Quick Start Guide

This guide provides instructions for running the efficient variant **PIGS_p5f** introduced in our paper.

---

## 1. Install FastSAM in the PIGS Environment

```bash
Code_path="$PWD"
cd ${Code_path}

conda activate pigs325
mkdir p5f && cd p5f
git clone https://github.com/CASIA-IVA-Lab/FastSAM.git
cd FastSAM
pip install -r requirements.txt
```

---

## 2. Download FastSAM Weights

```bash
cd ${Code_path}/weights
python -m gdown https://drive.google.com/uc?id=1m1sjY4ihXBU1fZXdQ-Xdj-mDltW-2Rqv
```

---

## 3. Update Local Paths

Modify the following files to match your local FastSAM installation path:

* `x_fast_sam.py`
* `3_distance_refine_p5f.py`

---

## 4. Run ScanNetV2_p5f Pipeline

Before running, update the file `scannet_p5f.json` with your local paths and configuration.

```bash
cd ${Code_path}/bashes/ScanNetV2_p5f
bash run_1_ghps.sh scannet_p5f.json
bash run_2_mvsa.sh scannet_p5f.json
bash run_3_pigo.sh scannet_p5f.json
```
