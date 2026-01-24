#!/usr/bin/env bash
set -e
STEP_NAME="INIT"
trap 'echo -e "\n❌ ERROR at step: ${STEP_NAME}\n"; exit 1' ERR

# --- read settings from JSON file ---
CONFIG_JSON="$1"

if [[ ! -f "${CONFIG_JSON}" ]]; then
    echo "❌ Config file not found: ${CONFIG_JSON}"
    exit 1
fi

scene=$(jq -r '.scene' "${CONFIG_JSON}")
cudaid=$(jq -r '.cudaid' "${CONFIG_JSON}")
Code_path=$(jq -r '.Code_path' "${CONFIG_JSON}")
Data_path=$(jq -r '.Data_path' "${CONFIG_JSON}")
Data_type=$(jq -r '.Data_type' "${CONFIG_JSON}")
use_proxy=$(jq -r '.use_proxy' "${CONFIG_JSON}")
source $(jq -r '.conda_sh' "${CONFIG_JSON}")

# -------- Proxy (optional) --------
if [[ "${use_proxy}" == "true" ]]; then
    echo "🌐 Using proxy"
    export http_proxy="http://127.0.0.1:7898"
    export https_proxy="http://127.0.0.1:7898"
else
    echo "🚫 Proxy disabled"
fi

echo "========== (0) Prepare ScanNet Dataset =========="
conda activate pigs

cd ${Code_path}

# -------- Step 0.1 --------
STEP_NAME="Download .sens file"
echo "[0.1] Downloading sens file..."
mkdir -p ${Data_path}/ScanNetV2/scans_sens
wget -P ${Data_path}/ScanNetV2/scans_sens \
    http://kaldir.vc.in.tum.de/scannet/v1/scans/${scene}/${scene}.sens

# -------- Step 0.2 --------
STEP_NAME="Unpack sens"
echo "[0.2] Unpacking sens..."
cd Data_preprocess/ScanNetV2
python reader_test.py \
    --scans_folder ${Data_path}/ScanNetV2/scans_sens/ \
    --single_debug_scan_id "${scene}" \
    --output_path ${Data_path}/ScanNetV2/scans_test/scans/scans/ \
    --export_depth_images \
    --export_color_images \
    --export_poses \
    --export_intrinsics

# -------- Step 0.3 --------
STEP_NAME="Download mesh & annotations"
echo "[0.3] Downloading mesh and annotations..."
base_url=http://kaldir.vc.in.tum.de/scannet/v2/scans/${scene}
wget -P ${Data_path}/ScanNetV2/scans_test/scans/scans/${scene} \
    "$base_url"/{"${scene}.txt","${scene}_vh_clean_2.ply","${scene}_vh_clean_2.labels.ply","${scene}_vh_clean_2.0.010000.segs.json","${scene}.aggregation.json"}

# -------- Step 0.4 --------
STEP_NAME="Copy to hive format"
echo "[0.4] Copying to hive format..."
python copy_hive.py \
    --scene_name ${scene} \
    --path_ori ${Data_path}/ScanNetV2/scans_test/scans/scans \
    --path_dest ${Data_path}/ScanNetV2/scans_hive/scans

echo "✅ (0) Dataset preparation finished"
