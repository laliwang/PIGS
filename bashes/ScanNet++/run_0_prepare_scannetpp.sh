#!/usr/bin/env bash
# ==============================================================================
# (0) Prepare ScanNet++ Dataset
# ==============================================================================

set -e
STEP_NAME="INIT"
# Trap errors and print the failed step
trap 'echo -e "\n❌ ERROR at step: ${STEP_NAME}\n"; exit 1' ERR

# -------- Args / Config --------
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
Token=$(jq -r '.Token' "${CONFIG_JSON}")
source $(jq -r '.conda_sh' "${CONFIG_JSON}")

# -------- Proxy (optional) --------
if [[ "${use_proxy}" == "true" ]]; then
    echo "🌐 Using proxy"
    export http_proxy="http://127.0.0.1:7898"
    export https_proxy="http://127.0.0.1:7898"
else
    echo "🚫 Proxy disabled"
fi

# -------- Activate base env --------
echo "========== (0) Prepare ScanNet++ Dataset =========="
conda activate pigs325
cd "${Code_path}"

# =====================================================
# Step 0.1 Download ScanNet++ raw data
# =====================================================
STEP_NAME="Download ScanNet++ raw data"
echo "[0.1] Downloading ScanNet++ raw data..."

mkdir -p "${Data_path}/${Data_type}/scans_data"
cd "Data_preprocess/${Data_type}/scannetpp"

python download_data.py \
    --scene "${scene}" \
    --token "${Token}" \
    --output_dir "${Data_path}/${Data_type}/scans_data"

# =====================================================
# Step 0.2 Extract RGB images from iPhone videos
# =====================================================
STEP_NAME="Extract RGB from video"
echo "[0.2] Extracting RGB images from video..."

python -m iphone.prepare_iphone_data \
    "${scene}" \
    "${Data_path}/${Data_type}/scans_data"

# =====================================================
# Step 0.3 Generate sensor_data & hive format
# =====================================================
STEP_NAME="Generate sensor_data & hive format"
echo "[0.3] Generating sensor_data and hive format..."

python gen_sensor.py \
    --scene_id "${scene}" \
    --path_data "${Data_path}/${Data_type}/scans_data" \
    --path_test "${Data_path}/${Data_type}/scans_test/scans/scans" \
    --path_hive "${Data_path}/${Data_type}/scans_hive/scans"

# =====================================================
# Step 0.4 Convert ScanNet++ annotations to ScanNetV2 format
# =====================================================
STEP_NAME="Convert annotations format"
echo "[0.4] Converting ScanNet++ annotations to ScanNetV2 format..."

cd "${Code_path}/Data_preprocess/${Data_type}/Segmentator"

./segmentator \
    "${Data_path}/${Data_type}/scans_test/scans/scans/${scene}/mesh_aligned_0.05.ply" \
    0.01 \
    20

python gen_json_pp.py \
    --scene_id "${scene}" \
    --path_test "${Data_path}/${Data_type}/scans_test/scans/scans"

echo "✅ (0) ScanNet++ dataset preparation finished"