#!/usr/bin/env bash
# ==============================================================================
# (4) Ground Truth Generation for Evaluation
# ==============================================================================

set -e
# Trap errors and print the failed step
trap 'echo -e "\n\033[1;31m❌ ERROR at step: ${STEP_NAME}\033[0m\n"; exit 1' ERR
export PYTHONUNBUFFERED=1

# --- Read JSON Configuration ---
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

echo "========== (4) Generate Evaluation Ground Truth =========="
conda activate pigs

# --- Paths and Environment Variables ---
cd "${Code_path}/MVSA_module/airplanes_part/benchmark"
export PYTHONPATH="${Code_path}/MVSA_module:${PYTHONPATH}"

# ============================================================
# Step Execution Function (Simplified)
# ============================================================
run_step () {
    STEP_NAME="$1"
    shift
    CMD="$@"

    echo -e "\n\033[1;34m▶▶▶ [START] ${STEP_NAME}\033[0m"
    bash -c "${CMD}"
    echo -e "\033[1;32m✔ DONE: ${STEP_NAME}\033[0m"
}

# ============================================================
# Execution Flow
# ============================================================

run_step "0. Generate GT plane meshes" \
"python 0_generate_ground_truth.py \
    --scannet ${Data_path}/${Data_type}/scans_test/scans/scans \
    --output ${Data_path}/${Data_type}/gt_plane_meshes \
    --tsv_path scannetv2-labels.combined.tsv \
    --scene_name ${scene}"

run_step "1. Rendering GT planes" \
"python 1_rendering.py \
    --data-dir ${Data_path}/${Data_type}/scans_test/scans/scans \
    --planes-dir ${Data_path}/${Data_type}/gt_plane_meshes \
    --output-dir ${Data_path}/${Data_type}/gt_plane_renders/${scene} \
    --height 192 \
    --width 256 \
    --render-depth"

run_step "2. Generate visibility volumes" \
"python 2_generate_visibility_volumes.py \
    --scan_data_root ${Data_path}/${Data_type}/scans_test/scans/scans \
    --rendered_depths_root ${Data_path}/${Data_type}/gt_plane_renders \
    --output_dir ${Data_path}/${Data_type}/gt_visibility_volumes \
    --scene_name ${scene}"

# ============================================================
# Completion Message
# ============================================================

echo -e "\n\033[1;33m============================================================\033[0m"
echo -e "\033[1;33m✅ (4) Ground Truth generation finished successfully.\033[0m"
echo -e "\033[1;33m============================================================\033[0m"