#!/usr/bin/env bash
# ==============================================================================
# (5) Benchmark Evaluation
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
Ray_tmp=$(jq -r '.ray_tmp' "${CONFIG_JSON}")
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

echo "========== (5) Benchmark Evaluation =========="
conda activate pigs

# --- Paths and PYTHONPATH ---
PIGS_folder="${Data_path}/${Data_type}/scans_pigs"
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

run_step "1. Segmentation evaluation" \
"python segmentation.py \
    --pred-root ${PIGS_folder} \
    --gt-root ${Data_path}/${Data_type}/gt_plane_meshes \
    --output-score-dir ${PIGS_folder}/${scene}/scores \
    --scene-name ${scene} \
    --mesh-name pigs"

run_step "2. Meshing evaluation (Planar Geometry)" \
"python meshing.py \
    --pred-root ${PIGS_folder} \
    --gt-root ${Data_path}/${Data_type}/gt_plane_meshes \
    --output-score-dir ${PIGS_folder}/${scene}/scores \
    --ray-tmp-dir ${Ray_tmp} \
    --scene-name ${scene} \
    --mesh-name pigs \
    --use-planar-metrics \
    --k 20"

run_step "3. Meshing evaluation (Overall Geometry)" \
"python meshing.py \
    --pred-root ${PIGS_folder} \
    --gt-root ${Data_path}/${Data_type}/gt_plane_meshes \
    --output-score-dir ${PIGS_folder}/${scene}/scores \
    --ray-tmp-dir ${Ray_tmp} \
    --scene-name ${scene} \
    --mesh-name pigs"

run_step "4. Meshing evaluation (Scale Planar Geometry)" \
"python meshing.py \
    --pred-root ${PIGS_folder} \
    --gt-root ${Data_path}/${Data_type}/gt_plane_meshes \
    --output-score-dir ${PIGS_folder}/${scene}/scores \
    --ray-tmp-dir ${Ray_tmp} \
    --scene-name ${scene} \
    --mesh-name pigs \
    --scale-aware-metrics \
    --k 20"

# ============================================================
# Completion Message
# ============================================================

echo -e "\n\033[1;33m============================================================\033[0m"
echo -e "\033[1;33m✅ (5) Benchmark finished successfully.\033[0m"
echo -e "\033[1;33m============================================================\033[0m"