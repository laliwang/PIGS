#!/usr/bin/env bash
# ==============================================================================
# PIGO + Meshing Processing Script (3)
# ==============================================================================

set -e
# Catch errors and print the failed step
trap 'echo -e "\n\033[1;31m❌ ERROR at step: ${STEP_NAME}\033[0m\n"; exit 1' ERR
export PYTHONUNBUFFERED=1  # Force Python to flush output stream in real-time (tqdm)

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

echo "========== (3) PIGO + Meshing =========="
conda activate pigs

# --- Path Definitions ---
Data_folder="${Data_path}/${Data_type}/scans_hive/scans/${scene}_step"
Seg_folder="${Data_path}/${Data_type}/planeseg/result_seg"
PIGO_folder="${Code_path}/PIGO_module/output_${Data_type}/${scene}"
PIGS_folder="${Data_path}/${Data_type}/scans_pigs"
Log_folder="${Code_path}/logs/${Data_type}/${scene}_step"
# Use a unified summary log
SUMMARY_LOG="${Log_folder}/$(basename "$0" .sh)_summary.log"

mkdir -p "${Log_folder}"

# Initialize summary log header
{
    echo "============================================================"
    echo "  PIGO EXECUTION SUMMARY - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  Scene: ${scene} | GPU_ID: ${cudaid}"
    echo "============================================================"
} > "${SUMMARY_LOG}"

# ============================================================
# Monitor Function: Supports progress bar, CPU/GPU peak stats
# ============================================================
run_step () {
    STEP_NAME="$1"
    shift
    CMD="$@"

    echo -e "\n\033[1;34m▶▶▶ [START] ${STEP_NAME}\033[0m"

    STEP_START=$(date +%s)
    GPU_TMP_LOG=$(mktemp)
    TIME_OUT_LOG=$(mktemp)

    # 1. Background GPU VRAM monitoring (0.2s sampling rate)
    (
        while true; do
            nvidia-smi --id="${cudaid}" --query-gpu=memory.used --format=csv,noheader,nounits >> "${GPU_TMP_LOG}" 2>/dev/null || true
            sleep 0.2
        done
    ) &
    GPU_MON_PID=$!

    # 2. Execute command
    # Using --output to avoid redirecting stderr so progress bars show on screen
    /usr/bin/time -v --output="${TIME_OUT_LOG}" bash -c "${CMD}"

    STEP_END=$(date +%s)
    STEP_TIME=$((STEP_END - STEP_START))

    # 3. Stop GPU monitoring
    kill ${GPU_MON_PID} >/dev/null 2>&1 || true
    wait ${GPU_MON_PID} 2>/dev/null || true

    # 4. Data parsing and conversion (KB -> MB)
    GPU_PEAK=$(awk 'BEGIN{m=0}{if($1>m)m=$1}END{print m}' "${GPU_TMP_LOG}")
    [ -z "${GPU_PEAK}" ] && GPU_PEAK=0
    
    CPU_PEAK_KB=$(grep "Maximum resident set size" "${TIME_OUT_LOG}" | awk -F': ' '{print $2}')
    CPU_PEAK_MB=$(echo "scale=2; ${CPU_PEAK_KB} / 1024" | bc)

    # 5. Terminal output and write to summary log
    echo -e "\033[1;32m✔ DONE: ${STEP_NAME}\033[0m"
    printf "   %-15s : %d s\n" "Duration" "${STEP_TIME}"
    printf "   %-15s : %.2f MB\n" "Max CPU RAM" "${CPU_PEAK_MB}"
    printf "   %-15s : %d MB\n" "Max GPU VRAM" "${GPU_PEAK}"

    {
        printf "Step: %-20s | Time: %4ds | CPU: %8.2fMB | GPU: %5dMB\n" \
                "${STEP_NAME}" "${STEP_TIME}" "${CPU_PEAK_MB}" "${GPU_PEAK}"
    } >> "${SUMMARY_LOG}"

    # Cleanup temporary files
    rm -f "${GPU_TMP_LOG}" "${TIME_OUT_LOG}"
}

# ============================================================
# Execution Flow
# ============================================================

# Note: PIGO module requires switching to its directory for execution
cd "${Code_path}/PIGO_module"

run_step "11. PIGO training" \
"CUDA_VISIBLE_DEVICES=${cudaid} python 11_train_scannet_ablation.py \
    -s ${Data_folder} \
    -m ${PIGO_folder} \
    --seg_path ${Seg_folder}/hive_pigs/${scene}_step/mvsa_output \
    --iterations 10000 \
    --mask_type fusion \
    --use_temp --increment"

run_step "12. PIGO rendering" \
"CUDA_VISIBLE_DEVICES=${cudaid} python 12_render.py \
    -m ${PIGO_folder} \
    --max_depth 10.0 \
    --voxel_size 0.01 \
    --mask_type fusion"

run_step "13. Extract instance" \
"CUDA_VISIBLE_DEVICES=${cudaid} python 13_get_inst_ply.py \
    --data_folder ${Data_folder} \
    --seg_folder ${Seg_folder}/hive_pigs/${scene}_step/mvsa_output \
    --model_path ${PIGO_folder} \
    --mask_type fusion \
    --use_render --filter"

# Steps 14 & 15 require switching to MVSA_module for execution
run_step "14. PIGO RANSAC" \
"cd ${Code_path}/MVSA_module/scripts && \
python 10_ransac_3d.py \
    --clustering-path ${PIGO_folder}/mesh/filter \
    --points3d-path ${PIGO_folder}/mesh/points3d_pigo.ply \
    --down-sample \
    --distance-inlier-threshold 0.075"

run_step "15. Ball Pivoting" \
"cd ${Code_path}/MVSA_module/scripts && \
python 15_ball_pivoting.py \
    --input_pcd_path ${PIGO_folder}/mesh/points3d_pigo.ply \
    --output_mesh_path ${PIGS_folder}/${scene}/${scene}_planar_mesh_pigs.ply"

# --- Final Summary Printing ---
echo -e "\n\033[1;33m============================================================\033[0m"
echo -e "\033[1;33m✅ (3) PIGO & Meshing finished. SUMMARY:\033[0m"
cat "${SUMMARY_LOG}"
echo -e "\033[1;33m============================================================\033[0m"
echo "Full summary log saved to: ${SUMMARY_LOG}"