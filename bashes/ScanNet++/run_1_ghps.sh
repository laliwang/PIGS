#!/usr/bin/env bash
# ==============================================================================
# GHPS Full Pipeline Processing Script (1)
# ==============================================================================

set -e
# Trap errors and print the failed step
trap 'echo -e "\n\033[1;31m❌ ERROR at step: ${STEP_NAME}\033[0m\n"; exit 1' ERR
export PYTHONUNBUFFERED=1  # Force Python to flush output stream in real-time

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
sam_path=$(jq -r '.sam_path' "${CONFIG_JSON}")
source $(jq -r '.conda_sh' "${CONFIG_JSON}")

# -------- Proxy (optional) --------
if [[ "${use_proxy}" == "true" ]]; then
    echo "🌐 Using proxy"
    export http_proxy="http://127.0.0.1:7898"
    export https_proxy="http://127.0.0.1:7898"
else
    echo "🚫 Proxy disabled"
fi

echo "========== (1) GHPS Module =========="
conda activate pigs

# --- Path Definitions ---
Data_folder="${Data_path}/${Data_type}/scans_hive/scans/${scene}_step"
Seg_folder="${Data_path}/${Data_type}/planeseg/result_seg/hive_2d/${scene}_step"
Log_folder="${Code_path}/logs/${Data_type}/${scene}_step"
SUMMARY_LOG="${Log_folder}/$(basename $0 .sh)_summary.log"

mkdir -p "${Log_folder}"
cd "${Code_path}/GHPS_module/scripts"

# Initialize summary log header
{
    echo "============================================================"
    echo "  GHPS EXECUTION SUMMARY - $(date '+%Y-%m-%d %H:%M:%S')"
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
            nvidia-smi --id=${cudaid} --query-gpu=memory.used --format=csv,noheader,nounits >> "${GPU_TMP_LOG}" 2>/dev/null || true
            sleep 0.2
        done
    ) &
    GPU_MON_PID=$!

    # 2. Execute command
    # Use --output to ensure stats don't block stderr, allowing tqdm progress bars to show correctly
    /usr/bin/time -v --output="${TIME_OUT_LOG}" bash -c "${CMD}"

    STEP_END=$(date +%s)
    STEP_TIME=$((STEP_END - STEP_START))

    # 3. Stop GPU monitoring
    kill ${GPU_MON_PID} >/dev/null 2>&1 || true
    wait ${GPU_MON_PID} 2>/dev/null || true

    # 4. Data parsing and conversion
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
# Execution Flow: run_step "Step Name" "Specific Command"
# ============================================================

run_step "1. Metric3D" \
"CUDA_VISIBLE_DEVICES=${cudaid} python 1_mono_estimator.py \
    --data_folder ${Data_folder} \
    --seg_folder ${Seg_folder} \
    --depth --crop 6 --sdf_trunc 0.1 --voxel_size 0.025 --model m3d"

run_step "2. Normal clustering" \
"CUDA_VISIBLE_DEVICES=${cudaid} python 2_normal_cluster.py \
    --data_folder ${Data_folder} \
    --seg_folder ${Seg_folder}"

run_step "3. SAM Refine" \
"CUDA_VISIBLE_DEVICES=${cudaid} python 3_distance_refine.py \
    --data_folder ${Data_folder} \
    --seg_folder ${Seg_folder} \
    --weight_pth ${sam_path} \
    --open_flag --coarse_flag --post_flag --dist_var 10"

run_step "4. X-PDNet" \
"CUDA_VISIBLE_DEVICES=${cudaid} python 4_xpd_net.py \
    --config=XPDNet_101_config \
    --trained_model=${Code_path}/GHPS_module/X-PDNet/weights/XPDNet_101_9_125000.pth \
    --images=${Data_folder}/color/:${Seg_folder}/mask_xpd \
    --crop 6 --small"

run_step "5. Sparse fusion" \
"CUDA_VISIBLE_DEVICES=${cudaid} python 5_sparse_fusion.py \
    --data_folder ${Data_folder} \
    --seg_folder ${Seg_folder} \
    --crop 6"

run_step "6. Smooth parallel" \
"CUDA_VISIBLE_DEVICES=${cudaid} python 6_smooth_parallel.py \
    --data_folder ${Data_folder} \
    --seg_folder ${Seg_folder} \
    --mask_type fusion"

run_step "7. Render & project" \
"python 7_render_proj.py \
    --data_folder ${Data_folder} \
    --depth_model m3d"

# --- Final Summary Printing ---
echo -e "\n\033[1;33m============================================================\033[0m"
echo -e "\033[1;33m✅ ALL STEPS FINISHED. SUMMARY:\033[0m"
cat "${SUMMARY_LOG}"
echo -e "\033[1;33m============================================================\033[0m"
echo "Full log saved to: ${SUMMARY_LOG}"