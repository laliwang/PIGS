#!/usr/bin/env bash
# ==============================================================================
# MVSA Module Full Pipeline Processing Script (2)
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

echo "========== (2) MVSA Module =========="
conda activate pigs

# --- Path Definitions ---
Data_folder="${Data_path}/${Data_type}/scans_hive/scans/${scene}_step"
Seg_folder="${Data_path}/${Data_type}/planeseg/result_seg"
Log_folder="${Code_path}/logs/${Data_type}/${scene}_step"
# Using consistent summary log format
SUMMARY_LOG="${Log_folder}/$(basename "$0" .sh)_summary.log"

mkdir -p "${Log_folder}"
cd "${Code_path}/MVSA_module/scripts"

# Initialize summary log header
{
    echo "============================================================"
    echo "  MVSA EXECUTION SUMMARY - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  Scene: ${scene} | GPU_ID: ${cudaid}"
    echo "============================================================"
} > "${SUMMARY_LOG}"

# ============================================================
# Monitor Function: Aligned with Script 1, supports progress bars
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
    # Use --output to ensure stats don't redirect stderr, allowing progress bars to show on screen
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

run_step "8. MC stage I" \
"CUDA_VISIBLE_DEVICES=${cudaid} python 8_MC_stage1.py \
    --config scannet --debug \
    --data_folder ${Data_folder} \
    --seg_folder ${Seg_folder}/hive_pigs/${scene}_step \
    --model m3d --mask fusion --render"

run_step "9. MC stage II" \
"CUDA_VISIBLE_DEVICES=${cudaid} python 9_MC_stage2.py \
    --data_folder ${Data_folder} \
    --seg_folder ${Seg_folder}/hive_pigs/${scene}_step \
    --model m3d --mask fusion --render"

run_step "10. 3D RANSAC" \
"python 10_ransac_3d.py \
    --clustering-path ${Seg_folder}/hive_pigs/${scene}_step/mvsa_output/object_pcd \
    --points3d-path ${Data_folder}/points3d.ply \
    --normal-inlier-threshold 0.85 \
    --distance-inlier-threshold 0.1"

# --- Final Summary Printing ---
echo -e "\n\033[1;33m============================================================\033[0m"
echo -e "\033[1;33m✅ (2) MVSA finished. SUMMARY:\033[0m"
cat "${SUMMARY_LOG}"
echo -e "\033[1;33m============================================================\033[0m"
echo "Full summary log saved to: ${SUMMARY_LOG}"