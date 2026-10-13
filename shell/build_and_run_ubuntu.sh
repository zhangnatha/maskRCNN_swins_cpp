#!/usr/bin/env bash
# ==============================================================================
# Script: shell/build_and_run_ubuntu.sh
# Description: Ubuntu 18.04 下一键编译本项目推理程序并运行测试验证
# ==============================================================================

set -euo pipefail

# 过滤环境变量中的 conda/anaconda/miniconda 路径，避免 Anaconda 库干扰链接
CLEAN_PATH=$(echo "${PATH}" | tr ':' '\n' | grep -v -E "anaconda|miniconda" | tr '\n' ':' | sed 's/:$//')
export PATH="${CLEAN_PATH}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BUILD_DIR="${REPO_ROOT}/build"
DEVICE="${1:-cpu}"       # 默认 cpu，可传入 gpu
DATASET="${2:-followed}" # 默认 followed，可传入 fixed

echo "================================================================================"
echo ">>> 开始编译 Mask R-CNN (Swin-S) C++ 推理程序"
echo "    工作目录: ${REPO_ROOT}"
echo "    编译目录: ${BUILD_DIR}"
echo "    推理设备: ${DEVICE}"
echo "    数据集:   ${DATASET}"
echo "================================================================================"

# 检查 3rdparty 依赖
if [[ ! -d "${REPO_ROOT}/3rdparty/opencv" ]]; then
    echo "警告: 未检测到 3rdparty/opencv，正在自动调用依赖编译脚本..."
    bash "${SCRIPT_DIR}/build_3rdparty_ubuntu1804.sh"
fi

cmake -B "${BUILD_DIR}" -S "${REPO_ROOT}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" -j"$(nproc || echo 4)"

echo ">>> 编译完成！可执行程序: ${BUILD_DIR}/test_model_inference"

# 运行推理测试
if [[ "${DATASET}" == "fixed" ]]; then
    MODEL_PATH="${REPO_ROOT}/model/model_fixed_20260929.onnx"
    NAMES_PATH="${REPO_ROOT}/model/fixed.names"
    INPUT_PATH="${REPO_ROOT}/assets/fixed"
    OUTPUT_PATH="${BUILD_DIR}/pred_fixed"
else
    MODEL_PATH="${REPO_ROOT}/model/model_followed_20260906.onnx"
    NAMES_PATH="${REPO_ROOT}/model/followed.names"
    INPUT_PATH="${REPO_ROOT}/assets/followed"
    OUTPUT_PATH="${BUILD_DIR}/pred_followed"
fi

if [[ -f "${MODEL_PATH}" && -f "${NAMES_PATH}" && -d "${INPUT_PATH}" ]]; then
    echo "================================================================================"
    echo ">>> 开始运行推理测试验证 ..."
    echo "================================================================================"
    "${BUILD_DIR}/test_model_inference" \
        "${MODEL_PATH}" \
        "${NAMES_PATH}" \
        "${INPUT_PATH}" \
        "${DEVICE}" \
        "${OUTPUT_PATH}"

    echo "================================================================================"
    echo ">>> 推理测试运行成功！输出结果保存在: ${OUTPUT_PATH}"
    echo "================================================================================"
else
    echo ">>> 提示: 模型或测试图片不存在，跳过自动推理运行测试。"
fi
