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

# 若存在路径不匹配的旧 CMake 缓存，则自动清理
if [[ -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
    CACHED_SRC=$(grep "maskrcnn_swins_inference_SOURCE_DIR:" "${BUILD_DIR}/CMakeCache.txt" 2>/dev/null | cut -d'=' -f2 || true)
    if [[ -n "${CACHED_SRC}" && "${CACHED_SRC}" != "${REPO_ROOT}" ]]; then
        echo ">>> 检测到旧缓存路径 (${CACHED_SRC}) 与当前仓库路径 (${REPO_ROOT}) 不匹配，自动清理旧构建缓存..."
        rm -rf "${BUILD_DIR}/CMakeCache.txt" "${BUILD_DIR}/CMakeFiles"
    fi
fi

# 检查 3rdparty 依赖
if [[ ! -d "${REPO_ROOT}/3rdparty/opencv" ]]; then
    echo "警告: 未检测到 3rdparty/opencv，正在自动调用依赖编译脚本..."
    bash "${SCRIPT_DIR}/build_3rdparty_ubuntu1804.sh"
fi

cmake -B "${BUILD_DIR}" -S "${REPO_ROOT}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" -j"$(nproc || echo 4)"

echo ">>> 编译完成！可执行程序: ${BUILD_DIR}/test_model_inference"

# 若使用 GPU，支持通过 3rdparty/cuda11-runtime 或 CUDA_HOME / CUDA_PATH 环境变量 / 标准系统路径加载 CUDA 运行库
if [[ "${DEVICE}" == "gpu" ]]; then
    if [[ -d "${REPO_ROOT}/3rdparty/cuda11-runtime/lib" ]]; then
        export LD_LIBRARY_PATH="${REPO_ROOT}/3rdparty/cuda11-runtime/lib:${LD_LIBRARY_PATH:-}"
    elif [[ -d "${REPO_ROOT}/3rdparty/cuda-runtime/lib" ]]; then
        export LD_LIBRARY_PATH="${REPO_ROOT}/3rdparty/cuda-runtime/lib:${LD_LIBRARY_PATH:-}"
    fi
    if [[ -n "${CUDA_HOME:-}" && -d "${CUDA_HOME}/lib64" ]]; then
        export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
    elif [[ -n "${CUDA_PATH:-}" && -d "${CUDA_PATH}/lib64" ]]; then
        export LD_LIBRARY_PATH="${CUDA_PATH}/lib64:${LD_LIBRARY_PATH:-}"
    elif [[ -d "/usr/local/cuda/lib64" ]]; then
        export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
    fi
fi

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
