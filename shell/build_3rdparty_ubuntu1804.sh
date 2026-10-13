#!/usr/bin/env bash
# ==============================================================================
# Script: shell/build_3rdparty_ubuntu1804.sh
# Description: 一键编译/配置 Ubuntu 18.04 依赖库到 3rdparty 文件夹 (OpenCV + ONNX Runtime)
#              构建完成后自动清理所有下载文件与中间临时编译文件。
# ==============================================================================

set -euo pipefail

# 过滤环境变量中的 conda/anaconda/miniconda 路径，避免 Anaconda 中的高版本库污染系统级构建
CLEAN_PATH=$(echo "${PATH}" | tr ':' '\n' | grep -v -E "anaconda|miniconda" | tr '\n' ':' | sed 's/:$//')
export PATH="${CLEAN_PATH}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
THIRDPARTY_DIR="${REPO_ROOT}/3rdparty"

OPENCV_VERSION="${OPENCV_VERSION:-4.5.5}"
ORT_VERSION="${ORT_VERSION:-1.20.0}"
USE_ORT_GPU="${USE_ORT_GPU:-1}" # 1: GPU(CUDA11/12), 0: CPU
FORCE_REBUILD="${FORCE_REBUILD:-0}"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu)
            USE_ORT_GPU=0
            shift
            ;;
        --gpu)
            USE_ORT_GPU=1
            shift
            ;;
        --force)
            FORCE_REBUILD=1
            shift
            ;;
        --opencv-version)
            OPENCV_VERSION="$2"
            shift 2
            ;;
        --ort-version)
            ORT_VERSION="$2"
            shift 2
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  --cpu               使用 CPU 版本 ONNX Runtime"
            echo "  --gpu               使用 GPU (CUDA) 版本 ONNX Runtime (默认)"
            echo "  --force             强制重新下载与编译，覆盖已有 3rdparty 依赖"
            echo "  --opencv-version V  指定 OpenCV 源码版本 (默认: 4.5.5)"
            echo "  --ort-version V     指定 ONNX Runtime 发行版本 (默认: 1.20.0)"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

echo "================================================================================"
echo ">>> 开始在 Ubuntu 18.04 上构建/准备 3rdparty 依赖库"
echo "    仓库目录: ${REPO_ROOT}"
echo "    目标目录: ${THIRDPARTY_DIR}"
echo "    OpenCV 版本: ${OPENCV_VERSION}"
echo "    ONNX Runtime: ${ORT_VERSION} (GPU: ${USE_ORT_GPU})"
echo "================================================================================"

mkdir -p "${THIRDPARTY_DIR}"

# 创建临时工作目录并在退出时自动清理
TMP_BUILD_DIR="$(mktemp -d -t maskrcnn_3rdparty_build_XXXXXX)"
cleanup() {
    echo ">>> 清理临时下载与构建目录: ${TMP_BUILD_DIR} ..."
    rm -rf "${TMP_BUILD_DIR}"
    echo ">>> 临时文件清理完成。"
}
trap cleanup EXIT

# 检查基础工具依赖
check_cmd() {
    if ! command -v "$1" &>/dev/null; then
        echo "错误: 未找到依赖工具 '$1'，请先安装 (例如: sudo apt-get install -y $1)"
        exit 1
    fi
}

check_cmd gcc
check_cmd g++
check_cmd cmake
check_cmd make
check_cmd tar

DOWNLOADER="curl"
if command -v curl &>/dev/null; then
    download_file() {
        local url="$1"
        local output="$2"
        echo ">>> 下载: ${url} -> ${output}"
        curl -fSL --retry 3 --connect-timeout 15 "${url}" -o "${output}"
    }
elif command -v wget &>/dev/null; then
    DOWNLOADER="wget"
    download_file() {
        local url="$1"
        local output="$2"
        echo ">>> 下载: ${url} -> ${output}"
        wget -q --show-progress -c --tries=3 --timeout=15 "${url}" -O "${output}"
    }
else
    echo "错误: 需要 curl 或 wget 下载工具，请通过 apt 安装。"
    exit 1
fi

# ==============================================================================
# 1. 编译并安装 OpenCV
# ==============================================================================
OPENCV_INSTALL_DIR="${THIRDPARTY_DIR}/opencv"

if [[ -f "${OPENCV_INSTALL_DIR}/lib/libopencv_core.so" && "${FORCE_REBUILD}" -eq 0 ]]; then
    echo ">>> 检测到已存在的 OpenCV 安装: ${OPENCV_INSTALL_DIR}，跳过编译 (使用 --force 可重新编译)。"
else
    echo ">>> 开始下载并编译 OpenCV ${OPENCV_VERSION} ..."
    OPENCV_TAR="${TMP_BUILD_DIR}/opencv-${OPENCV_VERSION}.tar.gz"
    OPENCV_SRC="${TMP_BUILD_DIR}/opencv-${OPENCV_VERSION}"
    OPENCV_BDIR="${TMP_BUILD_DIR}/opencv-build"

    download_file "https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VERSION}.tar.gz" "${OPENCV_TAR}"

    echo ">>> 解压 OpenCV 源码包 ..."
    tar -xzf "${OPENCV_TAR}" -C "${TMP_BUILD_DIR}"

    mkdir -p "${OPENCV_BDIR}"
    echo ">>> 配置 OpenCV CMake 工程 ..."
    cmake -S "${OPENCV_SRC}" -B "${OPENCV_BDIR}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${OPENCV_INSTALL_DIR}" \
        -DBUILD_LIST=core,imgproc,imgcodecs \
        -DBUILD_SHARED_LIBS=ON \
        -DBUILD_TESTS=OFF \
        -DBUILD_PERF_TESTS=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_DOCS=OFF \
        -DBUILD_opencv_apps=OFF \
        -DBUILD_JAVA=OFF \
        -DBUILD_opencv_python2=OFF \
        -DBUILD_opencv_python3=OFF \
        -DBUILD_ZLIB=ON \
        -DBUILD_JPEG=ON \
        -DBUILD_PNG=ON \
        -DBUILD_TIFF=OFF \
        -DBUILD_WEBP=OFF \
        -DBUILD_OPENJPEG=OFF \
        -DBUILD_JASPER=OFF \
        -DBUILD_OPENEXR=OFF \
        -DWITH_TIFF=OFF \
        -DWITH_WEBP=OFF \
        -DWITH_OPENJPEG=OFF \
        -DWITH_JASPER=OFF \
        -DWITH_OPENEXR=OFF \
        -DWITH_PROTOBUF=OFF \
        -DWITH_CUDA=OFF \
        -DWITH_IPP=OFF \
        -DWITH_OPENCL=OFF \
        -DWITH_FFMPEG=OFF \
        -DWITH_GSTREAMER=OFF \
        -DWITH_1394=OFF \
        -DWITH_TBB=OFF \
        "-DCMAKE_IGNORE_PATH=/home/zja/anaconda3;/home/zja/miniconda3"

    NUM_JOBS="$(nproc || echo 4)"
    echo ">>> 正在使用 ${NUM_JOBS} 线程编译 OpenCV ..."
    cmake --build "${OPENCV_BDIR}" --target install -j"${NUM_JOBS}"

    echo ">>> OpenCV 编译并安装成功: ${OPENCV_INSTALL_DIR}"
fi

# ==============================================================================
# 2. 准备 ONNX Runtime
# ==============================================================================
if [[ "${USE_ORT_GPU}" -eq 1 ]]; then
    ORT_PKG_NAME="onnxruntime-linux-x64-gpu-${ORT_VERSION}"
    ORT_TARGET_DIR="${THIRDPARTY_DIR}/onnxruntime-cuda11-${ORT_VERSION}"
else
    ORT_PKG_NAME="onnxruntime-linux-x64-${ORT_VERSION}"
    ORT_TARGET_DIR="${THIRDPARTY_DIR}/onnxruntime-linux-x64-${ORT_VERSION}"
fi

if [[ -f "${ORT_TARGET_DIR}/lib/libonnxruntime.so" && "${FORCE_REBUILD}" -eq 0 ]]; then
    echo ">>> 检测到已存在的 ONNX Runtime: ${ORT_TARGET_DIR}，跳过下载 (使用 --force 可重新准备)。"
else
    echo ">>> 准备 ONNX Runtime ${ORT_VERSION} (${ORT_PKG_NAME}) ..."
    ORT_TAR="${TMP_BUILD_DIR}/${ORT_PKG_NAME}.tgz"
    ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_PKG_NAME}.tgz"

    download_file "${ORT_URL}" "${ORT_TAR}"

    echo ">>> 解压 ONNX Runtime ..."
    tar -xzf "${ORT_TAR}" -C "${TMP_BUILD_DIR}"

    rm -rf "${ORT_TARGET_DIR}"
    mv "${TMP_BUILD_DIR}/${ORT_PKG_NAME}" "${ORT_TARGET_DIR}"

    # 规范化软链接
    pushd "${ORT_TARGET_DIR}/lib" >/dev/null
    if [[ -f "libonnxruntime.so.${ORT_VERSION}" ]]; then
        ln -sf "libonnxruntime.so.${ORT_VERSION}" "libonnxruntime.so.1"
        ln -sf "libonnxruntime.so.1" "libonnxruntime.so"
    fi
    popd >/dev/null

    echo ">>> ONNX Runtime 准备完成: ${ORT_TARGET_DIR}"
fi

echo "================================================================================"
echo ">>> 恭喜！3rdparty 依赖已全部就绪！"
echo "    OpenCV: ${OPENCV_INSTALL_DIR}"
echo "    ONNX Runtime: ${ORT_TARGET_DIR}"
echo "    可在仓库根目录执行: cmake -B build && cmake --build build"
echo "================================================================================"
