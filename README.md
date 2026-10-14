# Mask R-CNN (Swin-S) C++ 推理与评估

本项目提供基于 ONNX Runtime C++ 的 Mask R-CNN 实例分割推理实现，以及用于格式转换与精度评估的 Python 工具。
支持在 **Ubuntu 18.04+** 与 **Windows 10/11** 系统下编译依赖、构建程序并稳定运行。

---

## 目录
- [1. 自动化工具箱 (shell 文件夹)](#1-自动化工具箱-shell-文件夹)
- [2. Ubuntu 18.04+ 编译与运行](#2-ubuntu-1804-编译与运行)
  - [2.1 一键编译依赖到 3rdparty](#21-一键编译依赖到-3rdparty)
  - [2.2 一键编译程序并运行推理](#22-一键编译程序并运行推理)
  - [2.3 手动构建](#23-手动构建)
- [3. Windows 10/11 (MinGW-w64) 编译与运行](#3-windows-1011-mingw-w64-编译与运行)
  - [3.1 PowerShell 一键安装环境 (MinGW + CMake)](#31-powershell-一键安装环境-mingw--cmake)
  - [3.2 PowerShell 一键编译依赖到 3rdparty](#32-powershell-一键编译依赖到-3rdparty)
  - [3.3 PowerShell 一键编译程序并运行推理](#33-powershell-一键编译程序并运行推理)
  - [3.4 Windows 全流程一条龙脚本](#34-windows-全流程一条龙脚本)
  - [3.5 临时文件与清理机制说明](#35-临时文件与清理机制说明)
- [4. 模型推理调用方式](#4-模型推理调用方式)
- [5. Python 工具与评估](#5-python-工具与评估)

---

## 1. 自动化工具箱 (shell 文件夹)

仓库提供 `shell/` 文件夹，包含 Ubuntu 18.04 与 Windows 10/11 的自动化构建与环境配置脚本：

| 平台 | 脚本文件 | 功能说明 |
| :--- | :--- | :--- |
| **Ubuntu 18.04** | `shell/build_3rdparty_ubuntu1804.sh` | 从源码编译 OpenCV 4.5.5 并下载配置 ONNX Runtime 到 `3rdparty/`，完成后自动清理临时文件 |
| **Ubuntu 18.04** | `shell/build_and_run_ubuntu.sh` | 一键编译本项目 `test_model_inference` 并运行样例推理验证 |
| **Windows 10/11** | `shell/setup_env_windows.ps1` | PowerShell 一键静默安装 MinGW64 与 CMake 4.4.4 并自动配置环境变量，完成后自动清理安装包 |
| **Windows 10/11** | `shell/build_3rdparty_windows.ps1` | 使用 MinGW-w64 源码编译 OpenCV 4.5.5 并配置 ONNX Runtime 到 `3rdparty/`，完成后自动清理中间构建产物 |
| **Windows 10/11** | `shell/build_and_run_windows.ps1` | 使用 MinGW 编译项目程序，自动拷贝所需 DLL 到输出目录并执行模型推理测试 |
| **Windows 10/11** | `shell/setup_and_build_all_windows.ps1` | 全流程一键串联：环境安装 -> 依赖编译 -> 项目构建 -> 推理验证 |

---

## 2. Ubuntu 18.04+ 编译与运行

### 2.1 一键编译依赖到 3rdparty
在仓库根目录执行：
```bash
bash shell/build_3rdparty_ubuntu1804.sh
```
- **工作机制**：
  - 自动下载 OpenCV 4.5.5 源码，针对本项目所需的核心模块（`core`、`imgproc`、`imgcodecs`）进行极速构建并安装到 `3rdparty/opencv`。
  - 自动下载部署官方 ONNX Runtime 1.20.0 发行库到 `3rdparty/onnxruntime-cuda11-1.20.0`，并建立符号链接。
  - 支持参数：`--gpu`（默认 GPU/CUDA 版）或 `--cpu`（下载 CPU 版），以及 `--force`（强制重新编译覆盖）。
  - **自动清理**：下载的 `.tar.gz` 源码包与中间构建目录在脚本结束时通过 trap 机制彻底清除，不占额外磁盘空间。

### 2.2 一键编译程序并运行推理
```bash
# 默认使用 CPU 运行 followed 数据集推理验证
bash shell/build_and_run_ubuntu.sh cpu followed

# 运行 fixed 数据集推理验证
bash shell/build_and_run_ubuntu.sh cpu fixed

# 若配置有 CUDA/GPU 环境，可指定 gpu
bash shell/build_and_run_ubuntu.sh gpu followed
```

### 2.3 手动构建
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

---

## 3. Windows 10/11 (MinGW-w64) 编译与运行

在 Windows 10/11 上，无需安装体量庞大的 Visual Studio，推荐使用专为现代 C++ 优化的 **MinGW-w64 (GCC 16.2.0 UCRT)** 与 **CMake 4.4.4**。

### 3.1 PowerShell 一键安装环境 (MinGW + CMake)
以管理员或普通用户身份打开 **PowerShell**，进入项目目录执行：
```powershell
.\shell\setup_env_windows.ps1
```
- **安装内容**：
  1. **MinGW-w64**: 下载来自 niXman 官方发布的 GCC 16.2.0 UCRT 版本：
     [`x86_64-16.2.0-release-win32-seh-ucrt-rt_v14-rev1.7z`](https://github.com/niXman/mingw-builds-binaries/releases/download/16.2.0-rt_v14-rev1/x86_64-16.2.0-release-win32-seh-ucrt-rt_v14-rev1.7z)
     自动解压并部署到 `C:\mingw64`（若权限受限则自动回退至用户目录 `$HOME\mingw64`）。
  2. **CMake**: 下载官方安装包：
     [`cmake-4.4.4-windows-x86_64.msi`](https://github.com/Kitware/CMake/releases/download/v4.4.4/cmake-4.4.4-windows-x86_64.msi)
     自动执行静默无感安装。
  3. **环境变量自动配置**：
     自动将 MinGW 与 CMake 的 `bin` 目录写入**用户级永久环境变量 `Path`**，并实时同步到当前 PowerShell 会话。新开终端即可全局使用 `gcc`、`g++` 与 `cmake`。
  4. **清理**：安装完成后，下载的 `.msi` 与 `.7z` 临时安装包立即自动删除。

### 3.2 PowerShell 一键编译依赖到 3rdparty
在 PowerShell 中执行：
```powershell
.\shell\build_3rdparty_windows.ps1
```
- **工作机制**：
  - 自动获取 OpenCV 4.5.5 源码，调用 MinGW-w64 (`MinGW Makefiles`) 编译核心动态库并安装至 `3rdparty\opencv`。
  - 自动获取 ONNX Runtime 1.20.0 Windows x64 预编译发行包，部署至 `3rdparty\onnxruntime-win-x64-1.20.0`，并自动生成 MinGW 兼容导入库 `libonnxruntime.dll.a`。
  - **自动清理**：编译安装完成后，OpenCV 源码包、临时解压目录以及整个构建缓存树自动彻底删除。

### 3.3 PowerShell 一键编译程序并运行推理
在 PowerShell 中执行：
```powershell
# 运行 followed 数据集推理 (默认 CPU)
.\shell\build_and_run_windows.ps1 -Device cpu -Dataset followed

# 运行 fixed 数据集推理 (CPU)
.\shell\build_and_run_windows.ps1 -Device cpu -Dataset fixed
```
- 自动使用 MinGW 编译 `test_model_inference.exe`。
- 构建时 **CMake 自动将依赖的 `onnxruntime.dll` 及所有 OpenCV DLL 复制到可执行程序同级目录下**，无需手动搬运 DLL。
- 自动根据 `-Dataset` 参数加载对应模型（`model_followed_*.onnx` 或 `model_fixed_*.onnx`）与图片目录（`assets/followed` 或 `assets/fixed`），执行推理验证。

### 3.4 Windows 全流程一条龙脚本
若在一台全新的 Windows 10/11 电脑上，只需一条命令即可完成**环境安装 -> 依赖编译 -> 项目构建 -> 推理测试**：
```powershell
.\shell\setup_and_build_all_windows.ps1
```

### 3.5 临时文件与清理机制说明
所有自动化脚本遵循“零垃圾残留”原则：
- **下载缓存**：所有 `.msi`、`.7z`、`.zip`、`.tar.gz` 安装包保存在系统临时目录 `$env:TEMP` / `/tmp` 中，安装解压完成后即刻删除。
- **构建中间文件**：OpenCV 源码编译过程中生成的上千个临时 `.obj` / `.o` 文件与中间构建目录在执行 `install` 后立即全量清除，仅保留最终 `3rdparty` 所需的头文件与库文件。

---

## 4. 模型推理调用方式

输入支持**单张图片路径**或**图片目录**（目录推理将逐张处理并自动跳过非图片）。输出包含可视化结果图片及同名检测 JSON 文件（包含 bbox、掩码 RLE、多边形轮廓及哈希）。

### 方式一：具名参数（推荐，支持按类别设置阈值）

#### 1. 跟踪模型 (`followed`):
**Linux**:
```bash
build/test_model_inference \
  --model model/model_followed_20260906.onnx \
  --input assets/followed \
  --output build/pred_followed \
  --classes nipple,leg,tail,milkcup \
  --class-conf nipple=0.60,leg=0.50 \
  --class-iou nipple=0.50,leg=0.50 \
  --device cpu # 可选 cpu 或 gpu
```

**Windows (PowerShell)**:
```powershell
.\build\test_model_inference.exe `
  --model model\model_followed_20260906.onnx `
  --input assets\followed `
  --output build\pred_followed `
  --classes nipple,leg,tail,milkcup `
  --class-conf nipple=0.60,leg=0.50 `
  --class-iou nipple=0.50,leg=0.50 `
  --device cpu
```

#### 2. 固定模型 (`fixed`):
**Linux**:
```bash
build/test_model_inference \
  --model model/model_fixed_20260929.onnx \
  --input assets/fixed \
  --output build/pred_fixed \
  --classes leg,breast,cord,milkcup \
  --class-conf leg=0.50,breast=0.50 \
  --class-iou leg=0.50,breast=0.50 \
  --device cpu
```

**Windows (PowerShell)**:
```powershell
.\build\test_model_inference.exe `
  --model model\model_fixed_20260929.onnx `
  --input assets\fixed `
  --output build\pred_fixed `
  --classes leg,breast,cord,milkcup `
  --class-conf leg=0.50,breast=0.50 `
  --class-iou leg=0.50,breast=0.50 `
  --device cpu
```
> **说明**：`--classes` 顺序必须严格对应模型类别 ID 顺序；未指定的类别置信度与 IoU 阈值默认均为 0.50。

### 方式二：位置参数（简易调用）
格式：`test_model_inference <MODEL> <LABELS> <IMAGE_OR_DIR> <cpu|gpu> [OUTPUT_DIR]`

**Linux**:
```bash
# followed
build/test_model_inference model/model_followed_20260906.onnx model/followed.names assets/followed cpu build/pred_followed

# fixed
build/test_model_inference model/model_fixed_20260929.onnx model/fixed.names assets/fixed cpu build/pred_fixed
```

**Windows (PowerShell)**:
```powershell
# followed
.\build\test_model_inference.exe model\model_followed_20260906.onnx model\followed.names assets\followed cpu build\pred_followed

# fixed
.\build\test_model_inference.exe model\model_fixed_20260929.onnx model\fixed.names assets\fixed cpu build\pred_fixed
```

---

## 5. Python 工具与评估

### 5.1 Python 最小运行环境
Python 工具跨平台通用，仅需 Python 3.8+ 及基础依赖（**无需配置庞大的深度学习框架**）：
```bash
pip install numpy opencv-python
```
*(可选：`pip install tqdm pycocotools`，用于显示进度条或处理 COCO 压缩 RLE 格式)*

### 5.2 预测结果转 LabelMe 标注 (`convert_to_labelme.py`)
将 C++ 推理生成的 JSON 转换为 LabelMe / X-AnyLabeling 多边形标注格式，便于在标注软件中人工复核或微调：
```bash
python code/convert_to_labelme.py \
  --input build/pred_followed \
  --output-dir build/labelme_followed
```
- `--input`：输入 JSON 文件或目录。
- `--output-dir`：输出目录；若省略则原地修改，可加 `--backup` 备份原文件。

### 5.3 实例分割精度评估 (`compare_json.py`)
将推理输出与 LabelMe 真值 JSON（按文件名匹配）进行逐实例掩码 IoU 匹配评估，输出 TP/FP/FN、Precision/Recall/F1、HTML 评测报告及错误样本叠加图：
```bash
python code/compare_json.py \
  --gt-dir /path/to/ground_truth \
  --pred-dir build/pred_followed \
  --output-dir build/evaluation \
  --classes nipple leg tail milkcup \
  --iou-threshold 0.50 \
  --class-conf nipple=0.60,leg=0.50 \
  --visualize all
```
- `--classes`：待评估类别列表（空格分隔），省略则自动提取所有类别。
- `--iou-threshold`：判定为 TP 的掩码 IoU 阈值（默认 0.50）。
- `--visualize`：可视化输出模式，可选 `all`（全部）、`errors`（仅错误样本）或 `none`。
