# ==============================================================================
# Script: shell/setup_and_build_all_windows.ps1
# Description: Windows 10/11 PowerShell 全自动一键流程：
#              1. 安装 MinGW64 与 CMake 并配置环境变量
#              2. 编译并安装 OpenCV 与 ONNX Runtime 到 3rdparty
#              3. 编译本项目可执行文件并运行模型推理测试
#              4. 全程自动清理下载和中间构建临时文件
# ==============================================================================

[CmdletBinding()]
param (
    [Parameter(Mandatory = $false)]
    [string]$Device = "cpu"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ">>> [全流程一键] 正在执行 Windows 编译环境准备、依赖库构建与推理测试" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan

# 步骤 1: 安装环境与配置 PATH
Write-Host ">>> [Step 1/3] 检查并配置 MinGW-w64 与 CMake 环境 ..." -ForegroundColor Yellow
& "$ScriptDir\setup_env_windows.ps1"

# 步骤 2: 编译/准备 3rdparty 依赖库
Write-Host ">>> [Step 2/3] 编译并配置 3rdparty 依赖库 (OpenCV + ONNX Runtime) ..." -ForegroundColor Yellow
& "$ScriptDir\build_3rdparty_windows.ps1"

# 步骤 3: 编译工程并运行推理测试
Write-Host ">>> [Step 3/3] 编译项目推理程序并运行测试验证 ..." -ForegroundColor Yellow
& "$ScriptDir\build_and_run_windows.ps1" -Device $Device

Write-Host "================================================================================" -ForegroundColor Green
Write-Host ">>> [全流程一键] 执行完毕！所有组件均已就绪且推理测试通过！" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Green
