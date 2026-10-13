# ==============================================================================
# Script: shell/build_and_run_windows.ps1
# Description: Windows 10/11 下使用 MinGW-w64 一键编译本项目并运行模型推理测试
# ==============================================================================

[CmdletBinding()]
param (
    [Parameter(Mandatory = $false)]
    [string]$Device = "cpu",

    [Parameter(Mandatory = $false)]
    [ValidateSet("followed", "fixed")]
    [string]$Dataset = "followed"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
$BuildDir = Join-Path $RepoRoot "build"

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ">>> Windows 10/11 MinGW 编译并运行 Mask R-CNN (Swin-S) 推理程序" -ForegroundColor Cyan
Write-Host "    工作目录: $RepoRoot"
Write-Host "    构建目录: $BuildDir"
Write-Host "    推理设备: $Device"
Write-Host "    数据集:   $Dataset"
Write-Host "================================================================================" -ForegroundColor Cyan

# 自动补充 MinGW 与 CMake 环境变量
function Ensure-ToolchainInPath {
    $candidateMinGW = @(
        "C:\mingw64\bin",
        (Join-Path $HOME "mingw64\bin")
    )
    foreach ($p in $candidateMinGW) {
        if (Test-Path (Join-Path $p "gcc.exe")) {
            if (-not ($env:Path -split ';' | Where-Object { $_ -eq $p })) {
                $env:Path = "$p;$env:Path"
            }
            break
        }
    }

    if (-not (Get-Command "cmake" -ErrorAction SilentlyContinue)) {
        $candidateCMake = @(
            "C:\Program Files\CMake\bin",
            "C:\Program Files (x86)\CMake\bin",
            "$env:ProgramFiles\CMake\bin"
        )
        foreach ($p in $candidateCMake) {
            if (Test-Path (Join-Path $p "cmake.exe")) {
                $env:Path = "$p;$env:Path"
                break
            }
        }
    }
}
Ensure-ToolchainInPath

# 检查 3rdparty 依赖库是否存在
$opencvDir = Join-Path $RepoRoot "3rdparty\opencv"
if (-not (Test-Path $opencvDir)) {
    Write-Host ">>> 未检测到 3rdparty/opencv，正在自动调用依赖编译脚本..." -ForegroundColor Yellow
    & "$ScriptDir\build_3rdparty_windows.ps1"
}

# 1. CMake 构建
Write-Host ">>> 正在配置 CMake (MinGW Makefiles) ..." -ForegroundColor Yellow
& cmake -S $RepoRoot -B $BuildDir -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
if ($LASTEXITCODE -ne 0) {
    throw "CMake 配置失败！"
}

$numProcs = if ($env:NUMBER_OF_PROCESSORS) { $env:NUMBER_OF_PROCESSORS } else { 4 }
Write-Host ">>> 正在编译工程 (使用 $numProcs 线程) ..." -ForegroundColor Yellow
& cmake --build $BuildDir -j $numProcs
if ($LASTEXITCODE -ne 0) {
    throw "项目编译失败！"
}

$exePath = Join-Path $BuildDir "test_model_inference.exe"
if (-not (Test-Path $exePath)) {
    # 兼容可能输出在 Release/ 目录下的情况
    $exeAlt = Join-Path $BuildDir "Release\test_model_inference.exe"
    if (Test-Path $exeAlt) { $exePath = $exeAlt }
}

Write-Host ">>> 编译完成！可执行程序: $exePath" -ForegroundColor Green

# 2. 运行模型推理测试
if ($Dataset -eq "fixed") {
    $modelPath = Join-Path $RepoRoot "model\model_fixed_20260929.onnx"
    $namesPath = Join-Path $RepoRoot "model\fixed.names"
    $inputPath = Join-Path $RepoRoot "assets\fixed"
    $outputPath = Join-Path $BuildDir "pred_fixed"
} else {
    $modelPath = Join-Path $RepoRoot "model\model_followed_20260906.onnx"
    $namesPath = Join-Path $RepoRoot "model\followed.names"
    $inputPath = Join-Path $RepoRoot "assets\followed"
    $outputPath = Join-Path $BuildDir "pred_followed"
}

if ((Test-Path $modelPath) -and (Test-Path $namesPath) -and (Test-Path $inputPath)) {
    Write-Host "================================================================================" -ForegroundColor Cyan
    Write-Host ">>> 开始运行模型推理测试 ..." -ForegroundColor Cyan
    Write-Host "================================================================================" -ForegroundColor Cyan

    & $exePath $modelPath $namesPath $inputPath $Device $outputPath
    if ($LASTEXITCODE -eq 0) {
        Write-Host "================================================================================" -ForegroundColor Green
        Write-Host ">>> 推理测试成功！结果输出目录: $outputPath" -ForegroundColor Green
        Write-Host "================================================================================" -ForegroundColor Green
    } else {
        Write-Error "推理运行退出码非零: $LASTEXITCODE"
    }
} else {
    Write-Host ">>> 提示: 模型文件或测试图片不存在，跳过自动推理测试。" -ForegroundColor Yellow
}
