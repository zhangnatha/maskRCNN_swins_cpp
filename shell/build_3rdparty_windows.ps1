# ==============================================================================
# Script: shell/build_3rdparty_windows.ps1
# Description: Windows 10/11 PowerShell 下一键通过 MinGW-w64 与 CMake
#              编译/配置 3rdparty 依赖库 (OpenCV + ONNX Runtime) 到 3rdparty 文件夹。
#              构建安装完成后自动清理下载的压缩包和源码编译临时目录。
# ==============================================================================

[CmdletBinding()]
param (
    [Parameter(Mandatory = $false)]
    [string]$OpenCvVersion = "4.5.5",

    [Parameter(Mandatory = $false)]
    [string]$OrtVersion = "1.20.0",

    [Parameter(Mandatory = $false)]
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
$ThirdPartyDir = Join-Path $RepoRoot "3rdparty"

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ">>> Windows 10/11 依赖库一键构建脚本 (MinGW-w64 + CMake)" -ForegroundColor Cyan
Write-Host "    仓库根目录: $RepoRoot"
Write-Host "    依赖目标目录: $ThirdPartyDir"
Write-Host "    OpenCV 版本: $OpenCvVersion"
Write-Host "    ONNX Runtime 版本: $OrtVersion"
Write-Host "================================================================================" -ForegroundColor Cyan

# 自动定位 MinGW 与 CMake 路径（若当前 PATH 中不存在）
function Ensure-ToolchainInPath {
    $needPathUpdate = $false
    
    $candidateMinGW = @(
        "C:\mingw64\bin",
        (Join-Path $HOME "mingw64\bin")
    )
    foreach ($p in $candidateMinGW) {
        if (Test-Path (Join-Path $p "gcc.exe")) {
            if (-not ($env:Path -split ';' | Where-Object { $_ -eq $p })) {
                $env:Path = "$p;$env:Path"
                Write-Host ">>> 已将 MinGW 自动加入当前 PATH: $p" -ForegroundColor Gray
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
                Write-Host ">>> 已将 CMake 自动加入当前 PATH: $p" -ForegroundColor Gray
                $needPathUpdate = $true
                break
            }
        }
    }

    if (-not (Get-Command "gcc" -ErrorAction SilentlyContinue) -or -not (Get-Command "cmake" -ErrorAction SilentlyContinue)) {
        Write-Error @"
未在当前环境中找到 gcc 或 cmake！
请先运行环境一键安装脚本:
    .\shell\setup_env_windows.ps1
或者手动将 MinGW 与 CMake 的 bin 目录加入环境变量 PATH。
"@
        exit 1
    }
}

Ensure-ToolchainInPath

if (-not (Test-Path $ThirdPartyDir)) {
    New-Item -ItemType Directory -Path $ThirdPartyDir -Force | Out-Null
}

# 创建独立临时工作区
$TempWorkDir = Join-Path $env:TEMP ("maskrcnn_3rdparty_build_" + [System.Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $TempWorkDir -Force | Out-Null

function Download-WithProgress {
    param (
        [string]$Url,
        [string]$OutputFile
    )
    Write-Host ">>> 正在下载: $Url -> $OutputFile ..." -ForegroundColor Yellow
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.SecurityProtocolType]::Tls12 -bor [System.Net.SecurityProtocolType]::Tls13
    
    try {
        $webClient = New-Object System.Net.WebClient
        $webClient.DownloadFile($Url, $OutputFile)
    } catch {
        Write-Warning "WebClient 下载异常，切换 Invoke-WebRequest..."
        Invoke-WebRequest -Uri $Url -OutFile $OutputFile -UseBasicParsing
    }
}

try {
    # --------------------------------------------------------------------------
    # 1. 下载并使用 MinGW 编译 OpenCV
    # --------------------------------------------------------------------------
    $openCvInstallDir = Join-Path $ThirdPartyDir "opencv"
    $openCvExists = (Test-Path (Join-Path $openCvInstallDir "include")) -and `
                    ((Test-Path (Join-Path $openCvInstallDir "lib")) -or (Test-Path (Join-Path $openCvInstallDir "x64\mingw\lib")))

    if ($openCvExists -and -not $Force) {
        Write-Host ">>> 检测到已存在的 OpenCV 目录: $openCvInstallDir，跳过编译 (使用 -Force 可重新编译)。" -ForegroundColor Green
    } else {
        Write-Host "--------------------------------------------------------------------------------" -ForegroundColor Cyan
        Write-Host ">>> 步骤 1/2: 下载并使用 MinGW 编译 OpenCV $OpenCvVersion ..." -ForegroundColor Cyan

        $openCvZip = Join-Path $TempWorkDir "opencv-$OpenCvVersion.zip"
        $openCvUrl = "https://github.com/opencv/opencv/archive/refs/tags/$OpenCvVersion.zip"
        Download-WithProgress -Url $openCvUrl -OutputFile $openCvZip

        Write-Host ">>> 解压 OpenCV 源码 ..." -ForegroundColor Yellow
        $openCvSrcDir = Join-Path $TempWorkDir "opencv-src"
        Expand-Archive -Path $openCvZip -DestinationPath $openCvSrcDir -Force

        $srcRoot = Join-Path $openCvSrcDir "opencv-$OpenCvVersion"
        $buildRoot = Join-Path $TempWorkDir "opencv-build"
        New-Item -ItemType Directory -Path $buildRoot -Force | Out-Null

        Write-Host ">>> 配置 OpenCV CMake (MinGW Makefiles) ..." -ForegroundColor Yellow
        $cmakeArgs = @(
            "-S", $srcRoot,
            "-B", $buildRoot,
            "-G", "MinGW Makefiles",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_INSTALL_PREFIX=$openCvInstallDir",
            "-DBUILD_LIST=core,imgproc,imgcodecs",
            "-DBUILD_SHARED_LIBS=ON",
            "-DBUILD_TESTS=OFF",
            "-DBUILD_PERF_TESTS=OFF",
            "-DBUILD_EXAMPLES=OFF",
            "-DBUILD_DOCS=OFF",
            "-DBUILD_opencv_apps=OFF",
            "-DBUILD_JAVA=OFF",
            "-DBUILD_opencv_python2=OFF",
            "-DBUILD_opencv_python3=OFF",
            "-DBUILD_ZLIB=ON",
            "-DBUILD_JPEG=ON",
            "-DBUILD_PNG=ON",
            "-DWITH_CUDA=OFF",
            "-DWITH_IPP=OFF",
            "-DWITH_OPENCL=OFF",
            "-DWITH_MSMF=OFF",
            "-DENABLE_PRECOMPILED_HEADERS=OFF"
        )
        & cmake @cmakeArgs
        if ($LASTEXITCODE -ne 0) {
            throw "OpenCV CMake 配置失败！"
        }

        $numProcs = if ($env:NUMBER_OF_PROCESSORS) { $env:NUMBER_OF_PROCESSORS } else { 4 }
        Write-Host ">>> 正在使用 $numProcs 线程编译并安装 OpenCV ..." -ForegroundColor Yellow
        & cmake --build $buildRoot --target install -j $numProcs
        if ($LASTEXITCODE -ne 0) {
            throw "OpenCV 编译安装失败！"
        }

        Write-Host ">>> OpenCV 编译与安装成功: $openCvInstallDir" -ForegroundColor Green
    }

    # --------------------------------------------------------------------------
    # 2. 准备 ONNX Runtime Windows x64 预编译库
    # --------------------------------------------------------------------------
    $ortTargetDir = Join-Path $ThirdPartyDir "onnxruntime-win-x64-$OrtVersion"
    $ortExists = (Test-Path (Join-Path $ortTargetDir "include")) -and (Test-Path (Join-Path $ortTargetDir "lib\onnxruntime.dll"))

    if ($ortExists -and -not $Force) {
        Write-Host ">>> 检测到已存在的 ONNX Runtime: $ortTargetDir，跳过下载 (使用 -Force 可重新下载)。" -ForegroundColor Green
    } else {
        Write-Host "--------------------------------------------------------------------------------" -ForegroundColor Cyan
        Write-Host ">>> 步骤 2/2: 下载并配置 ONNX Runtime Windows x64 ($OrtVersion) ..." -ForegroundColor Cyan

        $ortZip = Join-Path $TempWorkDir "onnxruntime-win-x64-$OrtVersion.zip"
        $ortUrl = "https://github.com/microsoft/onnxruntime/releases/download/v$OrtVersion/onnxruntime-win-x64-$OrtVersion.zip"
        Download-WithProgress -Url $ortUrl -OutputFile $ortZip

        Write-Host ">>> 解压 ONNX Runtime ..." -ForegroundColor Yellow
        $ortExtractDir = Join-Path $TempWorkDir "ort_unpacked"
        Expand-Archive -Path $ortZip -DestinationPath $ortExtractDir -Force

        if (Test-Path $ortTargetDir) {
            Remove-Item -Path $ortTargetDir -Recurse -Force
        }
        $innerOrt = Join-Path $ortExtractDir "onnxruntime-win-x64-$OrtVersion"
        if (Test-Path $innerOrt) {
            Move-Item -Path $innerOrt -Destination $ortTargetDir -Force
        } else {
            Move-Item -Path $ortExtractDir -Destination $ortTargetDir -Force
        }

        # 若存在 gendef 与 dlltool，为 MinGW 生成专用导入库 libonnxruntime.dll.a
        $gendefCmd = Get-Command "gendef" -ErrorAction SilentlyContinue
        $dlltoolCmd = Get-Command "dlltool" -ErrorAction SilentlyContinue
        $ortLibDir = Join-Path $ortTargetDir "lib"
        $ortDllPath = Join-Path $ortLibDir "onnxruntime.dll"

        if ($gendefCmd -and $dlltoolCmd -and (Test-Path $ortDllPath)) {
            Write-Host ">>> 使用 gendef & dlltool 为 MinGW 生成 libonnxruntime.dll.a ..." -ForegroundColor Yellow
            Push-Location $ortLibDir
            try {
                & gendef onnxruntime.dll
                if (Test-Path "onnxruntime.def") {
                    & dlltool -d onnxruntime.def -l libonnxruntime.dll.a -D onnxruntime.dll
                    Remove-Item "onnxruntime.def" -Force -ErrorAction SilentlyContinue
                    Write-Host ">>> 成功生成 MinGW 导入库: libonnxruntime.dll.a" -ForegroundColor Green
                }
            } catch {
                Write-Warning "生成 libonnxruntime.dll.a 出现提示，GCC 仍可直接链接 onnxruntime.lib / onnxruntime.dll。"
            } finally {
                Pop-Location
            }
        }

        Write-Host ">>> ONNX Runtime 准备完成: $ortTargetDir" -ForegroundColor Green
    }

} finally {
    # --------------------------------------------------------------------------
    # 3. 清理下载与临时编译目录
    # --------------------------------------------------------------------------
    Write-Host "--------------------------------------------------------------------------------" -ForegroundColor Cyan
    Write-Host ">>> 步骤: 清理临时下载与构建目录: $TempWorkDir ..." -ForegroundColor Yellow
    if (Test-Path $TempWorkDir) {
        Remove-Item -Path $TempWorkDir -Recurse -Force -ErrorAction SilentlyContinue
    }
    Write-Host ">>> 临时文件已彻底清理完成！" -ForegroundColor Green
}

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ">>> 依赖库构建与配置全部完成！" -ForegroundColor Green
Write-Host "    OpenCV:       $openCvInstallDir"
Write-Host "    ONNX Runtime: $ortTargetDir"
Write-Host "可在仓库根目录执行: cmake -B build -G `"MinGW Makefiles`" && cmake --build build" -ForegroundColor Cyan
Write-Host "或直接执行: .\shell\build_and_run_windows.ps1" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
