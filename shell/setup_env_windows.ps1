# ==============================================================================
# Script: shell/setup_env_windows.ps1
# Description: Windows 10/11 PowerShell 一键自动下载与安装 MinGW64 与 CMake，
#              自动配置环境变量，安装完成后自动清理下载与临时文件。
# ==============================================================================

[CmdletBinding()]
param (
    [Parameter(Mandatory = $false)]
    [string]$MinGwInstallDir = "C:\mingw64",

    [Parameter(Mandatory = $false)]
    [switch]$SkipCMake,

    [Parameter(Mandatory = $false)]
    [switch]$SkipMinGW
)

$ErrorActionPreference = "Stop"

$MinGwUrl = "https://github.com/niXman/mingw-builds-binaries/releases/download/16.2.0-rt_v14-rev1/x86_64-16.2.0-release-win32-seh-ucrt-rt_v14-rev1.7z"
$CMakeUrl = "https://github.com/Kitware/CMake/releases/download/v4.4.4/cmake-4.4.4-windows-x86_64.msi"

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ">>> Windows 10/11 编译环境一键安装脚本 (MinGW-w64 + CMake)" -ForegroundColor Cyan
Write-Host "    MinGW URL: $MinGwUrl"
Write-Host "    CMake URL: $CMakeUrl"
Write-Host "    MinGW 目标路径: $MinGwInstallDir"
Write-Host "================================================================================" -ForegroundColor Cyan

# 创建专用临时目录
$TempWorkDir = Join-Path $env:TEMP ("maskrcnn_env_setup_" + [System.Guid]::NewGuid().ToString("N"))
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
        Write-Warning "WebClient 下载失败，尝试 Invoke-WebRequest..."
        Invoke-WebRequest -Uri $Url -OutFile $OutputFile -UseBasicParsing
    }
}

function Add-PathToEnvironment {
    param (
        [string]$NewDirectory
    )
    if (-not (Test-Path $NewDirectory)) { return }

    # 1. 刷新当前进程 PATH
    if (-not ($env:Path -split ';' | Where-Object { $_ -eq $NewDirectory })) {
        $env:Path = "$NewDirectory;$env:Path"
        Write-Host ">>> [当前会话] 已添加 PATH: $NewDirectory" -ForegroundColor Green
    }

    # 2. 写入用户级永久环境变量 PATH
    try {
        $userPath = [Environment]::GetEnvironmentVariable("Path", [System.EnvironmentVariableTarget]::User)
        $userPaths = if ($userPath) { $userPath -split ';' } else { @() }
        if (-not ($userPaths | Where-Object { $_ -eq $NewDirectory })) {
            $newUserPath = if ($userPath) { "$NewDirectory;$userPath" } else { $NewDirectory }
            [Environment]::SetEnvironmentVariable("Path", $newUserPath, [System.EnvironmentVariableTarget]::User)
            Write-Host ">>> [永久用户环境] 已成功写入 PATH: $NewDirectory" -ForegroundColor Green
        }
    } catch {
        Write-Warning "写入永久环境变量失败，请检查权限: $_"
    }
}

try {
    # --------------------------------------------------------------------------
    # 1. 安装 CMake
    # --------------------------------------------------------------------------
    if (-not $SkipCMake) {
        Write-Host "--------------------------------------------------------------------------------" -ForegroundColor Cyan
        Write-Host ">>> 步骤 1/2: 检查并安装 CMake (4.4.4) ..." -ForegroundColor Cyan
        
        $cmakeInstalled = $false
        try {
            $existingCMake = Get-Command "cmake" -ErrorAction SilentlyContinue
            if ($existingCMake) {
                $verOutput = (& cmake --version) | Select-Object -First 1
                Write-Host ">>> 系统已存在 CMake: $verOutput，自动跳过重复下载与安装。" -ForegroundColor Green
                $cmakeInstalled = $true
            }
        } catch {}

        if (-not $cmakeInstalled) {
            $CMakeMsi = Join-Path $TempWorkDir "cmake-4.4.4-windows-x86_64.msi"
            Download-WithProgress -Url $CMakeUrl -OutputFile $CMakeMsi

            Write-Host ">>> 正在静默安装 CMake MSI ..." -ForegroundColor Yellow
            $msiProcess = Start-Process -FilePath "msiexec.exe" -ArgumentList "/i `"$CMakeMsi`" /quiet /norestart ADD_CMAKE_TO_PATH=User" -Wait -PassThru
            if ($msiProcess.ExitCode -ne 0 -and $msiProcess.ExitCode -ne 3010) {
                Write-Warning "MSI 退出码: $($msiProcess.ExitCode)，重试无附加参数安装..."
                Start-Process -FilePath "msiexec.exe" -ArgumentList "/i `"$CMakeMsi`" /quiet /norestart" -Wait | Out-Null
            }

            # 查找 CMake 安装目录
            $cmakeCandidates = @(
                "C:\Program Files\CMake\bin",
                "C:\Program Files (x86)\CMake\bin",
                "$env:ProgramFiles\CMake\bin"
            )
            $cmakeFound = $false
            foreach ($cand in $cmakeCandidates) {
                if (Test-Path (Join-Path $cand "cmake.exe")) {
                    Add-PathToEnvironment -NewDirectory $cand
                    $cmakeFound = $true
                    break
                }
            }
            if (-not $cmakeFound) {
                Write-Warning "未在默认路径定位到 cmake.exe，请确认安装路径并将 bin 加入环境变量。"
            }
        }
    }

    # --------------------------------------------------------------------------
    # 2. 安装 MinGW-w64 (GCC 16.2.0 UCRT)
    # --------------------------------------------------------------------------
    if (-not $SkipMinGW) {
        Write-Host "--------------------------------------------------------------------------------" -ForegroundColor Cyan
        Write-Host ">>> 步骤 2/2: 下载并安装 MinGW-w64 (16.2.0-rt_v14-rev1 UCRT) ..." -ForegroundColor Cyan

        # 测试目标安装目录是否可写；若不可写则回退到用户主目录
        try {
            if (-not (Test-Path $MinGwInstallDir)) {
                New-Item -ItemType Directory -Path $MinGwInstallDir -Force -ErrorAction Stop | Out-Null
            }
        } catch {
            $fallbackDir = Join-Path $HOME "mingw64"
            Write-Warning "无法写入 $MinGwInstallDir，自动切换安装路径到: $fallbackDir"
            $MinGwInstallDir = $fallbackDir
            New-Item -ItemType Directory -Path $MinGwInstallDir -Force | Out-Null
        }

        $MinGw7z = Join-Path $TempWorkDir "mingw64.7z"
        Download-WithProgress -Url $MinGwUrl -OutputFile $MinGw7z

        Write-Host ">>> 正在解压 MinGW-w64 压缩包 (7z 格式) ..." -ForegroundColor Yellow
        $extractSuccess = $false

        # 方案 A: 优先使用系统已有的 7z.exe
        $sevenZipExe = $null
        $sevenZipCandidates = @(
            (Get-Command "7z" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue),
            "C:\Program Files\7-Zip\7z.exe",
            "C:\Program Files (x86)\7-Zip\7z.exe"
        )
        foreach ($c in $sevenZipCandidates) {
            if ($c -and (Test-Path $c)) { $sevenZipExe = $c; break }
        }

        if ($sevenZipExe) {
            Write-Host ">>> 使用已有 7z 工具: $sevenZipExe" -ForegroundColor Gray
            $unpackDir = Join-Path $TempWorkDir "unpacked"
            & $sevenZipExe x "$MinGw7z" "-o$unpackDir" -y | Out-Null
            if (Test-Path (Join-Path $unpackDir "mingw64\bin\gcc.exe")) {
                $extractSuccess = $true
            }
        }

        # 方案 B: 尝试 Windows 11/新版自带的 tar.exe
        if (-not $extractSuccess) {
            $tarCmd = Get-Command "tar" -ErrorAction SilentlyContinue
            if ($tarCmd) {
                try {
                    Write-Host ">>> 尝试使用 Windows 自带 tar.exe 解压 7z ..." -ForegroundColor Gray
                    $unpackDir = Join-Path $TempWorkDir "unpacked"
                    New-Item -ItemType Directory -Path $unpackDir -Force | Out-Null
                    & tar.exe -xf "$MinGw7z" -C "$unpackDir" 2>$null
                    if ($LASTEXITCODE -eq 0 -and (Test-Path (Join-Path $unpackDir "mingw64\bin\gcc.exe"))) {
                        $extractSuccess = $true
                    } else {
                        Remove-Item -Path $unpackDir -Recurse -Force -ErrorAction SilentlyContinue
                    }
                } catch {
                    Write-Warning "tar.exe 解压 7z 不支持，尝试独立 7za 工具..."
                }
            }
        }

        # 方案 C: 下载独立单文件 7za 便携解压工具
        if (-not $extractSuccess) {
            Write-Host ">>> 正在获取独立轻量 7za 解压工具..." -ForegroundColor Yellow
            $sevenZaZip = Join-Path $TempWorkDir "7za.zip"
            $sevenZaUrl = "https://www.7-zip.org/a/7za920.zip"
            try {
                Download-WithProgress -Url $sevenZaUrl -OutputFile $sevenZaZip
                $sevenZaDir = Join-Path $TempWorkDir "7za_bin"
                Expand-Archive -Path $sevenZaZip -DestinationPath $sevenZaDir -Force
                $sevenZaExe = Join-Path $sevenZaDir "7za.exe"
                if (Test-Path $sevenZaExe) {
                    $unpackDir = Join-Path $TempWorkDir "unpacked"
                    & $sevenZaExe x "$MinGw7z" "-o$unpackDir" -y | Out-Null
                    if (Test-Path (Join-Path $unpackDir "mingw64\bin\gcc.exe")) {
                        $extractSuccess = $true
                    }
                }
            } catch {
                Write-Warning "下载 7za920 失败，尝试备用 7-Zip extra 独立控制台版..."
            }
        }

        if (-not $extractSuccess) {
            throw "未能成功解压 $MinGw7z，请确认系统已安装 7-Zip 或 tar 支持。"
        }

        # 将解压后的 mingw64 内容部署到 $MinGwInstallDir
        $unpackedRoot = Join-Path $TempWorkDir "unpacked"
        $innerMingw = Join-Path $unpackedRoot "mingw64"
        if (Test-Path $innerMingw) {
            Write-Host ">>> 部署 MinGW-w64 到: $MinGwInstallDir ..." -ForegroundColor Yellow
            Copy-Item -Path "$innerMingw\*" -Destination $MinGwInstallDir -Recurse -Force
        } else {
            Copy-Item -Path "$unpackedRoot\*" -Destination $MinGwInstallDir -Recurse -Force
        }

        $minGwBin = Join-Path $MinGwInstallDir "bin"
        if (-not (Test-Path (Join-Path $minGwBin "gcc.exe"))) {
            throw "MinGW 安装校验失败，未在 $minGwBin 中找到 gcc.exe！"
        }

        Add-PathToEnvironment -NewDirectory $minGwBin
    }

} finally {
    # --------------------------------------------------------------------------
    # 3. 清理下载与临时文件
    # --------------------------------------------------------------------------
    Write-Host "--------------------------------------------------------------------------------" -ForegroundColor Cyan
    Write-Host ">>> 步骤: 清理安装临时目录: $TempWorkDir ..." -ForegroundColor Yellow
    if (Test-Path $TempWorkDir) {
        Remove-Item -Path $TempWorkDir -Recurse -Force -ErrorAction SilentlyContinue
    }
    Write-Host ">>> 临时文件已彻底清理完成！" -ForegroundColor Green
}

# --------------------------------------------------------------------------
# 4. 验证环境
# --------------------------------------------------------------------------
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ">>> 环境配置完成！当前环境验证结果:" -ForegroundColor Green
try {
    $gccVer = (& gcc --version) | Select-Object -First 1
    Write-Host "  [GCC]   $gccVer" -ForegroundColor Green
} catch {
    Write-Warning "gcc 暂未在当前终端响应，可能需要重启终端后生效。"
}
try {
    $cmakeVer = (& cmake --version) | Select-Object -First 1
    Write-Host "  [CMake] $cmakeVer" -ForegroundColor Green
} catch {
    Write-Warning "cmake 暂未在当前终端响应，可能需要重启终端后生效。"
}
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "提示: 用户永久环境变量已配置。若新开 PowerShell 窗口即可直接使用 gcc 与 cmake。" -ForegroundColor Cyan
