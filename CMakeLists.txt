cmake_minimum_required(VERSION 3.10)
project(maskrcnn_swins_inference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 在 UNIX 平台开启位置无关代码 (PIC) 与线程库支持，并排除 conda 环境路径避免系统 ABI 冲突
if(UNIX)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    find_package(Threads REQUIRED)
    if(DEFINED ENV{CONDA_PREFIX})
        list(APPEND CMAKE_IGNORE_PATH "$ENV{CONDA_PREFIX}")
    endif()
    if(DEFINED ENV{HOME})
        list(APPEND CMAKE_IGNORE_PATH "$ENV{HOME}/anaconda3" "$ENV{HOME}/miniconda3")
    endif()
endif()

# ==============================================================================
# OpenCV 依赖配置
# ==============================================================================
if(NOT OpenCV_DIR)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/opencv/lib/cmake/opencv4")
        set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/opencv/lib/cmake/opencv4" CACHE PATH "OpenCV directory")
    elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/opencv/x64/mingw/lib")
        set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/opencv/x64/mingw/lib" CACHE PATH "OpenCV directory")
    elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/opencv/x64/vc16/lib")
        set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/opencv/x64/vc16/lib" CACHE PATH "OpenCV directory")
    elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/opencv/build")
        set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/opencv/build" CACHE PATH "OpenCV directory")
    elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/opencv")
        set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/opencv" CACHE PATH "OpenCV directory")
    endif()
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
message(STATUS "OpenCV found: ${OpenCV_VERSION}")
message(STATUS "  Include dirs: ${OpenCV_INCLUDE_DIRS}")

# ==============================================================================
# ONNX Runtime 依赖配置
# ==============================================================================
if(NOT ONNXRUNTIME_DIR)
    if(DEFINED ENV{ONNXRUNTIME_DIR})
        set(ONNXRUNTIME_DIR "$ENV{ONNXRUNTIME_DIR}")
    elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/onnxruntime-cuda12-1.20.0")
        set(ONNXRUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/onnxruntime-cuda12-1.20.0")
    elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/onnxruntime-cuda11-1.20.0")
        set(ONNXRUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/onnxruntime-cuda11-1.20.0")
    elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/onnxruntime-linux-x64-1.20.0")
        set(ONNXRUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/onnxruntime-linux-x64-1.20.0")
    elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/onnxruntime-win-x64-1.20.0")
        set(ONNXRUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/onnxruntime-win-x64-1.20.0")
    elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/onnxruntime-win-x64-gpu-1.20.0")
        set(ONNXRUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/onnxruntime-win-x64-gpu-1.20.0")
    elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/onnxruntime")
        set(ONNXRUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/onnxruntime")
    endif()
endif()

if(ONNXRUNTIME_DIR)
    message(STATUS "Using ONNX Runtime from: ${ONNXRUNTIME_DIR}")
    set(ORT_INCLUDE_DIR "${ONNXRUNTIME_DIR}/include")
    if(WIN32)
        find_library(ORT_LIB NAMES onnxruntime libonnxruntime
            PATHS "${ONNXRUNTIME_DIR}/lib" "${ONNXRUNTIME_DIR}/lib/x64" "${ONNXRUNTIME_DIR}/bin" NO_DEFAULT_PATH)
        find_file(ORT_DLL NAMES onnxruntime.dll
            PATHS "${ONNXRUNTIME_DIR}/lib" "${ONNXRUNTIME_DIR}/bin" "${ONNXRUNTIME_DIR}/lib/x64" NO_DEFAULT_PATH)
    else()
        find_library(ORT_LIB NAMES onnxruntime
            PATHS "${ONNXRUNTIME_DIR}/lib" NO_DEFAULT_PATH)
    endif()
else()
    find_path(ORT_INCLUDE_DIR NAMES onnxruntime_cxx_api.h PATH_SUFFIXES onnxruntime)
    find_library(ORT_LIB NAMES onnxruntime libonnxruntime)
endif()

if(NOT ORT_INCLUDE_DIR OR NOT ORT_LIB)
    message(FATAL_ERROR "ONNX Runtime not found! Please set -DONNXRUNTIME_DIR=<path_to_onnxruntime>")
endif()
message(STATUS "ONNX Runtime include: ${ORT_INCLUDE_DIR}")
message(STATUS "ONNX Runtime library: ${ORT_LIB}")

# ==============================================================================
# 可执行目标配置
# ==============================================================================
add_executable(test_model_inference
    code/MaskRCNN-InstanceSegmentation.cpp
    code/test_model_inference.cpp
)

target_include_directories(test_model_inference PRIVATE
    code
    ${OpenCV_INCLUDE_DIRS}
    ${ORT_INCLUDE_DIR}
)

target_link_libraries(test_model_inference PRIVATE
    ${OpenCV_LIBS}
    ${ORT_LIB}
)

if(UNIX)
    target_link_libraries(test_model_inference PRIVATE Threads::Threads)
    # 设置 Linux 下纯相对路径的 RPATH ($ORIGIN)，不依赖任何绝对路径，实现完全跨机器可移植
    set_target_properties(test_model_inference PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH_USE_LINK_PATH FALSE
        INSTALL_RPATH "$ORIGIN:$ORIGIN/lib:$ORIGIN/../lib:$ORIGIN/../3rdparty/opencv/lib:$ORIGIN/../3rdparty/onnxruntime/lib:$ORIGIN/../3rdparty/onnxruntime-cuda11-1.20.0/lib:$ORIGIN/../3rdparty/onnxruntime-cuda12-1.20.0/lib:$ORIGIN/../3rdparty/onnxruntime-linux-x64-1.20.0/lib:$ORIGIN/../3rdparty/cuda11-runtime/lib:$ORIGIN/../3rdparty/cuda-runtime/lib:$ORIGIN/3rdparty/opencv/lib:$ORIGIN/3rdparty/onnxruntime/lib:$ORIGIN/3rdparty/onnxruntime-cuda11-1.20.0/lib:$ORIGIN/3rdparty/cuda11-runtime/lib:$ORIGIN/3rdparty/cuda-runtime/lib"
    )
    # 二进制中的 DT_RUNPATH 不会被通过 dlopen 延迟加载的动态库继承；
    # 通过禁用 new-dtags 使用传统 DT_RPATH，确保 CUDA provider 可以顺利解析 CUDA 11/cuDNN 8 依赖库。
    set_property(TARGET test_model_inference APPEND_STRING PROPERTY
        LINK_FLAGS " -Wl,--disable-new-dtags")
endif()

if(WIN32)
    if(MSVC)
        target_compile_options(test_model_inference PRIVATE /W3 /DNOMINMAX /DWIN32_LEAN_AND_MEAN /utf-8)
    else()
        target_compile_options(test_model_inference PRIVATE -Wall -DNOMINMAX -DWIN32_LEAN_AND_MEAN)
        target_compile_definitions(test_model_inference PRIVATE _stdcall=__stdcall)
    endif()

    # Windows 下构建完成后自动将 ONNX Runtime DLL 复制到目标目录
    if(ORT_DLL)
        add_custom_command(TARGET test_model_inference POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ORT_DLL}" "$<TARGET_FILE_DIR:test_model_inference>"
            COMMENT "Copying onnxruntime.dll to target directory"
        )
    endif()

    # 若存在 OpenCV DLL 则自动复制到目标目录
    file(GLOB CV_DLLS
        "${OpenCV_DIR}/bin/*.dll"
        "${OpenCV_DIR}/../bin/*.dll"
        "${OpenCV_DIR}/x64/mingw/bin/*.dll"
        "${OpenCV_DIR}/x64/vc*/bin/*.dll"
        "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/opencv/bin/*.dll"
        "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/opencv/x64/mingw/bin/*.dll"
    )
    foreach(dll_file ${CV_DLLS})
        add_custom_command(TARGET test_model_inference POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${dll_file}" "$<TARGET_FILE_DIR:test_model_inference>"
            COMMENT "Copying OpenCV DLL (${dll_file}) to target directory"
        )
    endforeach()
endif()
