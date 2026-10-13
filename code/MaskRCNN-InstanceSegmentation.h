/** @file MaskRCNN-InstanceSegmentation.h
 *  @brief 基于 ONNX Runtime 的 Swin-S Mask R-CNN 实例分割推理检测器头文件
 *  @author zhangJA
 *  @date 2026.10.13
 *  @version 1.0
 *  @copyright zhangnatha@qq.com
 */

#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#if __has_include("s/vision/DeepLearning/MaskRCNN-Util.h")
#include "s/vision/DeepLearning/MaskRCNN-Util.h"
#else
#include "MaskRCNN-Util.h"
#endif

/**
 * @brief MaskRCNNDetector 负责加载 Swin-S Mask R-CNN ONNX 模型、预处理图像、
 *        运行推理和将结果转换为 DU/YOLO26 风格的实例分割结果。
 *
 * 模型契约：输入为 [3, 480, 640] 的 BGR/CHW、未归一化 float tensor；
 * 输出包含 boxes [N,4]、labels [N]、scores [N] 和 masks [N,1,28,28]。
 */
class MaskRCNNDetector {
public:
    // 推理参数与独立 C++ 推理程序保持一致。
    struct SegmentOptions {
        float scoreThreshold{0.50f};
        float iouThreshold{0.50f};
        std::vector<float> classScoreThresholds;
        std::vector<float> classIouThresholds;
        size_t maxDetections{1000};
    };

    MaskRCNNDetector(const std::string &modelPath,
                     const std::string &labelsPath,
                     bool useGPU = false);

    std::vector<maskrcnn::SegmentedDetection>
    segment(const cv::Mat &image, float confThreshold = 0.50f,
            float iouThreshold = 0.50f);

    std::vector<maskrcnn::SegmentedDetection>
    segment(const cv::Mat &image, const SegmentOptions &options);

    void drawSegmentationsAndBoxes(
        cv::Mat &image,
        const std::vector<maskrcnn::SegmentedDetection> &results,
        float maskAlpha = 0.45f) const;

    void drawSegmentations(
        cv::Mat &image,
        const std::vector<maskrcnn::SegmentedDetection> &results,
        float maskAlpha = 0.45f) const;

#if __has_include("s/vision/Vision2D/Utility/UtilityBuffer.h")
    void setROI(s::vision::ROI2DRect roi) { _roi2d = roi; }
#endif

    const std::vector<std::string> &getClassNames() const { return _classNames; }
    const std::vector<cv::Scalar> &getClassColors() const { return _classColors; }
    void printSegmentationResults(
        const std::vector<maskrcnn::SegmentedDetection> &results) const;

private:
    struct ModelOutputs {
        size_t boxes{std::numeric_limits<size_t>::max()};
        size_t labels{std::numeric_limits<size_t>::max()};
        size_t masks{std::numeric_limits<size_t>::max()};
        size_t scores{std::numeric_limits<size_t>::max()};
    };

    Ort::Env _env;
    Ort::SessionOptions _sessionOptions;
    Ort::Session _session{nullptr};

    cv::Size _inputImageShape{640, 480};
    std::vector<int64_t> _inputTensorShape{3, 480, 640};

    std::vector<Ort::AllocatedStringPtr> _inputNameAllocs;
    std::vector<const char *> _inputNames;
    std::vector<Ort::AllocatedStringPtr> _outputNameAllocs;
    std::vector<const char *> _outputNames;

    size_t _numInputNodes{0};
    size_t _numOutputNodes{0};
    ModelOutputs _outputIndices;

    std::vector<std::string> _classNames;
    std::vector<cv::Scalar> _classColors;

    cv::Mat preprocess(const cv::Mat &image, float *&blobPtr,
                       std::vector<int64_t> &inputTensorShape);

    std::vector<maskrcnn::SegmentedDetection>
    postprocess(const cv::Size &origSize, const cv::Size &modelInputSize,
                const std::vector<Ort::Value> &outputs, float confThreshold,
                float iouThreshold);

    std::vector<maskrcnn::SegmentedDetection>
    postprocess(const cv::Size &origSize, const cv::Size &modelInputSize,
                const std::vector<Ort::Value> &outputs,
                const SegmentOptions &options);

    static bool isFloatType(ONNXTensorElementDataType type);
    static bool isInt64Type(ONNXTensorElementDataType type);

#if __has_include("s/vision/Vision2D/Utility/UtilityBuffer.h")
    s::vision::ROI2DRect _roi2d;

    std::vector<maskrcnn::SegmentedDetection> filterDetectionsByROI(
        const std::vector<maskrcnn::SegmentedDetection> &results,
        s::vision::ROI2DRect &roi);
#endif
};
