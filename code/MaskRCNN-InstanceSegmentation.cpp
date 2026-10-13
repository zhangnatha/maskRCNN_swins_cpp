/** @file MaskRCNN-InstanceSegmentation.cpp
 *  @brief 基于 ONNX Runtime 的 Swin-S Mask R-CNN 实例分割推理检测器实现
 *  @author zhangJA
 *  @date 2026.10.13
 *  @version 1.0
 *  @copyright zhangnatha@qq.com
 */

#if __has_include("s/vision/DeepLearning/swin-MaskRCNN/MaskRCNN-InstanceSegmentation.h")
#include "s/vision/DeepLearning/swin-MaskRCNN/MaskRCNN-InstanceSegmentation.h"
#else
#include "MaskRCNN-InstanceSegmentation.h"
#endif

#include <cmath>
#include <chrono>
#include <iomanip>
#include <map>
#include <stdexcept>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#if __has_include("s/common/Core/Log.h")
#include "s/common/Core/Log.h"
#define MASKRCNN_LOG_DEBUG(fmt, ...) s::common::Log::debug(USER_LOG, fmt, ##__VA_ARGS__)
#define MASKRCNN_LOG_WARN(fmt, ...) s::common::Log::warn(USER_LOG, fmt, ##__VA_ARGS__)
#else
#define MASKRCNN_LOG_DEBUG(fmt, ...) std::cout << "[MaskRCNN] " << fmt << std::endl
#define MASKRCNN_LOG_WARN(fmt, ...) std::cerr << "[MaskRCNN WARN] " << fmt << std::endl
#endif

namespace {

constexpr size_t kInvalidOutputIndex = std::numeric_limits<size_t>::max();
constexpr float kMaskThreshold = 0.50f;

struct RawDetection {
    float x1{0.0f};
    float y1{0.0f};
    float x2{0.0f};
    float y2{0.0f};
    float score{0.0f};
    int classId{-1};
    size_t maskIndex{0};
};

float boxIoU(const RawDetection &left, const RawDetection &right) {
    const float x1 = std::max(left.x1, right.x1);
    const float y1 = std::max(left.y1, right.y1);
    const float x2 = std::min(left.x2, right.x2);
    const float y2 = std::min(left.y2, right.y2);
    const float intersection = std::max(0.0f, x2 - x1) *
                               std::max(0.0f, y2 - y1);
    const float leftArea = std::max(0.0f, left.x2 - left.x1) *
                           std::max(0.0f, left.y2 - left.y1);
    const float rightArea = std::max(0.0f, right.x2 - right.x1) *
                            std::max(0.0f, right.y2 - right.y1);
    const float unionArea = leftArea + rightArea - intersection;
    return unionArea > 0.0f ? intersection / unionArea : 0.0f;
}

int clampCoordinate(float value, int upperBound, bool roundUp) {
    const float bounded = std::max(0.0f, std::min(value,
                                                   static_cast<float>(upperBound)));
    const int coordinate = roundUp ? static_cast<int>(std::ceil(bounded))
                                   : static_cast<int>(std::floor(bounded));
    return std::max(0, std::min(coordinate, upperBound));
}

cv::Mat ensureBGR8(const cv::Mat &image) {
    if (image.empty()) return cv::Mat();

    cv::Mat result;
    if (image.channels() == 1) {
        cv::cvtColor(image, result, cv::COLOR_GRAY2BGR);
    } else if (image.channels() == 4) {
        cv::cvtColor(image, result, cv::COLOR_BGRA2BGR);
    } else if (image.channels() == 3) {
        result = image.clone();
    } else {
        throw std::runtime_error("MaskRCNN expects a 1-, 3- or 4-channel image");
    }

    if (result.depth() != CV_8U) result.convertTo(result, CV_8UC3);
    return result;
}

double sampleMaskValue(const float *mask, int maskHeight, int maskWidth,
                       double gridX, double gridY) {
    const double imageX = gridX * maskWidth - 0.5;
    const double imageY = gridY * maskHeight - 0.5;
    const int x0 = static_cast<int>(std::floor(imageX));
    const int y0 = static_cast<int>(std::floor(imageY));
    const double alphaX = imageX - x0;
    const double alphaY = imageY - y0;
    double value = 0.0;
    for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
            const int x = x0 + dx;
            const int y = y0 + dy;
            if (x < 0 || x >= maskWidth || y < 0 || y >= maskHeight) continue;
            const double weightX = dx == 0 ? 1.0 - alphaX : alphaX;
            const double weightY = dy == 0 ? 1.0 - alphaY : alphaY;
            value += weightX * weightY * mask[y * maskWidth + x];
        }
    }
    return value;
}

cv::Mat pasteMask(const float *roiMask, int maskHeight, int maskWidth,
                  const float box[4], const cv::Size &imageSize) {
    const double x0 = box[0];
    const double y0 = box[1];
    const double x1 = box[2];
    const double y1 = box[3];
    cv::Mat output = cv::Mat::zeros(imageSize, CV_8UC1);
    const int left = std::max(static_cast<int>(std::floor(x0)) - 1, 0);
    const int top = std::max(static_cast<int>(std::floor(y0)) - 1, 0);
    const int right = std::min(static_cast<int>(std::ceil(x1)) + 1,
                               imageSize.width);
    const int bottom = std::min(static_cast<int>(std::ceil(y1)) + 1,
                                imageSize.height);
    if (right <= left || bottom <= top || x1 <= x0 || y1 <= y0) return output;

    for (int y = top; y < bottom; ++y) {
        uchar *row = output.ptr<uchar>(y);
        for (int x = left; x < right; ++x) {
            const double gridX = (x + 0.5 - x0) / (x1 - x0);
            const double gridY = (y + 0.5 - y0) / (y1 - y0);
            row[x] = sampleMaskValue(roiMask, maskHeight, maskWidth, gridX,
                                     gridY) >= kMaskThreshold
                         ? 255
                         : 0;
        }
    }
    return output;
}

cv::Scalar colorForClass(const std::vector<cv::Scalar> &colors, int classId) {
    if (colors.empty()) return cv::Scalar(200, 200, 200);
    const size_t index = classId >= 0 ? static_cast<size_t>(classId) % colors.size() : 0;
    return colors[index];
}

void blendMask(cv::Mat &image, const maskrcnn::SegmentedDetection &detection,
               const std::vector<cv::Scalar> &colors, float maskAlpha) {
    if (image.empty() || detection.mask.empty()) return;

    cv::Mat maskGray;
    if (detection.mask.channels() == 1) {
        maskGray = detection.mask;
    } else if (detection.mask.channels() == 3) {
        cv::cvtColor(detection.mask, maskGray, cv::COLOR_BGR2GRAY);
    } else {
        return;
    }

    cv::Mat maskBinary;
    if (maskGray.type() == CV_8UC1) {
        cv::compare(maskGray, cv::Scalar(127), maskBinary, cv::CMP_GE);
    } else {
        cv::Mat maskFloat;
        maskGray.convertTo(maskFloat, CV_32FC1);
        cv::compare(maskFloat, cv::Scalar(kMaskThreshold), maskBinary,
                    cv::CMP_GE);
    }

    if (maskBinary.size() != image.size()) {
        cv::resize(maskBinary, maskBinary, image.size(), 0.0, 0.0,
                   cv::INTER_NEAREST);
    }
    const cv::Scalar color = colorForClass(colors, detection.classId);
    const float alpha = std::max(0.0f, std::min(1.0f, maskAlpha));
    for (int y = 0; y < image.rows; ++y) {
        const uchar *maskRow = maskBinary.ptr<uchar>(y);
        cv::Vec3b *imageRow = image.ptr<cv::Vec3b>(y);
        for (int x = 0; x < image.cols; ++x) {
            if (maskRow[x] == 0) continue;
            for (int channel = 0; channel < 3; ++channel) {
                imageRow[x][channel] = cv::saturate_cast<uchar>(
                    (1.0f - alpha) * imageRow[x][channel] +
                    alpha * static_cast<float>(color[channel]));
            }
        }
    }
}

} // namespace

bool MaskRCNNDetector::isFloatType(ONNXTensorElementDataType type) {
    return type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

bool MaskRCNNDetector::isInt64Type(ONNXTensorElementDataType type) {
    return type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
}

void MaskRCNNDetector::printSegmentationResults(
    const std::vector<maskrcnn::SegmentedDetection> &results) const {
    std::cout << "[MaskRCNN] Segmentation Results (" << results.size()
              << " objects):\n";
    /*
    for (size_t i = 0; i < results.size(); ++i) {
        const auto &detection = results[i];
        std::cout << "  Object " << i + 1 << ": Class ID: " << detection.classId
                  << " Confidence: " << detection.conf
                  << " Bounding Box: [x: " << detection.box.x
                  << ", y: " << detection.box.y
                  << ", w: " << detection.box.width
                  << ", h: " << detection.box.height
                  << "] Mask: " << detection.mask.cols << "x"
                  << detection.mask.rows << " (area: "
                  << cv::countNonZero(detection.mask) << " pixels)\n";
    }
    */
}

MaskRCNNDetector::MaskRCNNDetector(const std::string &modelPath,
                                    const std::string &labelsPath,
                                    bool useGPU)
    : _env(ORT_LOGGING_LEVEL_WARNING, "MaskRCNN") {
    const unsigned int hardwareThreads = std::thread::hardware_concurrency();
    _sessionOptions.SetIntraOpNumThreads(
        std::max(1, std::min(6, static_cast<int>(hardwareThreads))));
    _sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    const std::vector<std::string> providers = Ort::GetAvailableProviders();
    if (useGPU && std::find(providers.begin(), providers.end(),
                            "CUDAExecutionProvider") != providers.end()) {
        OrtCUDAProviderOptions cudaOptions{};
        cudaOptions.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
        cudaOptions.gpu_mem_limit = 8ULL * 1024ULL * 1024ULL * 1024ULL;
        cudaOptions.arena_extend_strategy = 1;
        cudaOptions.do_copy_in_default_stream = 1;
        _sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
        std::cout << "[INFO] Using GPU (CUDA) for MaskRCNN inference.\n";
    } else {
        if (useGPU) {
            MASKRCNN_LOG_WARN("%s", "CUDAExecutionProvider is unavailable; falling back to CPU");
        }
        std::cout << "[INFO] Using CPU for MaskRCNN inference.\n";
    }

#ifdef _WIN32
    std::wstring wModelPath;
    if (!modelPath.empty()) {
        int wlen = MultiByteToWideChar(CP_UTF8, 0, modelPath.c_str(), -1, nullptr, 0);
        if (wlen > 0) {
            wModelPath.resize(static_cast<size_t>(wlen - 1));
            MultiByteToWideChar(CP_UTF8, 0, modelPath.c_str(), -1, &wModelPath[0], wlen);
        }
    }
    _session = Ort::Session(_env, wModelPath.c_str(), _sessionOptions);
#else
    _session = Ort::Session(_env, modelPath.c_str(), _sessionOptions);
#endif

    _numInputNodes = _session.GetInputCount();
    _numOutputNodes = _session.GetOutputCount();
    if (_numInputNodes != 1) {
        throw std::runtime_error("MaskRCNN model must have exactly one input");
    }

    Ort::AllocatorWithDefaultOptions allocator;

    {
        auto inputName = _session.GetInputNameAllocated(0, allocator);
        _inputNameAllocs.emplace_back(std::move(inputName));
        _inputNames.push_back(_inputNameAllocs.back().get());

        const auto typeInfo = _session.GetInputTypeInfo(0);
        const auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
        const auto shape = tensorInfo.GetShape();
        if (!isFloatType(tensorInfo.GetElementType()) || shape.size() != 3) {
            throw std::runtime_error("MaskRCNN input must be a float tensor [3,H,W]");
        }
        if (shape[0] != 3 && shape[0] != -1) {
            throw std::runtime_error("MaskRCNN input channel count must be 3");
        }

        _inputTensorShape = shape;
        if (_inputTensorShape[0] <= 0) _inputTensorShape[0] = 3;
        if (_inputTensorShape[1] <= 0) _inputTensorShape[1] = 480;
        if (_inputTensorShape[2] <= 0) _inputTensorShape[2] = 640;
        _inputImageShape = cv::Size(static_cast<int>(_inputTensorShape[2]),
                                    static_cast<int>(_inputTensorShape[1]));
    }

    for (size_t index = 0; index < _numOutputNodes; ++index) {
        auto outputName = _session.GetOutputNameAllocated(index, allocator);
        _outputNameAllocs.emplace_back(std::move(outputName));
        _outputNames.push_back(_outputNameAllocs.back().get());

        const auto typeInfo = _session.GetOutputTypeInfo(index);
        const auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
        const auto shape = tensorInfo.GetShape();
        const auto type = tensorInfo.GetElementType();

        // 导出的模型还包含一个 int64 [2] 的原始图像尺寸输出。
        // 通过类型和维度识别四个语义输出节点，并在 Run() 之后校验具体形状。
        if (isFloatType(type) && shape.size() == 2 &&
            (shape.size() < 2 || shape[1] == 4 || shape[1] == -1)) {
            _outputIndices.boxes = index;
        } else if (isInt64Type(type) && shape.size() == 1 && shape[0] != 2) {
            _outputIndices.labels = index;
        } else if (isFloatType(type) && shape.size() == 4) {
            _outputIndices.masks = index;
        } else if (isFloatType(type) && shape.size() == 1) {
            _outputIndices.scores = index;
        }
    }

    if (_outputIndices.boxes == kInvalidOutputIndex ||
        _outputIndices.labels == kInvalidOutputIndex ||
        _outputIndices.masks == kInvalidOutputIndex ||
        _outputIndices.scores == kInvalidOutputIndex) {
        throw std::runtime_error(
            "MaskRCNN model outputs do not match boxes/labels/masks/scores signature");
    }

    _classNames = maskrcnn::utils::getClassNames(labelsPath);
    _classColors = maskrcnn::utils::generateHSVColors(static_cast<int>(_classNames.size()));

    std::cout << "[INFO] MaskRCNN loaded: " << modelPath << std::endl
              << "      Input Shape: " << _inputImageShape << std::endl
              << "      Output Count: " << _numOutputNodes << std::endl
              << "      Class Count: " << _classNames.size() << std::endl;
}

cv::Mat MaskRCNNDetector::preprocess(const cv::Mat &image, float *&blobPtr,
                                     std::vector<int64_t> &inputTensorShape) {
    if (image.empty()) throw std::runtime_error("input image is empty");

    cv::Mat resized;
    cv::resize(image, resized, _inputImageShape, 0.0, 0.0, cv::INTER_LINEAR);
    if (resized.type() != CV_8UC3) {
        throw std::runtime_error("MaskRCNN preprocessing expects CV_8UC3 image");
    }

    inputTensorShape = _inputTensorShape;
    const size_t planeSize = static_cast<size_t>(resized.rows) *
                             static_cast<size_t>(resized.cols);
    const size_t inputSize = 3 * planeSize;
    blobPtr = new float[inputSize];

    // export_swin.py 使用 BGR、CHW 输入，保持 OpenCV 通道顺序且不除以 255。
    for (int row = 0; row < resized.rows; ++row) {
        const auto *pixelRow = resized.ptr<cv::Vec3b>(row);
        for (int col = 0; col < resized.cols; ++col) {
            const size_t offset = static_cast<size_t>(row) * resized.cols + col;
            blobPtr[offset] = static_cast<float>(pixelRow[col][0]);
            blobPtr[planeSize + offset] = static_cast<float>(pixelRow[col][1]);
            blobPtr[2 * planeSize + offset] = static_cast<float>(pixelRow[col][2]);
        }
    }
    return resized;
}

std::vector<maskrcnn::SegmentedDetection> MaskRCNNDetector::postprocess(
    const cv::Size &origSize, const cv::Size &modelInputSize,
    const std::vector<Ort::Value> &outputs, float confThreshold,
    float iouThreshold) {
    SegmentOptions options;
    options.scoreThreshold = confThreshold;
    options.iouThreshold = iouThreshold;
    return postprocess(origSize, modelInputSize, outputs, options);
}

std::vector<maskrcnn::SegmentedDetection> MaskRCNNDetector::postprocess(
    const cv::Size &origSize, const cv::Size &modelInputSize,
    const std::vector<Ort::Value> &outputs, const SegmentOptions &options) {
    std::vector<maskrcnn::SegmentedDetection> results;
    if (outputs.size() != _numOutputNodes) {
        throw std::runtime_error("MaskRCNN output count is inconsistent with the model");
    }

    const auto &boxesValue = outputs[_outputIndices.boxes];
    const auto &labelsValue = outputs[_outputIndices.labels];
    const auto &masksValue = outputs[_outputIndices.masks];
    const auto &scoresValue = outputs[_outputIndices.scores];

    const auto boxesInfo = boxesValue.GetTensorTypeAndShapeInfo();
    const auto boxesShape = boxesInfo.GetShape();
    if (!isFloatType(boxesInfo.GetElementType()) || boxesShape.size() != 2 ||
        boxesShape[1] != 4) {
        throw std::runtime_error("runtime boxes output has invalid shape");
    }

    const size_t count = static_cast<size_t>(std::max<int64_t>(0, boxesShape[0]));
    const auto labelsInfo = labelsValue.GetTensorTypeAndShapeInfo();
    const auto scoresInfo = scoresValue.GetTensorTypeAndShapeInfo();
    const auto masksInfo = masksValue.GetTensorTypeAndShapeInfo();
    const auto masksShape = masksInfo.GetShape();

    if (!isInt64Type(labelsInfo.GetElementType()) || labelsInfo.GetShape().size() != 1 ||
        labelsInfo.GetElementCount() < count ||
        !isFloatType(scoresInfo.GetElementType()) || scoresInfo.GetShape().size() != 1 ||
        scoresInfo.GetElementCount() < count || !isFloatType(masksInfo.GetElementType()) ||
        masksShape.size() != 4 || masksShape[0] != static_cast<int64_t>(count) ||
        masksShape[1] != 1 || masksShape[2] <= 0 || masksShape[3] <= 0) {
        throw std::runtime_error("runtime labels/masks/scores outputs have invalid shapes");
    }
    if (count == 0) return results;

    const int maskHeight = static_cast<int>(masksShape[2]);
    const int maskWidth = static_cast<int>(masksShape[3]);
    const size_t maskStride = masksInfo.GetElementCount() / count;
    if (maskStride < static_cast<size_t>(maskHeight) * maskWidth) {
        throw std::runtime_error("runtime mask output has insufficient elements");
    }

    const float *boxes = boxesValue.GetTensorData<float>();
    const int64_t *labels = labelsValue.GetTensorData<int64_t>();
    const float *masks = masksValue.GetTensorData<float>();
    const float *scores = scoresValue.GetTensorData<float>();
    const float scaleX = static_cast<float>(origSize.width) / modelInputSize.width;
    const float scaleY = static_cast<float>(origSize.height) / modelInputSize.height;

    std::vector<RawDetection> candidates;
    candidates.reserve(count);
    for (size_t index = 0; index < count; ++index) {
        const int64_t rawClassId = labels[index];
        const float classThreshold =
            rawClassId >= 0 &&
                    static_cast<size_t>(rawClassId) < options.classScoreThresholds.size()
                ? options.classScoreThresholds[static_cast<size_t>(rawClassId)]
                : options.scoreThreshold;
        if (labels[index] < 0 ||
            labels[index] >= static_cast<int64_t>(_classNames.size()) ||
            scores[index] < classThreshold) {
            continue;
        }

        RawDetection candidate;
        candidate.x1 = std::max(0.0f, std::min(boxes[index * 4],
                                                static_cast<float>(modelInputSize.width)));
        candidate.y1 = std::max(0.0f, std::min(boxes[index * 4 + 1],
                                                static_cast<float>(modelInputSize.height)));
        candidate.x2 = std::max(0.0f, std::min(boxes[index * 4 + 2],
                                                static_cast<float>(modelInputSize.width)));
        candidate.y2 = std::max(0.0f, std::min(boxes[index * 4 + 3],
                                                static_cast<float>(modelInputSize.height)));
        if (candidate.x2 <= candidate.x1 || candidate.y2 <= candidate.y1) continue;

        candidate.score = scores[index];
        candidate.classId = static_cast<int>(labels[index]);
        candidate.maskIndex = index;
        candidates.emplace_back(candidate);
    }
    if (candidates.empty()) return results;

    // 按类别执行贪心 NMS，再按置信度排序并应用整图最大检测数。
    std::map<int, std::vector<int>> groupedCandidates;
    for (size_t index = 0; index < candidates.size(); ++index) {
        groupedCandidates[candidates[index].classId].push_back(
            static_cast<int>(index));
    }

    std::vector<int> selectedIndices;
    for (auto &group : groupedCandidates) {
        auto &indices = group.second;
        std::sort(indices.begin(), indices.end(), [&candidates](int left, int right) {
            return candidates[left].score > candidates[right].score;
        });

        std::vector<int> kept;
        kept.reserve(indices.size());
        const float classIouThreshold =
            group.first >= 0 &&
                    static_cast<size_t>(group.first) < options.classIouThresholds.size()
                ? options.classIouThresholds[static_cast<size_t>(group.first)]
                : options.iouThreshold;
        for (const int candidateIndex : indices) {
            bool suppressed = false;
            for (const int keptIndex : kept) {
                if (boxIoU(candidates[candidateIndex], candidates[keptIndex]) >
                    classIouThreshold) {
                    suppressed = true;
                    break;
                }
            }
            if (!suppressed) kept.push_back(candidateIndex);
        }

        selectedIndices.insert(selectedIndices.end(), kept.begin(), kept.end());
    }
    std::sort(selectedIndices.begin(), selectedIndices.end(), [&candidates](int left, int right) {
        return candidates[left].score > candidates[right].score;
    });

    if (selectedIndices.size() > options.maxDetections) {
        selectedIndices.resize(options.maxDetections);
    }
    results.reserve(selectedIndices.size());
    for (const int candidateIndex : selectedIndices) {
        const RawDetection &candidate = candidates[candidateIndex];
        const float scaledBox[4] = {candidate.x1 * scaleX, candidate.y1 * scaleY,
                                     candidate.x2 * scaleX, candidate.y2 * scaleY};
        const int left = clampCoordinate(scaledBox[0], origSize.width, false);
        const int top = clampCoordinate(scaledBox[1], origSize.height, false);
        const int right = clampCoordinate(scaledBox[2], origSize.width, true);
        const int bottom = clampCoordinate(scaledBox[3], origSize.height, true);
        if (right <= left || bottom <= top) continue;

        maskrcnn::SegmentedDetection detection;
        detection.box = maskrcnn::SegBoundingBox(left, top, right - left,
                                                 bottom - top);
        detection.conf = candidate.score;
        detection.classId = candidate.classId;

        const float *maskData = masks + candidate.maskIndex * maskStride;
        detection.mask = pasteMask(maskData, maskHeight, maskWidth, scaledBox,
                                   origSize);
        results.emplace_back(std::move(detection));
    }
    return results;
}

void MaskRCNNDetector::drawSegmentationsAndBoxes(
    cv::Mat &image, const std::vector<maskrcnn::SegmentedDetection> &results,
    float maskAlpha) const {
    for (const auto &detection : results) {
        blendMask(image, detection, _classColors, maskAlpha);

        const cv::Scalar color = colorForClass(_classColors, detection.classId);
        cv::rectangle(image, cv::Point(detection.box.x, detection.box.y),
                      cv::Point(detection.box.x + detection.box.width,
                                detection.box.y + detection.box.height),
                      color, 2);

        const std::string className =
            detection.classId >= 0 &&
                    static_cast<size_t>(detection.classId) < _classNames.size()
                ? _classNames[detection.classId]
                : "Class " + std::to_string(detection.classId);

        // 格式化置信度为保留 3 位小数
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(3) << detection.conf;
        const std::string label = className + " " + ss.str();

        int baseLine = 0;
        const cv::Size labelSize = cv::getTextSize(
            label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        const int top = std::max(detection.box.y, labelSize.height + 5);

        // 绘制文字背景框与文字
        cv::rectangle(image, cv::Point(detection.box.x,
                                       top - labelSize.height - 5),
                      cv::Point(detection.box.x + labelSize.width + 5, top),
                      color, cv::FILLED);
        cv::putText(image, label, cv::Point(detection.box.x + 2, top - 2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
}

void MaskRCNNDetector::drawSegmentations(
    cv::Mat &image, const std::vector<maskrcnn::SegmentedDetection> &results,
    float maskAlpha) const {
    for (const auto &detection : results) {
        blendMask(image, detection, _classColors, maskAlpha);
    }
}

std::vector<maskrcnn::SegmentedDetection> MaskRCNNDetector::segment(
    const cv::Mat &image, float confThreshold, float iouThreshold) {
    SegmentOptions options;
    options.scoreThreshold = confThreshold;
    options.iouThreshold = iouThreshold;
    return segment(image, options);
}

std::vector<maskrcnn::SegmentedDetection> MaskRCNNDetector::segment(
    const cv::Mat &image, const SegmentOptions &options) {
    if (image.empty()) throw std::runtime_error("input image is empty");
    if (options.scoreThreshold < 0.0f || options.scoreThreshold > 1.0f) {
        throw std::invalid_argument("confidence threshold must be in [0, 1]");
    }
    if (options.iouThreshold < 0.0f || options.iouThreshold > 1.0f) {
        throw std::invalid_argument("IoU threshold must be in [0, 1]");
    }
    if (options.maxDetections == 0) {
        throw std::invalid_argument("maximum detections must be positive");
    }
    for (const float value : options.classScoreThresholds) {
        if (value < 0.0f || value > 1.0f) {
            throw std::invalid_argument("class confidence threshold must be in [0, 1]");
        }
    }
    for (const float value : options.classIouThresholds) {
        if (value < 0.0f || value > 1.0f) {
            throw std::invalid_argument("class IoU threshold must be in [0, 1]");
        }
    }

    const cv::Mat src = ensureBGR8(image);
    float *blobPtr = nullptr;
    std::vector<int64_t> inputShape;
    const cv::Mat resized = preprocess(src, blobPtr, inputShape);
    std::unique_ptr<float[]> blobGuard(blobPtr);
    const size_t inputSize = maskrcnn::utils::vectorProduct(inputShape);
    if (inputSize == 0) throw std::runtime_error("invalid MaskRCNN input shape");

    std::vector<float> inputValues(blobPtr, blobPtr + inputSize);
    const auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,
                                                       OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, inputValues.data(), inputValues.size(), inputShape.data(),
        inputShape.size());

    const auto inferenceStart = std::chrono::steady_clock::now();
    std::vector<Ort::Value> outputs = _session.Run(
        Ort::RunOptions{nullptr}, _inputNames.data(), &inputTensor,
        _numInputNodes, _outputNames.data(), _numOutputNodes);
    const std::chrono::duration<double, std::milli> inferenceTime =
        std::chrono::steady_clock::now() - inferenceStart;
    std::cout << "[INFO] Inference time: " << inferenceTime.count() << " ms\n";

    auto results = postprocess(src.size(), resized.size(), outputs, options);

#if __has_include("s/vision/Vision2D/Utility/UtilityBuffer.h")
    if (_roi2d.rect().width != src.cols || _roi2d.rect().height != src.rows) {
        results = filterDetectionsByROI(results, _roi2d);
    }
#endif
    return results;
}

#if __has_include("s/vision/Vision2D/Utility/UtilityBuffer.h")
std::vector<maskrcnn::SegmentedDetection> MaskRCNNDetector::filterDetectionsByROI(
    const std::vector<maskrcnn::SegmentedDetection> &results,
    s::vision::ROI2DRect &roi) {
    const auto rect = roi.rect();
    const int roiX = std::max(0, static_cast<int>(rect.center.x() - rect.width / 2.0));
    const int roiY = std::max(0, static_cast<int>(rect.center.y() - rect.height / 2.0));
    const int roiW = static_cast<int>(rect.width);
    const int roiH = static_cast<int>(rect.height);
    const cv::Rect cvRoi(roiX, roiY, roiW, roiH);
    if (cvRoi.width <= 0 || cvRoi.height <= 0) return results;

    std::vector<maskrcnn::SegmentedDetection> filtered;
    for (const auto &detection : results) {
        const cv::Rect detectionBox(detection.box.x, detection.box.y,
                                    detection.box.width, detection.box.height);
        const cv::Rect intersection = detectionBox & cvRoi;
        if (intersection.area() <= 0) continue;

        maskrcnn::SegmentedDetection clipped = detection;
        if (!clipped.mask.empty()) {
            cv::Mat clippedMask = cv::Mat::zeros(clipped.mask.size(),
                                                 clipped.mask.type());
            const cv::Rect imageRect(0, 0, clipped.mask.cols, clipped.mask.rows);
            const cv::Rect maskIntersection = cvRoi & imageRect;
            if (maskIntersection.area() > 0) {
                clipped.mask(maskIntersection).copyTo(
                    clippedMask(maskIntersection));
            }
            clipped.mask = clippedMask;
            if (cv::countNonZero(clipped.mask) == 0) continue;

            const cv::Rect tightBox = cv::boundingRect(clipped.mask);
            clipped.box = maskrcnn::SegBoundingBox(tightBox.x, tightBox.y,
                                                   tightBox.width, tightBox.height);
        } else {
            clipped.box = maskrcnn::SegBoundingBox(intersection.x, intersection.y,
                                                   intersection.width,
                                                   intersection.height);
        }
        filtered.emplace_back(std::move(clipped));
    }
    return filtered;
}
#endif
