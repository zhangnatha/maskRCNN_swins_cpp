/** @file AlgMaskRCNN-InstanceSegm.cpp
 *  @brief Mask R-CNN 实例分割算法插件实现
 *  @author zhangJA
 *  @date 2026.10.13
 *  @version 1.0
 *  @copyright zhangnatha@qq.com
 */

#include "AlgMaskRCNN-InstanceSegm.h"

#if __has_include("salgorithm/Common/AlgoritmPlugin.h")
#include "common/SettingType.h"
#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/istreamwrapper.h"
#include "rapidjson/reader.h"
#include "rapidjson/stringbuffer.h"
#include "s/vision/Vision2D/Utility/UtilityImage.h"
#include <algorithm>
#include <fstream>
#include <istream>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <ctime>
#include "salgorithm/Common/Debug2D.h"

using namespace sapp;
using namespace s::common;
using namespace s::vision;

#define USE_GPU 1

namespace {

bool fileExists(const std::string &path) {
    struct stat buffer;
    return !path.empty() && stat(path.c_str(), &buffer) == 0;
}

std::string getClassName(const MaskRCNNDetector &detector, int classId) {
    const auto &names = detector.getClassNames();
    if (classId >= 0 && static_cast<size_t>(classId) < names.size()) {
        return names[static_cast<size_t>(classId)];
    }
    return "Class " + std::to_string(classId);
}

} // namespace

AlgMaskRCNNInstanceSegm::AlgMaskRCNNInstanceSegm()
    : AlgoritmBlockPlugin("AlgMaskRCNNInstanceSegm") {
    _lastIsGPU = false;
    _oldModelMTime = 0;
    _oldModelPath = "";
    _oldLabelsPath = "";
    _firstRUN = true;
}

bool AlgMaskRCNNInstanceSegm::checkValued() {
    return true;
}

AlgMaskRCNNInstanceSegm::~AlgMaskRCNNInstanceSegm() {
}

bool AlgMaskRCNNInstanceSegm::handleFailed() {
    return true;
}

bool AlgMaskRCNNInstanceSegm::getting_input_images() {
    auto in = sapp::BlockTask::getInputs();
    if (!in->has("InputDetectImage")) {
        S_THROW("Entered illegal code area!");
        return false;
    }

    if (in->s::common::PropertyList::findPropertyBase("InputDetectImage")
            ->getType().getTypeName() ==
        s::common::traitTypeName<s::vision::ImageColor>()) {
        s::vision::ImageColor image = in->get<s::vision::ImageColor>(
            "InputDetectImage");
        if (image.isEmpty()) {
            S_THROW("The image is empty!");
            return false;
        }
        s::vision::ImageUtility<unsigned char, 3>::duco2cv(image,
                                                            inputSrcimage);
    } else if (
        in->s::common::PropertyList::findPropertyBase("InputDetectImage")
                ->getType().getTypeName() ==
        s::common::traitTypeName<s::vision::ImageGray>()) {
        s::vision::ImageGray image = in->get<s::vision::ImageGray>(
            "InputDetectImage");
        if (image.isEmpty()) {
            S_THROW("The image is empty!");
            return false;
        }
        s::vision::ImageUtility<unsigned char, 1>::duco2cv(image,
                                                            inputSrcimage);
    } else {
        S_THROW("No finding the 'InputDetectImage' in the type of 'gray or color'!");
        return false;
    }
    return true;
}

static cv::Mat removeTopProportionOfForeground(const cv::Mat &inputMask,
                                               double proportion = 0.5) {
    if (inputMask.empty() || inputMask.channels() != 1 ||
        inputMask.type() != CV_8UC1 || proportion <= 0.0 || proportion >= 1.0) {
        return inputMask.clone();
    }
    if (inputMask.rows < 2) return inputMask.clone();

    cv::Mat outputMask = inputMask.clone();
    int minY = outputMask.rows;
    int maxY = -1;
    for (int y = 0; y < outputMask.rows; ++y) {
        if (cv::countNonZero(outputMask.row(y)) > 0) {
            minY = std::min(minY, y);
            maxY = std::max(maxY, y);
        }
    }
    if (maxY < 0 || minY >= maxY) return outputMask;

    const int chopHeight = static_cast<int>(proportion * (maxY - minY + 1));
    if (chopHeight <= 0) return outputMask;
    const int removeEndY = std::min(outputMask.rows, minY + chopHeight);
    if (removeEndY > minY) {
        outputMask(cv::Range(minY, removeEndY), cv::Range::all()).setTo(0);
    }
    return outputMask;
}

static int calculateMaxYLength(const cv::Mat &mask) {
    if (mask.empty()) return 0;
    if (mask.channels() != 1 || mask.type() != CV_8UC1) {
        throw std::invalid_argument("Input mask must be single channel (CV_8UC1)");
    }

    int maxLength = 0;
    for (int x = 0; x < mask.cols; ++x) {
        int currentLength = 0;
        for (int y = 0; y < mask.rows; ++y) {
            if (mask.at<uchar>(y, x) != 0) {
                ++currentLength;
                maxLength = std::max(maxLength, currentLength);
            } else {
                currentLength = 0;
            }
        }
    }
    return maxLength;
}

int AlgMaskRCNNInstanceSegm::solve() {
    try {
        auto pm = getParamsManager();
        const bool updateDebug2D = pm->getDebugInfos()
                                       ->findPropertyBase("OutputResultImage")
                                       ->getMetaTable()->get<bool>("update");

        const auto modelPathButton = pm->get<PButtonPath>("ModelPath").path();
        if (modelPathButton.empty()) S_THROW("The path is empty!");

        const std::string rootPath =
            System::getInstance()->getCurrentProjectPath();
        const std::string resourcePath = rootPath + "resource" + modelPathButton;

        // Swin-S 模型包使用 model.onnx。当资源按类似 YOLO26 准备时，也兼容 segmentation.onnx。
        std::string newModelPath = resourcePath + "/model.onnx";
        if (!fileExists(newModelPath)) {
            const std::string compatibilityPath = resourcePath + "/segmentation.onnx";
            if (fileExists(compatibilityPath)) newModelPath = compatibilityPath;
        }

        std::string newLabelsPath = resourcePath + "/classes.names";
        if (!fileExists(newLabelsPath)) {
            const std::string txtPath = resourcePath + "/classes.txt";
            newLabelsPath = fileExists(txtPath) ? txtPath : "";
        }

        _modelPath = newModelPath;
        _labelsPath = newLabelsPath;
        _max_obj = pm->get<PInt>("Numbers").value();
        _maskAlpha = pm->get<PDouble>("maskAlpha").value();

        struct stat modelStat;
        time_t newModelMTime = 0;
        if (stat(_modelPath.c_str(), &modelStat) == 0) {
            newModelMTime = modelStat.st_mtime;
        } else {
            s::common::Log::debug(USER_LOG, "Cannot stat model file: %s",
                                  _modelPath.c_str());
        }

#if USE_GPU
        const bool currentIsGPU = pm->get<PBool>("isGPU").value();
        const bool needRebuild =
            _firstRUN || currentIsGPU != _lastIsGPU ||
            _modelPath != _oldModelPath || _labelsPath != _oldLabelsPath ||
            (newModelMTime != _oldModelMTime && newModelMTime != 0);
        if (needRebuild) {
            _segmentor.reset();
            _segmentor.reset(new MaskRCNNDetector(
                _modelPath, _labelsPath, currentIsGPU));
            _lastIsGPU = currentIsGPU;
            _oldModelPath = _modelPath;
            _oldLabelsPath = _labelsPath;
            _oldModelMTime = newModelMTime;
            _firstRUN = false;
            s::common::Log::debug(
                USER_LOG,
                "Rebuilt MaskRCNNDetector (GPU=%d, model or labels changed)",
                currentIsGPU);
        }
#else
        if (_firstRUN || _modelPath != _oldModelPath ||
            _labelsPath != _oldLabelsPath ||
            (newModelMTime != _oldModelMTime && newModelMTime != 0)) {
            _segmentor.reset();
            _segmentor.reset(new MaskRCNNDetector(
                _modelPath, _labelsPath, false));
            _oldModelPath = _modelPath;
            _oldLabelsPath = _labelsPath;
            _oldModelMTime = newModelMTime;
            _firstRUN = false;
            s::common::Log::debug(
                USER_LOG,
                "Rebuilt MaskRCNNDetector (CPU mode, model or labels changed)");
        }
#endif

        if (!getting_input_images()) S_THROW("The image was not found!");

        cv::Mat image = inputSrcimage.clone();
        if (image.empty()) S_THROW("The image was not found!");
        if (image.channels() != 3) cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);

        const int maxImageSide = std::max(image.cols, image.rows);
        const double lineWidth = std::max(1.0, maxImageSide / 350.0);

        auto in = getInputs();
        (void)in;
        std::shared_ptr<PropertyMap> roiMap =
            pm->getROIInfos()->getPtr<PropertyMap>("InputROI");
        const int roiMode = roiMap->get<int>("mode");
        if (roiMode == 1) {
            if (roiMap->findPropertyBase("roi")->getType().getTypeName() ==
                traitTypeName<ROI2DRect>()) {
                _segmentor->setROI(roiMap->get<ROI2DRect>("roi"));
            }
        } else if (roiMode == 2) {
            const s::common::Vector2D center(image.cols / 2, image.rows / 2);
            const Rect2D rect(center, image.cols, image.rows);
            _segmentor->setROI(ROI2DRect(rect));
        }

        // ==================== 【根据 isFixedCamera 加载固定或随动模式的各自阈值】====================
        bool currentIsFixedCamera = pm->get<PBool>("isFixedCamera").value();

        MaskRCNNDetector::SegmentOptions segmentOptions;
        segmentOptions.maxDetections = static_cast<size_t>(std::max(1, _max_obj));
        const auto &classNames = _segmentor->getClassNames();
        segmentOptions.classScoreThresholds.reserve(classNames.size());
        segmentOptions.classIouThresholds.reserve(classNames.size());

        for (const std::string &className : classNames) {
            double current_conf = 0.50;
            double current_iou  = 0.50;

            if (currentIsFixedCamera) {
                // 固定相机类别的对应阈值过滤: leg, breast, cord, milkcup
                if (className == "leg") {
                    current_conf = pm->get<PDouble>("leg_confThreshold").value();
                    current_iou  = pm->get<PDouble>("leg_iouThreshold").value();
                } else if (className == "breast") {
                    current_conf = pm->get<PDouble>("breast_confThreshold").value();
                    current_iou  = pm->get<PDouble>("breast_iouThreshold").value();
                } else if (className == "cord") {
                    current_conf = pm->get<PDouble>("cord_confThreshold").value();
                    current_iou  = pm->get<PDouble>("cord_iouThreshold").value();
                } else if (className == "milkcup") {
                    current_conf = pm->get<PDouble>("milkcup_confThreshold").value();
                    current_iou  = pm->get<PDouble>("milkcup_iouThreshold").value();
                } else {
                    // 非当前流程关注的类别，默认置信度设为 1.0f 进行强过滤
                    current_conf = 1.0f;
                    current_iou  = 0.50f;
                }
            } else {
                // 随动相机类别的对应阈值过滤: leg, milkcup, nipple, tail
                if (className == "leg") {
                    current_conf = pm->get<PDouble>("leg_confThreshold").value();
                    current_iou  = pm->get<PDouble>("leg_iouThreshold").value();
                } else if (className == "milkcup") {
                    current_conf = pm->get<PDouble>("milkcup_confThreshold").value();
                    current_iou  = pm->get<PDouble>("milkcup_iouThreshold").value();
                } else if (className == "nipple") {
                    current_conf = pm->get<PDouble>("nipple_confThreshold").value();
                    current_iou  = pm->get<PDouble>("nipple_iouThreshold").value();
                } else if (className == "tail") {
                    current_conf = pm->get<PDouble>("tail_confThreshold").value();
                    current_iou  = pm->get<PDouble>("tail_iouThreshold").value();
                } else {
                    // 非当前流程关注的类别，默认置信度设为 1.0f 进行强过滤
                    current_conf = 0.50f;
                    current_iou  = 0.50f;
                }
            }

            segmentOptions.classScoreThresholds.push_back(static_cast<float>(current_conf));
            segmentOptions.classIouThresholds.push_back(static_cast<float>(current_iou));
        }

        Log::debug(USER_LOG,
                   "[AlgMaskRCNNInstanceSegm::solve()] segment start");
        std::vector<maskrcnn::SegmentedDetection> result =
            _segmentor->segment(image, segmentOptions);
        Log::debug(USER_LOG,
                   "[AlgMaskRCNNInstanceSegm::solve()] segment end");

        std::vector<s::vision::Region> regionMasksList;
        std::vector<s::vision::Region> legRegionList;
        std::vector<double> scoresList;
        std::vector<int> labelsList;
        std::vector<int> nippleLengthList;
        std::vector<std::string> labelNameList;
        std::vector<s::vision::Rect2D> boundBoxes;

        std::shared_ptr<sapp::Debug2D> debug2D = std::make_shared<sapp::Debug2D>();
        Debug2D::Debug2DLayer debugLayer("layer");
        std::sort(result.begin(), result.end(),
                  [](const maskrcnn::SegmentedDetection &left,
                     const maskrcnn::SegmentedDetection &right) {
                      return left.conf > right.conf;
                  });

        int valid_obj_count = 0;
        for (size_t index = 0; index < result.size(); ++index) {
            if (valid_obj_count >= _max_obj) break; // 达到最大目标数限制则跳出

            const std::string textName = getClassName(*_segmentor,
                                                      result[index].classId);

            // ==================== 【流程特异性类别校验与阈值双重过滤】====================
            double current_class_thresh = 0.50;
            if (currentIsFixedCamera) {
                if (textName == "leg")              current_class_thresh = pm->get<PDouble>("leg_confThreshold").value();
                else if (textName == "breast")      current_class_thresh = pm->get<PDouble>("breast_confThreshold").value();
                else if (textName == "cord")        current_class_thresh = pm->get<PDouble>("cord_confThreshold").value();
                else if (textName == "milkcup")     current_class_thresh = pm->get<PDouble>("milkcup_confThreshold").value();
                else continue; // 丢弃不属于固定流程的目标
            } else {
                if (textName == "leg")              current_class_thresh = pm->get<PDouble>("leg_confThreshold").value();
                else if (textName == "milkcup")     current_class_thresh = pm->get<PDouble>("milkcup_confThreshold").value();
                else if (textName == "nipple")      current_class_thresh = pm->get<PDouble>("nipple_confThreshold").value();
                else if (textName == "tail")        current_class_thresh = pm->get<PDouble>("tail_confThreshold").value();
                else continue; // 丢弃不属于随动流程的目标
            }

            if (result[index].conf < current_class_thresh) {
                continue; // 低于该类别阈值直接抛弃
            }
            valid_obj_count++;

            const double centerX = result[index].box.x + result[index].box.width / 2.0;
            const double centerY = result[index].box.y + result[index].box.height / 2.0;
            const auto rect = s::vision::Rect2D(
                s::common::Vector2D(centerX, centerY), result[index].box.width,
                result[index].box.height);
            boundBoxes.emplace_back(rect);
            scoresList.emplace_back(result[index].conf);
            labelsList.emplace_back(result[index].classId);
            labelNameList.emplace_back(textName);

            // 固定相机流程特有：乳房上方比例切除
            if (currentIsFixedCamera && textName == "breast") {
                const double cropRatio = pm->get<PDouble>("breastCropRatio").value();
                result[index].mask = removeTopProportionOfForeground(
                    result[index].mask, cropRatio);
            }

            // 随动相机流程特有：奶头长度统计
            if (textName == "nipple") {
                nippleLengthList.push_back(calculateMaxYLength(result[index].mask));
            }

            ImageGray maskGray;
            ImageUtility<S_8U, 1>::cv2duco(result[index].mask, maskGray);
            Region region = maskrcnn::utils::Mask2Region(maskGray);
            regionMasksList.emplace_back(region);
            if (textName == "leg") legRegionList.emplace_back(region);

            if (updateDebug2D) {
                int r, g, b;
                HSVtoRGB(&r, &g, &b,
                         static_cast<int>(360.0 / std::max<size_t>(1, classNames.size()) *
                                          std::max(0, result[index].classId)),
                         100, 100);

                Debug2D::BorderCfg borderBox;
                borderBox.color = s::vision::Color(0, 255, 0);
                borderBox.thickness = lineWidth;
                borderBox.type = Debug2D::BorderCfg::BorderType::BORDER_SOLID_LINE;
                borderBox.enable = true;
                Debug2D::FillCfg fill;
                fill.color = s::vision::Color(r, g, b);
                fill.color.setAlpha(static_cast<float>(_maskAlpha));
                fill.enable = false;
                debugLayer.add(rect, borderBox, fill);

                Debug2D::BorderCfg borderText;
                borderText.color = s::vision::Color(r, g, b);
                borderText.thickness = lineWidth / 4;
                borderText.type = Debug2D::BorderCfg::BorderType::BORDER_SOLID_LINE;
                borderText.enable = true;
                Text2D text;
                text.text = textName;
                text.pose = s::common::Pose2D(result[index].box.x,
                                              result[index].box.y, 0);
                Debug2D::FillCfg fillText;
                fillText.color = s::vision::Color(128, 128, 128);
                fillText.color.setAlpha(0.7f);
                fillText.enable = false;
                debugLayer.add(text, borderText, fillText);

                Debug2D::BorderCfg borderMask;
                borderMask.color = s::vision::Color(0, 255, 0);
                borderMask.thickness = 1;
                borderMask.type = Debug2D::BorderCfg::BorderType::BORDER_SOLID_LINE;
                borderMask.enable = false;
                fill.color = s::vision::Color(r, g, b);
                fill.color.setAlpha(static_cast<float>(_maskAlpha));
                fill.enable = true;
                debugLayer.add(region, borderMask, fill);
            }
        }

        // ==================== 【安全渲染 Debug2D 背景原图，避免破坏通道】====================
        if (updateDebug2D) {
            s::vision::ImageColor outputDebug;
            s::vision::ImageUtility<unsigned char, 3>::cv2duco(image, outputDebug);
            debugLayer.add(std::make_shared<s::vision::ImageColor>(outputDebug));
        }

        // ==================== 【安全绘制 ResultImage】====================
        const bool updateResult = pm->getDebugInfos()
                                      ->findPropertyBase("ResultImage")
                                      ->getMetaTable()->get<bool>("update");
        s::vision::ImageColor output;
        if (updateResult) {
            cv::Mat resultImg = image.clone();
            _segmentor->drawSegmentationsAndBoxes(
                resultImg, result, static_cast<float>(_maskAlpha));
            s::vision::ImageUtility<unsigned char, 3>::cv2duco(resultImg, output);
        }

        debug2D->add(debugLayer);
        sapp::BlockTask::clearOutput();
        sapp::BlockTask::clearDebug();
        sapp::BlockTask::addDebugValue("OutputResultImage", debug2D);
        sapp::BlockTask::addOutputValue("OutputLabels", labelsList);
        sapp::BlockTask::addOutputValue("OutputLabelNames", labelNameList);
        sapp::BlockTask::addOutputValue("OutputScore", scoresList);
        sapp::BlockTask::addOutputValue("OutputBoundBoxs", boundBoxes);
        sapp::BlockTask::addOutputValue("OutputMasks", regionMasksList);
        sapp::BlockTask::addOutputValue("LegRegionList", legRegionList);
        sapp::BlockTask::addOutputValue("ResultImage", output);
        sapp::BlockTask::addOutputValue("OutputNippleLenth", nippleLengthList);
        return 0;
    } catch (const Ort::Exception &error) {
        const std::string message = "ONNX Runtime Error: " + std::string(error.what());
        s::common::Log::error(USER_LOG, message.c_str());
        S_THROW(message.c_str());
    } catch (const std::exception &error) {
        const std::string message = "Standard Exception: " + std::string(error.what());
        s::common::Log::error(USER_LOG, message.c_str());
        S_THROW(message.c_str());
    } catch (...) {
        S_THROW("Unknown error occurred in solve()");
    }
}

std::shared_ptr<ParamManager> AlgMaskRCNNInstanceSegm::getDefultParams() {
    auto pm = std::make_shared<s::common::ParamManager>();
    auto inputImageReact = std::make_shared<s::common::ReactSelect>(
        s::common::ReactSelect::INPUT_TYPE);
    pm->addInputInfo(s::common::BlockInputInfo(
        "InputDetectImage",
        {traitPropertyType<s::vision::ImageColor>(),
         traitPropertyType<s::vision::ImageGray>()},
        true, {inputImageReact}));
    pm->addROIInfo(
        BlockROIInfo("InputROI", traitPropertyType<s::vision::ROI2DRect>(), true),
        s::vision::ROI2DRect(Rect2D({0, 0}, 0, 0)));

    pm->addDebugInfo(s::common::BlockDebugInfo("OutputResultImage",
                                               traitPropertyType<sapp::Debug2D>(),
                                               true));
    pm->addOutputInfo(s::common::BlockOutputInfo(
        "OutputLabels", traitPropertyType<std::vector<int>>(), true));
    pm->addOutputInfo(s::common::BlockOutputInfo(
        "OutputLabelNames", traitPropertyType<std::vector<std::string>>(), true));
    pm->addOutputInfo(s::common::BlockOutputInfo(
        "OutputScore", traitPropertyType<std::vector<double>>(), true));
    pm->addOutputInfo(s::common::BlockOutputInfo(
        "OutputBoundBoxs", traitPropertyType<std::vector<s::vision::Rect2D>>(),
        true));
    pm->addOutputInfo(s::common::BlockOutputInfo(
        "OutputMasks", traitPropertyType<std::vector<s::vision::Region>>(), true));
    pm->addOutputInfo(s::common::BlockOutputInfo(
        "LegRegionList", traitPropertyType<std::vector<s::vision::Region>>(),
        true));
    pm->addDebugInfo(s::common::BlockDebugInfo(
        "ResultImage", traitPropertyType<s::vision::ImageColor>(), true));
    pm->addOutputInfo(s::common::BlockOutputInfo(
        "OutputNippleLenth", traitPropertyType<std::vector<int>>(), true));

    pm->add("ModelPath", "",
            sapp::PButtonPath(
                "Opening", sapp::PButtonPath::PBPathMode::PBPATH_FILE_OPEN,
                sapp::PButtonPath::PBPathRoot::PBPATH_PROJ_RESOURCE,
                sapp::PButtonPath::PBPathFilter::PBPATH_DM_MODEL));
    pm->add("Numbers", "", sapp::PInt(10, 1, 1000, 1), true);
    pm->add("maskAlpha", "", sapp::PDouble(0.45, 0.0, 1.0, 0.05), true);
#if USE_GPU
    pm->add("isGPU", "", PBool(true), true);
#endif
    pm->add("isFixedCamera", "", PBool(true), true);
    pm->add("breastCropRatio", "", sapp::PDouble(0.3, 0.0, 1.0, 0.1), true);

    // ==================== 【固定流程与随动流程关注的类别独立 Conf 阈值】====================
    pm->add("leg_confThreshold", "Leg Thresh", sapp::PDouble(0.5, 0.0, 1.0, 0.05), true);
    pm->add("breast_confThreshold", "Breast Thresh", sapp::PDouble(0.5, 0.0, 1.0, 0.05), true);
    pm->add("cord_confThreshold", "Cord Thresh", sapp::PDouble(0.5, 0.0, 1.0, 0.05), true);
    pm->add("milkcup_confThreshold", "Milkcup Thresh", sapp::PDouble(0.5, 0.0, 1.0, 0.05), true);
    pm->add("nipple_confThreshold", "Nipple Thresh", sapp::PDouble(0.5, 0.0, 1.0, 0.05), true);
    pm->add("tail_confThreshold", "Tail Thresh", sapp::PDouble(0.5, 0.0, 1.0, 0.05), true);

    // ==================== 【固定流程与随动流程关注的类别独立 IoU 阈值】====================
    pm->add("leg_iouThreshold", "Leg IoU", sapp::PDouble(0.5, 0.0, 1.0, 0.05), true);
    pm->add("breast_iouThreshold", "Breast IoU", sapp::PDouble(0.5, 0.0, 1.0, 0.05), true);
    pm->add("cord_iouThreshold", "Cord IoU", sapp::PDouble(0.5, 0.0, 1.0, 0.05), true);
    pm->add("milkcup_iouThreshold", "Milkcup IoU", sapp::PDouble(0.5, 0.0, 1.0, 0.05), true);
    pm->add("nipple_iouThreshold", "Nipple IoU", sapp::PDouble(0.5, 0.0, 1.0, 0.05), true);
    pm->add("tail_iouThreshold", "Tail IoU", sapp::PDouble(0.5, 0.0, 1.0, 0.05), true);

    return pm;
}

void AlgMaskRCNNInstanceSegm::HSVtoRGB(int *r, int *g, int *b, int h, int s,
                                       int v) {
    const float rgbMax = v * 2.55f;
    const float rgbMin = rgbMax * (100 - s) / 100.0f;
    const int sector = h / 60;
    const int difference = h % 60;
    const float adjustment = (rgbMax - rgbMin) * difference / 60.0f;

    switch (sector) {
        case 0: *r = rgbMax; *g = rgbMin + adjustment; *b = rgbMin; break;
        case 1: *r = rgbMax - adjustment; *g = rgbMax; *b = rgbMin; break;
        case 2: *r = rgbMin; *g = rgbMax; *b = rgbMin + adjustment; break;
        case 3: *r = rgbMin; *g = rgbMax - adjustment; *b = rgbMax; break;
        case 4: *r = rgbMin + adjustment; *g = rgbMin; *b = rgbMax; break;
        default: *r = rgbMax; *g = rgbMin; *b = rgbMax - adjustment; break;
    }
}

#endif
