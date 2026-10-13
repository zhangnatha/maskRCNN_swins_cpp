/** @file AlgMaskRCNN-InstanceSegm.h
 *  @brief Mask R-CNN 实例分割算法插件头文件
 *  @author zhangJA
 *  @date 2026.10.13
 *  @version 1.0
 *  @copyright zhangnatha@qq.com
 */

#pragma once

#if __has_include("common/SettingType.h")
#include "common/SettingType.h"
#include "s/common/Core/Exception.h"
#include "s/common/Core/PropertyBase.h"
#include "s/vision/Core/Image.h"
#include "s/vision/Vision2D/Utility/UtilityBuffer.h"
#include "salgorithm/Common/AlgoritmPlugin.h"
#include "sapp/system/System.h"
#endif

#if __has_include("s/vision/DeepLearning/swin-MaskRCNN/MaskRCNN-InstanceSegmentation.h")
#include "s/vision/DeepLearning/swin-MaskRCNN/MaskRCNN-InstanceSegmentation.h"
#else
#include "MaskRCNN-InstanceSegmentation.h"
#endif

#if __has_include("salgorithm/Common/AlgoritmPlugin.h")
namespace sapp {

class AlgMaskRCNNInstanceSegm : public AlgoritmBlockPlugin
{
public:
    AlgMaskRCNNInstanceSegm();

    bool checkValued() override;
    bool handleFailed() override;
    int solve() override;
    std::shared_ptr<s::common::ParamManager> getDefultParams() override;

public:
    ~AlgMaskRCNNInstanceSegm() override;
    void HSVtoRGB(int *r, int *g, int *b, int h, int s, int v);
    int input_size = 640;

private:
    s::vision::ImageBuffer<S_8U> src_img;
    cv::Mat inputSrcimage;
    std::string model_type;
    double _confthreshold = 0.50f;
    double _iouthreshold = 0.50f;
    int _max_obj = 1000;
    double _maskAlpha = 0.45f;
    bool _drawMaskImage = true;
    std::string _modelPath = "";
    std::string _labelsPath = "";
    bool _isGPU = true;
    bool getting_input_images();
    std::unique_ptr<MaskRCNNDetector> _segmentor;

    std::string _oldModelPath = "";
    std::string _oldLabelsPath = "";
    time_t _oldModelMTime = 0;
    bool _lastIsGPU = false;
    bool _firstRUN = true;
};

PLUMA_INHERIT_PROVIDER(AlgMaskRCNNInstanceSegm, BlockTask);

} // namespace sapp
#endif
