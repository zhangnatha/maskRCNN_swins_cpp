/** @file MaskRCNN-Util.h
 *  @brief Mask R-CNN 实例分割通用工具结构体与辅助函数
 *  @author zhangJA
 *  @date 2026.10.13
 *  @version 1.0
 *  @copyright zhangnatha@qq.com
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

#if __has_include("s/vision/Vision2D/Utility/UtilityImage.h")
#include "s/vision/Vision2D/Utility/UtilityImage.h"
#endif

namespace maskrcnn {

/* ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ 分类结果 (Classification) ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ */
struct ClassificationResult {
    int classId{-1};
    float confidence{0.0f};
    std::string className{};

    ClassificationResult() = default;
    ClassificationResult(int id, float conf, std::string name)
        : classId(id), confidence(conf), className(std::move(name)) {}
};

/* ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ 目标检测 (Detection) ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ */
struct BoundingBox {
    int x{0};
    int y{0};
    int width{0};
    int height{0};

    BoundingBox() = default;
    BoundingBox(int x_, int y_, int width_, int height_)
        : x(x_), y(y_), width(width_), height(height_) {}
};

struct Detection {
    BoundingBox box;
    float conf{0.0f};
    int classId{-1};
};

/* ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ 实例分割 (Instance segmentation) ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ */
struct SegBoundingBox {
    int x{0};
    int y{0};
    int width{0};
    int height{0};

    SegBoundingBox() = default;
    SegBoundingBox(int x_, int y_, int width_, int height_)
        : x(x_), y(y_), width(width_), height(height_) {}

    float area() const {
        return static_cast<float>(std::max(0, width) * std::max(0, height));
    }

    SegBoundingBox intersect(const SegBoundingBox &other) const {
        int xStart = std::max(x, other.x);
        int yStart = std::max(y, other.y);
        int xEnd = std::min(x + width, other.x + other.width);
        int yEnd = std::min(y + height, other.y + other.height);
        return SegBoundingBox(xStart, yStart, std::max(0, xEnd - xStart),
                              std::max(0, yEnd - yStart));
    }
};

struct SegmentedDetection {
    SegBoundingBox box;
    float conf{0.0f};
    int classId{-1};
    cv::Mat mask; // 原始图像坐标系下的全尺寸 CV_8UC1 掩码
};

namespace utils {

template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, T>::type
inline clamp(const T &value, const T &low, const T &high) {
    T validLow = low < high ? low : high;
    T validHigh = low < high ? high : low;

    if (value < validLow) return validLow;
    if (value > validHigh) return validHigh;
    return value;
}

inline std::vector<std::string> getClassNames(const std::string &path) {
    std::vector<std::string> classNames;
    std::ifstream infile(path);

    if (infile) {
        std::string line;
        while (std::getline(infile, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            const size_t comment = line.find('#');
            if (comment != std::string::npos) line.erase(comment);
            std::istringstream values(line);
            std::string name;
            if (values >> name) classNames.emplace_back(std::move(name));
        }
    } else if (!path.empty()) {
        std::cerr << "ERROR: Failed to access class name path: " << path << std::endl;
    }

    if (classNames.empty()) {
        throw std::runtime_error("Class name file is empty or unreadable: " + path);
    }

    std::cout << "Loaded " << classNames.size() << " class names"
              << (path.empty() ? "" : " from " + path) << std::endl;
    return classNames;
}

inline size_t vectorProduct(const std::vector<int64_t> &values) {
    return std::accumulate(values.begin(), values.end(), size_t{1},
                           [](size_t left, int64_t right) {
                               if (right <= 0) return size_t{0};
                               return left * static_cast<size_t>(right);
                           });
}

inline void SegNMSBoxes(const std::vector<SegBoundingBox> &boxes,
                        const std::vector<float> &scores,
                        float scoreThreshold, float nmsThreshold,
                        std::vector<int> &indices) {
    indices.clear();
    if (boxes.empty() || boxes.size() != scores.size()) return;

    std::vector<int> order;
    order.reserve(boxes.size());
    for (size_t i = 0; i < boxes.size(); ++i) {
        if (scores[i] >= scoreThreshold) order.push_back(static_cast<int>(i));
    }
    std::sort(order.begin(), order.end(), [&scores](int left, int right) {
        return scores[left] > scores[right];
    });

    std::vector<float> areas(boxes.size(), 0.0f);
    for (size_t i = 0; i < boxes.size(); ++i) areas[i] = boxes[i].area();

    std::vector<bool> suppressed(boxes.size(), false);
    for (size_t i = 0; i < order.size(); ++i) {
        const int index = order[i];
        if (suppressed[index]) continue;
        indices.push_back(index);

        for (size_t j = i + 1; j < order.size(); ++j) {
            const int other = order[j];
            if (suppressed[other]) continue;

            const int x1 = std::max(boxes[index].x, boxes[other].x);
            const int y1 = std::max(boxes[index].y, boxes[other].y);
            const int x2 = std::min(boxes[index].x + boxes[index].width,
                                    boxes[other].x + boxes[other].width);
            const int y2 = std::min(boxes[index].y + boxes[index].height,
                                    boxes[other].y + boxes[other].height);
            const int intersectionWidth = std::max(0, x2 - x1);
            const int intersectionHeight = std::max(0, y2 - y1);
            if (intersectionWidth == 0 || intersectionHeight == 0) continue;

            const float intersection = static_cast<float>(intersectionWidth) *
                                       static_cast<float>(intersectionHeight);
            const float unionArea = areas[index] + areas[other] - intersection;
            const float iou = unionArea > 0.0f ? intersection / unionArea : 0.0f;
            if (iou > nmsThreshold) suppressed[other] = true;
        }
    }
}

// 辅助函数：HSV 转 RGB（整数 [0, 255]）
inline void HSVtoRGB(int *r, int *g, int *b, int h, int s, int v) {
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

// 生成各类别颜色列表（OpenCV 默认色彩空间顺序为 BGR）
inline std::vector<cv::Scalar> generateHSVColors(int numClasses) {
    std::vector<cv::Scalar> colors;
    colors.reserve(numClasses);
    int maxObj = std::max(1, numClasses);

    for (int i = 0; i < numClasses; ++i) {
        int r = 0, g = 0, b = 0;
        int hue = static_cast<int>(360.0 / maxObj * i);
        HSVtoRGB(&r, &g, &b, hue, 100, 100);
        // OpenCV 图像是 BGR 排布，传入 B, G, R
        colors.emplace_back(cv::Scalar(b, g, r));
    }
    return colors;
}

inline size_t countUniqueCategories(const std::string &filename) {
    std::ifstream file(filename);
    if (!file.is_open()) return 0;

    std::set<std::string> categories;
    std::string word;
    while (file >> word) categories.insert(word);
    return categories.size();
}

inline std::string getClassNameByIndex(const std::string &filename, size_t index) {
    const auto names = getClassNames(filename);
    if (index >= names.size()) {
        throw std::out_of_range("Class index " + std::to_string(index) +
                                " is out of range");
    }
    return names[index];
}

#if __has_include("s/vision/Vision2D/Utility/UtilityImage.h")
inline s::vision::Region Mask2Region(s::vision::ImageGray mask) {
    cv::Mat temp;
    s::vision::ImageUtility<S_8U, 1>::duco2cv(mask, temp);

    s::vision::Region region;
    region.region_row_vector.clear();
    region.region_start_col_vector.clear();
    region.region_end_col_vector.clear();
    if (temp.empty()) return region;

    for (int row = 0; row < temp.rows; ++row) {
        bool inRegion = false;
        for (int col = 0; col < temp.cols; ++col) {
            const bool foreground = temp.at<uchar>(row, col) != 0;
            if (foreground && !inRegion) {
                inRegion = true;
                region.region_row_vector.push_back(row);
                region.region_start_col_vector.push_back(col);
            } else if (!foreground && inRegion) {
                inRegion = false;
                region.region_end_col_vector.push_back(col);
            }
        }
        if (inRegion) region.region_end_col_vector.push_back(temp.cols);
    }
    return region;
}
#endif

} // namespace utils
} // namespace maskrcnn
