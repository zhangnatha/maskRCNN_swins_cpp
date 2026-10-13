/** @file test_model_inference.cpp
 *  @brief Mask R-CNN 模型独立 C++ 推理测试主程序
 *  @author zhangJA
 *  @date 2026.10.13
 *  @version 1.0
 *  @copyright zhangnatha@qq.com
 */

// 用法: test_model_inference MODEL LABELS IMAGE_OR_DIR cpu|gpu [OUTPUT_DIR]
// 或:   test_model_inference --model MODEL --input IMAGE_OR_DIR --output DIR
//         --classes class1,class2 [--class-conf class1=0.6]
//         [--class-iou class1=0.5] [--device cpu|gpu]
#include "MaskRCNN-InstanceSegmentation.h"
#include <opencv2/core/utils/filesystem.hpp>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <memory>
#include <cstdint>
#include <iomanip>
#include <locale>
#include <sstream>

namespace {

struct Arguments {
    std::string model, labels, input, output, device{"cpu"};
    std::vector<std::string> classes;
    std::string classConf, classIou;
    float scoreThreshold{0.50f};
};

std::string trim(const std::string &value) {
    const size_t first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    const size_t last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::vector<std::string> split(const std::string &value, char delimiter) {
    std::vector<std::string> items;
    std::istringstream stream(value);
    std::string item;
    while (std::getline(stream, item, delimiter)) {
        item = trim(item);
        if (item.empty()) throw std::runtime_error("empty item in comma-separated value");
        items.push_back(item);
    }
    if (!value.empty() && value.back() == delimiter)
        throw std::runtime_error("empty item in comma-separated value");
    return items;
}

inline std::string getTempDirectory() {
#ifdef _WIN32
    const char *tmp = std::getenv("TEMP");
    if (!tmp) tmp = std::getenv("TMP");
    if (tmp) return std::string(tmp);
    return ".";
#else
    const char *tmp = std::getenv("TMPDIR");
    if (tmp) return std::string(tmp);
    return "/tmp";
#endif
}

class TemporaryLabels {
public:
    explicit TemporaryLabels(const std::vector<std::string> &classes) {
        static std::atomic<uint64_t> counter{0};
        const auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        const std::string filename = "maskrcnn_classes_" + std::to_string(now) + "_" +
                                     std::to_string(++counter) + ".tmp";
        _path = cv::utils::fs::join(getTempDirectory(), filename);
        std::ofstream file(_path);
        if (!file.is_open()) {
            throw std::runtime_error("unable to create temporary labels file: " + _path);
        }
        for (const std::string &name : classes) {
            file << name << "\n";
        }
        file.close();
        if (!file) {
            std::remove(_path.c_str());
            throw std::runtime_error("unable to write temporary labels file: " + _path);
        }
    }
    ~TemporaryLabels() {
        if (!_path.empty()) {
            std::remove(_path.c_str());
        }
    }
    const std::string &path() const { return _path; }
    TemporaryLabels(const TemporaryLabels &) = delete;
    TemporaryLabels &operator=(const TemporaryLabels &) = delete;
private:
    std::string _path;
};

std::vector<float> parseClassThresholds(const std::string &value,
                                        const std::vector<std::string> &classes,
                                        float fallback, const char *option) {
    std::vector<float> thresholds(classes.size(), fallback);
    std::vector<bool> seen(classes.size(), false);
    if (value.empty()) return thresholds;
    for (const std::string &item : split(value, ',')) {
        const size_t equal = item.find('=');
        if (equal == std::string::npos)
            throw std::runtime_error(std::string(option) + " expects CLASS=VALUE[,CLASS=VALUE]");
        const std::string name = trim(item.substr(0, equal));
        const auto found = std::find(classes.begin(), classes.end(), name);
        if (found == classes.end()) throw std::runtime_error(std::string(option) + " contains unknown class: " + name);
        const size_t index = static_cast<size_t>(found - classes.begin());
        if (seen[index]) throw std::runtime_error(std::string(option) + " contains duplicate class: " + name);
        const std::string raw = trim(item.substr(equal + 1));
        size_t parsedLength = 0;
        const float threshold = std::stof(raw, &parsedLength);
        if (parsedLength != raw.size() || !std::isfinite(threshold) || threshold < 0.0f || threshold > 1.0f)
            throw std::runtime_error(std::string(option) + " values must be in [0, 1]");
        thresholds[index] = threshold;
        seen[index] = true;
    }
    return thresholds;
}

Arguments parseArguments(int argc, char **argv) {
    Arguments args;
    if (argc >= 2 && argv[1][0] != '-') {
        if (argc < 5 || argc > 6) throw std::runtime_error("invalid positional arguments");
        args.model = argv[1]; args.labels = argv[2]; args.input = argv[3]; args.device = argv[4];
        if (argc == 6) args.output = argv[5];
        return args;
    }
    for (int i = 1; i < argc; ++i) {
        const std::string key(argv[i]);
        if (key == "--help" || key == "-h") {
            std::cout << "Usage: test_model_inference --model MODEL --input IMAGE_OR_DIR --output DIR "
                         "--classes class1,class2,... [--class-conf class1=0.6] "
                         "[--class-iou class1=0.5] [--device cpu|gpu] [--score-threshold 0.5]\n"
                         "   or: test_model_inference MODEL LABELS IMAGE_OR_DIR cpu|gpu [OUTPUT_DIR]\n";
            std::exit(0);
        }
        if (i + 1 >= argc) throw std::runtime_error("missing value for " + key);
        const std::string value(argv[++i]);
        if (key == "--model") args.model = value;
        else if (key == "--input") args.input = value;
        else if (key == "--output") args.output = value;
        else if (key == "--device") args.device = value;
        else if (key == "--classes") args.classes = split(value, ',');
        else if (key == "--class-conf") args.classConf = value;
        else if (key == "--class-iou") args.classIou = value;
        else if (key == "--score-threshold") {
            size_t parsedLength = 0;
            args.scoreThreshold = std::stof(value, &parsedLength);
            if (parsedLength != value.size() || !std::isfinite(args.scoreThreshold) || args.scoreThreshold < 0.0f || args.scoreThreshold > 1.0f)
                throw std::runtime_error("--score-threshold must be in [0, 1]");
        } else throw std::runtime_error("unknown argument: " + key);
    }
    if (args.model.empty() || args.input.empty() || args.output.empty() || args.classes.empty())
        throw std::runtime_error("--model, --input, --output and --classes are required");
    for (size_t i = 0; i < args.classes.size(); ++i) {
        if (args.classes[i].find_first_of("\r\n") != std::string::npos)
            throw std::runtime_error("--classes names cannot contain newlines");
        if (std::find(args.classes.begin(), args.classes.begin() + i, args.classes[i]) != args.classes.begin() + i)
            throw std::runtime_error("--classes contains duplicate class: " + args.classes[i]);
    }
    return args;
}

std::string outputPrefix(const std::string &outputDir,
                         const std::string &imagePath) {
    const size_t nameStart = imagePath.find_last_of("/\\");
    std::string name = imagePath.substr(nameStart == std::string::npos ? 0 : nameStart + 1);
    const size_t extension = name.find_last_of('.');
    if (extension != std::string::npos && extension != 0) name.erase(extension);
    return cv::utils::fs::join(outputDir, name);
}

std::string jsonString(const std::string &value) {
    std::ostringstream out;
    out << '"';
    for (const unsigned char c : value) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 0x20)
            out << "\\u00" << std::hex << std::setw(2) << std::setfill('0')
                << static_cast<unsigned int>(c);
        else out << c;
    }
    out << '"';
    return out.str();
}

void writeJson(const std::string &path, const std::string &imagePath,
               const cv::Size &size,
               const std::vector<maskrcnn::SegmentedDetection> &results,
               const std::vector<std::string> &classes) {
    std::ofstream output(path);
    if (!output) throw std::runtime_error("failed to open JSON: " + path);
    output.imbue(std::locale::classic());
    output << std::setprecision(9) << "{\n  \"image\": " << jsonString(imagePath)
           << ",\n  \"width\": " << size.width << ",\n  \"height\": " << size.height
           << ",\n  \"detections\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const auto &d = results[i];
        output << "    {\"class_id\": " << d.classId << ", \"class_name\": "
               << jsonString(classes[d.classId]) << ", \"score\": " << d.conf
               << ", \"bbox_xyxy\": [" << d.box.x << ", " << d.box.y << ", "
               << d.box.x + d.box.width << ", " << d.box.y + d.box.height << "]";
        // 精确匹配行优先 0/1 掩码与哈希种子计算。
        uint64_t hash = 1469598103934665603ULL;
        size_t area = 0, run = 0;
        unsigned char current = 0;
        std::vector<size_t> runs;
        for (int y = 0; y < d.mask.rows; ++y) {
            const unsigned char *row = d.mask.ptr<unsigned char>(y);
            for (int x = 0; x < d.mask.cols; ++x) {
                const unsigned char value = row[x] != 0;
                area += value;
                hash ^= value;
                hash *= 1099511628211ULL;
                if (value == current) ++run;
                else {
                    runs.push_back(run);
                    current = value;
                    run = 1;
                }
            }
        }
        runs.push_back(run);
        output << ", \"mask_area\": " << area << ", \"mask_fnv1a64\": \""
               << std::hex << std::setw(16) << std::setfill('0') << hash
               << std::dec << std::setfill(' ') << "\", \"mask_rle\": [";
        for (size_t r = 0; r < runs.size(); ++r) {
            if (r) output << ", ";
            output << runs[r];
        }
        output << "], \"mask_polygons\": [";
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(d.mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        for (size_t c = 0; c < contours.size(); ++c) {
            if (c) output << ", ";
            output << "[";
            for (size_t p = 0; p < contours[c].size(); ++p) {
                if (p) output << ", ";
                output << "[" << contours[c][p].x << ", " << contours[c][p].y << "]";
            }
            output << "]";
        }
        output << "]}" << (i + 1 == results.size() ? "\n" : ",\n");
    }
    output << "  ]\n}\n";
    output.close();
    if (!output) throw std::runtime_error("failed to save JSON: " + path);
}

bool processImage(MaskRCNNDetector &detector, const std::string &imagePath,
                  const std::string &outputDir,
                  const MaskRCNNDetector::SegmentOptions &options) {
    const cv::Mat image = cv::imread(imagePath);
    if (image.empty()) return false;

    std::cout << "[INFO] Image: " << imagePath << '\n';
    const auto results = detector.segment(image, options);
    detector.printSegmentationResults(results);
    for (const auto &result : results) {
        if (result.classId < 0 ||
            static_cast<size_t>(result.classId) >= detector.getClassNames().size() ||
            !std::isfinite(result.conf) || result.conf < 0.0f || result.conf > 1.0f ||
            result.mask.size() != image.size() || result.mask.type() != CV_8UC1 ||
            cv::countNonZero(result.mask) == 0) {
            throw std::runtime_error("invalid detection or mask");
        }
    }

    if (!outputDir.empty()) {
        const std::string prefix = outputPrefix(outputDir, imagePath);
        cv::Mat overlay = image.clone();
        detector.drawSegmentationsAndBoxes(overlay, results);
        if (!cv::imwrite(prefix + ".png", overlay)) {
            throw std::runtime_error("failed to save overlay");
        }
        writeJson(prefix + ".json", imagePath, image.size(), results,
                  detector.getClassNames());
    }
    std::cout << "PASS: " << results.size() << " detections with valid masks\n";
    return true;
}

} // namespace

int main(int argc, char **argv) {
    try {
        const Arguments args = parseArguments(argc, argv);
        const bool useGPU = args.device == "gpu";
        if (!useGPU && args.device != "cpu") {
            throw std::runtime_error("backend must be cpu or gpu");
        }
        if (useGPU) {
            const auto providers = Ort::GetAvailableProviders();
            if (std::find(providers.begin(), providers.end(), "CUDAExecutionProvider") == providers.end()) {
                throw std::runtime_error("gpu requested but CUDAExecutionProvider is unavailable");
            }
        }
        if (args.classes.empty() && args.labels.empty())
            throw std::runtime_error("class names are required via --classes or LABELS");
        MaskRCNNDetector::SegmentOptions options;
        options.scoreThreshold = args.scoreThreshold;
        if (!args.classes.empty()) {
            options.classScoreThresholds = parseClassThresholds(
                args.classConf, args.classes, args.scoreThreshold, "--class-conf");
            options.classIouThresholds = parseClassThresholds(
                args.classIou, args.classes, options.iouThreshold, "--class-iou");
        }
        std::unique_ptr<TemporaryLabels> temporaryLabels;
        std::string labelsPath = args.labels;
        if (!args.classes.empty()) {
            temporaryLabels.reset(new TemporaryLabels(args.classes));
            labelsPath = temporaryLabels->path();
        }
        MaskRCNNDetector detector(args.model, labelsPath, useGPU);
        const std::string inputPath = args.input;
        const std::string outputDir = args.output;
        if (!outputDir.empty() && !cv::utils::fs::createDirectories(outputDir))
            throw std::runtime_error("failed to create output directory: " + outputDir);
        std::vector<cv::String> imagePaths;
        if (!cv::utils::fs::exists(inputPath))
            throw std::runtime_error("input path does not exist: " + inputPath);
        if (cv::utils::fs::isDirectory(inputPath)) {
            cv::glob(inputPath + "/*", imagePaths, false);
        } else {
            imagePaths.push_back(inputPath);
        }
        if (imagePaths.empty()) throw std::runtime_error("no readable images found");

        bool failed = false;
        size_t readableImages = 0;
        for (const auto &imagePath : imagePaths) {
            try {
                if (!processImage(detector, imagePath, outputDir, options)) continue;
                ++readableImages;
            } catch (const std::exception &error) {
                std::cerr << "[ERROR] " << imagePath << ": " << error.what() << '\n';
                failed = true;
            }
        }
        if (readableImages == 0) throw std::runtime_error("no readable images found");
        return failed ? 1 : 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
