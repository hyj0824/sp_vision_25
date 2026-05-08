#ifndef AUTO_BUFF__RUNE_DETECTOR_HPP
#define AUTO_BUFF__RUNE_DETECTOR_HPP

#include <chrono>
#include <memory>
#include <opencv2/opencv.hpp>
#include <optional>
#include <string>
#include <vector>

#include "tools/infer/trt_engine.hpp"

namespace auto_buff
{

struct RuneDetection
{
  std::chrono::steady_clock::time_point timestamp;
  std::vector<cv::Point2f> keypoints;
  cv::Rect2f rect;
  float confidence = 0.0f;
};

class RuneDetector
{
public:
  explicit RuneDetector(const std::string & config_path);

  std::optional<RuneDetection> detect(
    const cv::Mat & bgr_img, std::chrono::steady_clock::time_point timestamp);

private:
  enum class InputLayout { NHWC, NCHW };

  std::string model_path_;
  std::string engine_path_;
  std::unique_ptr<tools::infer::TrtEngine> trt_engine_;

  int input_width_ = 640;
  int input_height_ = 384;
  float confidence_threshold_ = 0.7f;
  bool normalize_input_ = false;
  bool debug_ = false;
  InputLayout input_layout_ = InputLayout::NHWC;

  std::vector<float> preprocess(
    const cv::Mat & bgr_img, double & x_scale_to_original, double & y_scale_to_original) const;

  cv::Mat to_output_mat(const std::vector<float> & raw, const std::vector<int64_t> & shape) const;

  std::optional<RuneDetection> parse_best(
    const cv::Mat & output, double x_scale_to_original, double y_scale_to_original,
    std::chrono::steady_clock::time_point timestamp) const;

  void draw_detection(cv::Mat & bgr_img, const RuneDetection & detection) const;
};

}  // namespace auto_buff

#endif  // AUTO_BUFF__RUNE_DETECTOR_HPP
