#include "rune_detector.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>

#include "tools/logger.hpp"

namespace auto_buff
{
namespace
{

std::string read_string(
  const YAML::Node & yaml, const char * primary, const char * fallback,
  const std::string & default_value = "")
{
  if (yaml[primary]) return yaml[primary].as<std::string>();
  if (yaml[fallback]) return yaml[fallback].as<std::string>();
  return default_value;
}

std::string to_onnx_path(const std::string & model_path)
{
  if (model_path.empty()) return "";

  std::filesystem::path path(model_path);
  if (path.extension().string() == ".onnx") return model_path;

  const auto fallback = path.replace_extension(".onnx").string();
  if (std::filesystem::exists(fallback)) {
    tools::logger()->warn(
      "rune_model_path={} is not ONNX, fallback to {} for TensorRT build.",
      model_path,
      fallback);
    return fallback;
  }

  tools::logger()->warn(
    "rune_model_path={} is not ONNX and no fallback .onnx found. "
    "Engine-only startup is still possible if rune_engine_path exists.",
    model_path);
  return "";
}

std::string to_default_engine_path(const std::string & path)
{
  if (path.empty()) return "";
  std::filesystem::path engine_path(path);
  engine_path.replace_extension(".engine");
  return engine_path.string();
}

std::size_t read_workspace_size_bytes(const YAML::Node & yaml)
{
  const auto workspace_mb =
    yaml["rune_trt_workspace_mb"]
      ? yaml["rune_trt_workspace_mb"].as<int>()
      : (yaml["trt_workspace_mb"] ? yaml["trt_workspace_mb"].as<int>() : 1024);
  if (workspace_mb <= 0) {
    throw std::runtime_error("rune_trt_workspace_mb must be a positive integer.");
  }
  return static_cast<std::size_t>(workspace_mb) * 1024ULL * 1024ULL;
}

bool read_bool(
  const YAML::Node & yaml, const char * primary, const char * fallback, bool default_value)
{
  if (yaml[primary]) return yaml[primary].as<bool>();
  if (yaml[fallback]) return yaml[fallback].as<bool>();
  return default_value;
}

float sigmoid(float x)
{
  if (x >= 0.0f) {
    const float z = std::exp(-x);
    return 1.0f / (1.0f + z);
  }
  const float z = std::exp(x);
  return z / (1.0f + z);
}

float decode_confidence(float raw)
{
  if (!std::isfinite(raw)) return 0.0f;
  if (raw >= 0.0f && raw <= 1.0f) return raw;
  return sigmoid(raw);
}

std::string shape_to_string(const std::vector<int64_t> & shape)
{
  std::string result = "[";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    result += std::to_string(shape[i]);
    if (i + 1 < shape.size()) result += ",";
  }
  result += "]";
  return result;
}

bool looks_like_attr_dim(int dim) { return dim >= 9 && dim <= 64; }

}  // namespace

RuneDetector::RuneDetector(const std::string & config_path)
{
  const auto yaml = YAML::LoadFile(config_path);

  const auto configured_model_path = read_string(yaml, "rune_model_path", "buff_model_path");
  model_path_ = to_onnx_path(configured_model_path);

  engine_path_ =
    yaml["rune_engine_path"] ? yaml["rune_engine_path"].as<std::string>()
                             : (yaml["buff_engine_path"]
                                  ? yaml["buff_engine_path"].as<std::string>()
                                  : to_default_engine_path(
                                      model_path_.empty() ? configured_model_path : model_path_));

  if (model_path_.empty() && engine_path_.empty()) {
    throw std::runtime_error(
      "TensorRT rune detector requires rune_model_path/buff_model_path to resolve to an ONNX "
      "file, or rune_engine_path/buff_engine_path to point to an existing engine.");
  }

  input_width_ = yaml["rune_input_width"] ? yaml["rune_input_width"].as<int>() : 640;
  input_height_ = yaml["rune_input_height"] ? yaml["rune_input_height"].as<int>() : 384;
  if (input_width_ <= 0 || input_height_ <= 0) {
    throw std::runtime_error("rune_input_width and rune_input_height must be positive.");
  }

  confidence_threshold_ =
    yaml["rune_confidence_threshold"]
      ? yaml["rune_confidence_threshold"].as<float>()
      : (yaml["numThresh"] ? yaml["numThresh"].as<float>() : 0.7f);
  normalize_input_ = yaml["rune_normalize_input"] ? yaml["rune_normalize_input"].as<bool>() : false;
  debug_ = yaml["rune_debug"] ? yaml["rune_debug"].as<bool>() : false;

  const auto layout =
    yaml["rune_input_layout"] ? yaml["rune_input_layout"].as<std::string>() : "nhwc";
  if (layout == "nhwc" || layout == "NHWC") {
    input_layout_ = InputLayout::NHWC;
  } else if (layout == "nchw" || layout == "NCHW") {
    input_layout_ = InputLayout::NCHW;
  } else {
    throw std::runtime_error("rune_input_layout must be 'nhwc' or 'nchw'.");
  }

  tools::infer::TrtOptions trt_options;
  trt_options.enable_fp16 = read_bool(yaml, "rune_trt_fp16", "trt_fp16", true);
  trt_options.force_rebuild = read_bool(yaml, "rune_trt_force_rebuild", "trt_force_rebuild", false);
  trt_options.workspace_size_bytes = read_workspace_size_bytes(yaml);

  const bool model_exists = !model_path_.empty() && std::filesystem::exists(model_path_);
  const bool engine_exists = !engine_path_.empty() && std::filesystem::exists(engine_path_);
  if ((!engine_exists || trt_options.force_rebuild) && !model_exists) {
    throw std::runtime_error(
      "TensorRT rune detector requires an existing engine or ONNX model. rune_model_path=" +
      (model_path_.empty() ? std::string("<empty>") : model_path_) +
      ", rune_engine_path=" + (engine_path_.empty() ? std::string("<empty>") : engine_path_));
  }

  const auto input_shape = (input_layout_ == InputLayout::NHWC)
                             ? std::vector<int64_t>{1, input_height_, input_width_, 3}
                             : std::vector<int64_t>{1, 3, input_height_, input_width_};
  trt_engine_ =
    std::make_unique<tools::infer::TrtEngine>(model_path_, engine_path_, input_shape, trt_options);
}

std::optional<RuneDetection> RuneDetector::detect(
  const cv::Mat & bgr_img, std::chrono::steady_clock::time_point timestamp)
{
  if (bgr_img.empty()) {
    tools::logger()->warn("[RuneDetector] Empty image.");
    return std::nullopt;
  }

  double x_scale_to_original = 1.0;
  double y_scale_to_original = 1.0;
  const auto input = preprocess(bgr_img, x_scale_to_original, y_scale_to_original);

  std::vector<float> raw_output;
  if (!trt_engine_->infer(input.data(), input.size(), raw_output)) {
    tools::logger()->error("[RuneDetector] TensorRT inference failed.");
    return std::nullopt;
  }

  const auto output = to_output_mat(raw_output, trt_engine_->output_shape());
  auto detection = parse_best(output, x_scale_to_original, y_scale_to_original, timestamp);

  if (debug_ && detection.has_value()) {
    cv::Mat debug_img = bgr_img.clone();
    draw_detection(debug_img, detection.value());
    cv::imshow("rune_detector", debug_img);
  }

  return detection;
}

std::vector<float> RuneDetector::preprocess(
  const cv::Mat & bgr_img, double & x_scale_to_original, double & y_scale_to_original) const
{
  x_scale_to_original = static_cast<double>(bgr_img.cols) / input_width_;
  y_scale_to_original = static_cast<double>(bgr_img.rows) / input_height_;

  cv::Mat resized;
  cv::resize(bgr_img, resized, cv::Size(input_width_, input_height_));
  cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
  resized.convertTo(resized, CV_32F, normalize_input_ ? 1.0 / 255.0 : 1.0);

  if (input_layout_ == InputLayout::NHWC) {
    const auto * begin = reinterpret_cast<const float *>(resized.data);
    return std::vector<float>(begin, begin + 3 * input_width_ * input_height_);
  }

  std::vector<float> chw(3 * input_width_ * input_height_);
  std::vector<cv::Mat> channels(3);
  for (int c = 0; c < 3; ++c) {
    channels[c] =
      cv::Mat(input_height_, input_width_, CV_32F, chw.data() + c * input_width_ * input_height_);
  }
  cv::split(resized, channels);
  return chw;
}

cv::Mat RuneDetector::to_output_mat(
  const std::vector<float> & raw, const std::vector<int64_t> & shape) const
{
  if (shape.size() == 3) {
    const int d1 = static_cast<int>(shape[1]);
    const int d2 = static_cast<int>(shape[2]);
    cv::Mat view(d1, d2, CV_32F, const_cast<float *>(raw.data()));
    if (looks_like_attr_dim(d2)) return view;
    if (looks_like_attr_dim(d1)) {
      cv::Mat transposed;
      cv::transpose(view, transposed);
      return transposed;
    }
  }

  if (shape.size() == 2) {
    const int d0 = static_cast<int>(shape[0]);
    const int d1 = static_cast<int>(shape[1]);
    cv::Mat view(d0, d1, CV_32F, const_cast<float *>(raw.data()));
    if (looks_like_attr_dim(d1)) return view;
    if (looks_like_attr_dim(d0)) {
      cv::Mat transposed;
      cv::transpose(view, transposed);
      return transposed;
    }
  }

  throw std::runtime_error(
    "Unexpected TensorRT rune output layout. Expected candidate attrs >= 9, got shape=" +
    shape_to_string(shape));
}

std::optional<RuneDetection> RuneDetector::parse_best(
  const cv::Mat & output, double x_scale_to_original, double y_scale_to_original,
  std::chrono::steady_clock::time_point timestamp) const
{
  int best_row = -1;
  float best_confidence = 0.0f;
  for (int row = 0; row < output.rows; ++row) {
    const float confidence = decode_confidence(output.at<float>(row, 8));
    if (confidence > best_confidence) {
      best_confidence = confidence;
      best_row = row;
    }
  }

  if (best_row < 0 || best_confidence < confidence_threshold_) {
    return std::nullopt;
  }

  RuneDetection detection;
  detection.timestamp = timestamp;
  detection.confidence = best_confidence;
  detection.keypoints.reserve(4);

  float min_x = std::numeric_limits<float>::max();
  float min_y = std::numeric_limits<float>::max();
  float max_x = std::numeric_limits<float>::lowest();
  float max_y = std::numeric_limits<float>::lowest();

  for (int i = 0; i < 4; ++i) {
    const float x = output.at<float>(best_row, i * 2) * static_cast<float>(x_scale_to_original);
    const float y =
      output.at<float>(best_row, i * 2 + 1) * static_cast<float>(y_scale_to_original);
    if (!std::isfinite(x) || !std::isfinite(y)) return std::nullopt;
    detection.keypoints.emplace_back(x, y);
    min_x = std::min(min_x, x);
    min_y = std::min(min_y, y);
    max_x = std::max(max_x, x);
    max_y = std::max(max_y, y);
  }

  detection.rect = cv::Rect2f(min_x, min_y, max_x - min_x, max_y - min_y);
  return detection;
}

void RuneDetector::draw_detection(cv::Mat & bgr_img, const RuneDetection & detection) const
{
  for (std::size_t i = 0; i < detection.keypoints.size(); ++i) {
    cv::circle(bgr_img, detection.keypoints[i], 4, cv::Scalar(0, 0, 255), -1);
    cv::putText(
      bgr_img, std::to_string(i), detection.keypoints[i], cv::FONT_HERSHEY_SIMPLEX, 0.8,
      cv::Scalar(255, 255, 255), 2);
  }
  cv::rectangle(bgr_img, detection.rect, cv::Scalar(0, 255, 0), 2);
  cv::putText(
    bgr_img, cv::format("%.2f", detection.confidence), detection.rect.tl(),
    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
}

}  // namespace auto_buff
