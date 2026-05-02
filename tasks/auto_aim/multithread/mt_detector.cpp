#include "mt_detector.hpp"

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <stdexcept>

namespace auto_aim
{
namespace multithread
{
namespace
{

std::string to_onnx_path(const std::string & model_path)
{
  std::filesystem::path path(model_path);
  if (path.extension().string() == ".onnx") {
    return model_path;
  }

  const auto fallback = path.replace_extension(".onnx").string();
  if (std::filesystem::exists(fallback)) {
    tools::logger()->warn(
      "[MultiThreadDetector] model path {} is not ONNX, fallback to {}", model_path, fallback);
    return fallback;
  }

  tools::logger()->warn(
    "[MultiThreadDetector] model path {} is not ONNX and fallback .onnx is missing. "
    "Engine-only startup is possible when yolov5_engine_path exists.",
    model_path);
  return "";
}

std::string to_default_engine_path(const std::string & onnx_path)
{
  if (onnx_path.empty()) {
    return "";
  }

  std::filesystem::path path(onnx_path);
  path.replace_extension(".engine");
  return path.string();
}

std::size_t read_workspace_size_bytes(const YAML::Node & yaml)
{
  const auto workspace_mb = yaml["trt_workspace_mb"] ? yaml["trt_workspace_mb"].as<int>() : 1024;
  if (workspace_mb <= 0) {
    throw std::runtime_error("trt_workspace_mb must be a positive integer.");
  }

  return static_cast<std::size_t>(workspace_mb) * 1024ULL * 1024ULL;
}

std::vector<float> to_chw_rgb_normalized(const cv::Mat & bgr)
{
  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

  std::vector<float> chw(3 * rgb.rows * rgb.cols);
  std::vector<cv::Mat> channels(3);
  for (int c = 0; c < 3; ++c) {
    channels[c] = cv::Mat(rgb.rows, rgb.cols, CV_32F, chw.data() + c * rgb.rows * rgb.cols);
  }
  cv::split(rgb, channels);

  return chw;
}

cv::Mat to_output_mat(const std::vector<float> & raw, const std::vector<int64_t> & shape)
{
  if (shape.size() != 3) {
    throw std::runtime_error("Unexpected TensorRT output rank for yolov5 multithread.");
  }

  const int d1 = static_cast<int>(shape[1]);
  const int d2 = static_cast<int>(shape[2]);
  cv::Mat view(d1, d2, CV_32F, const_cast<float *>(raw.data()));

  if (d2 == 22) {
    return view;
  }

  if (d1 == 22) {
    cv::Mat transposed;
    cv::transpose(view, transposed);
    return transposed;
  }

  throw std::runtime_error("Unexpected yolov5 output layout in multithread detector.");
}

}  // namespace

MultiThreadDetector::MultiThreadDetector(const std::string & config_path, bool debug)
: yolo_(config_path, debug)
{
  auto yaml = YAML::LoadFile(config_path);

  auto model_path = to_onnx_path(yaml["yolov5_model_path"].as<std::string>());
  auto engine_path =
    yaml["yolov5_engine_path"] ? yaml["yolov5_engine_path"].as<std::string>()
                               : to_default_engine_path(model_path);
  if (model_path.empty() && engine_path.empty()) {
    throw std::runtime_error(
      "TensorRT yolov5 requires yolov5_model_path to resolve to an ONNX file, "
      "or yolov5_engine_path to point to an existing engine.");
  }

  tools::infer::TrtOptions trt_options;
  trt_options.enable_fp16 = yaml["trt_fp16"] ? yaml["trt_fp16"].as<bool>() : true;
  trt_options.force_rebuild =
    yaml["trt_force_rebuild"] ? yaml["trt_force_rebuild"].as<bool>() : false;
  trt_options.workspace_size_bytes = read_workspace_size_bytes(yaml);

  trt_engine_ = std::make_unique<tools::infer::TrtEngine>(
    model_path, engine_path, std::vector<int64_t>{1, 3, 640, 640}, trt_options);

  tools::logger()->info("[MultiThreadDetector] initialized !");
}

void MultiThreadDetector::push(cv::Mat img, std::chrono::steady_clock::time_point t)
{
  auto x_scale = static_cast<double>(640) / img.rows;
  auto y_scale = static_cast<double>(640) / img.cols;
  auto scale = std::min(x_scale, y_scale);
  auto h = static_cast<int>(img.rows * scale);
  auto w = static_cast<int>(img.cols * scale);

  // preprocess: keep same letterbox rule as single-thread detector.
  auto input = cv::Mat(640, 640, CV_8UC3, cv::Scalar(0, 0, 0));
  auto roi = cv::Rect(0, 0, w, h);
  cv::resize(img, input(roi), {w, h});

  auto chw_input = to_chw_rgb_normalized(input);
  std::vector<float> raw_output;
  if (!trt_engine_->infer(chw_input.data(), chw_input.size(), raw_output)) {
    tools::logger()->error("[MultiThreadDetector] TensorRT inference failed.");
    return;
  }

  queue_.push({img.clone(), t, std::move(raw_output)});
}

std::tuple<std::list<Armor>, std::chrono::steady_clock::time_point> MultiThreadDetector::pop()
{
  auto [img, t, raw_output] = queue_.pop();

  // postprocess
  auto output = to_output_mat(raw_output, trt_engine_->output_shape());
  auto x_scale = static_cast<double>(640) / img.rows;
  auto y_scale = static_cast<double>(640) / img.cols;
  auto scale = std::min(x_scale, y_scale);
  auto armors = yolo_.postprocess(scale, output, img, 0);  //暂不支持ROI

  return {std::move(armors), t};
}

std::tuple<cv::Mat, std::list<Armor>, std::chrono::steady_clock::time_point>
MultiThreadDetector::debug_pop()
{
  auto [img, t, raw_output] = queue_.pop();

  // postprocess
  auto output = to_output_mat(raw_output, trt_engine_->output_shape());
  auto x_scale = static_cast<double>(640) / img.rows;
  auto y_scale = static_cast<double>(640) / img.cols;
  auto scale = std::min(x_scale, y_scale);
  auto armors = yolo_.postprocess(scale, output, img, 0);  //暂不支持ROI

  return {img, std::move(armors), t};
}

}  // namespace multithread

}  // namespace auto_aim
