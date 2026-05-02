#include "yolo11_buff.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>

#include "tools/logger.hpp"

namespace auto_buff
{
namespace
{

constexpr float ConfidenceThreshold = 0.7f;
constexpr float IouThreshold = 0.4f;

std::string read_model_path(const YAML::Node & yaml)
{
  if (yaml["buff_model_path"]) {
    return yaml["buff_model_path"].as<std::string>();
  }

  if (yaml["model"]) {
    tools::logger()->warn("Config key 'model' is deprecated; use 'buff_model_path' instead.");
    return yaml["model"].as<std::string>();
  }

  return "";
}

std::string to_onnx_path(const std::string & model_path)
{
  if (model_path.empty()) {
    return "";
  }

  std::filesystem::path path(model_path);
  if (path.extension().string() == ".onnx") {
    return model_path;
  }

  const auto fallback = path.replace_extension(".onnx").string();
  if (std::filesystem::exists(fallback)) {
    tools::logger()->warn(
      "buff_model_path={} is not ONNX, fallback to {} for TensorRT build.",
      model_path,
      fallback);
    return fallback;
  }

  tools::logger()->warn(
    "buff_model_path={} is not ONNX and no fallback .onnx found. "
    "Engine-only startup is still possible if buff_engine_path exists.",
    model_path);
  return "";
}

std::string to_default_engine_path(const std::string & path)
{
  if (path.empty()) {
    return "";
  }

  std::filesystem::path engine_path(path);
  engine_path.replace_extension(".engine");
  return engine_path.string();
}

std::string shape_to_string(const std::vector<int64_t> & shape)
{
  std::string result = "[";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    result += std::to_string(shape[i]);
    if (i + 1 < shape.size()) {
      result += ",";
    }
  }
  result += "]";
  return result;
}

std::string display_path(const std::string & path)
{
  return path.empty() ? std::string("<empty>") : path;
}

std::size_t read_workspace_size_bytes(const YAML::Node & yaml)
{
  const auto workspace_mb =
    yaml["buff_trt_workspace_mb"]
      ? yaml["buff_trt_workspace_mb"].as<int>()
      : (yaml["trt_workspace_mb"] ? yaml["trt_workspace_mb"].as<int>() : 1024);
  if (workspace_mb <= 0) {
    throw std::runtime_error("trt_workspace_mb must be a positive integer.");
  }

  return static_cast<std::size_t>(workspace_mb) * 1024ULL * 1024ULL;
}

bool read_bool(
  const YAML::Node & yaml, const char * primary, const char * fallback, bool default_value)
{
  if (yaml[primary]) {
    return yaml[primary].as<bool>();
  }

  if (yaml[fallback]) {
    return yaml[fallback].as<bool>();
  }

  return default_value;
}

}  // namespace

YOLO11_BUFF::YOLO11_BUFF(const std::string & config)
{
  const auto yaml = YAML::LoadFile(config);
  const auto configured_model_path = read_model_path(yaml);

  model_path_ = to_onnx_path(configured_model_path);
  engine_path_ =
    yaml["buff_engine_path"] ? yaml["buff_engine_path"].as<std::string>()
                             : to_default_engine_path(
                                 model_path_.empty() ? configured_model_path : model_path_);

  if (model_path_.empty() && engine_path_.empty()) {
    throw std::runtime_error(
      "TensorRT auto_buff requires buff_model_path to resolve to an ONNX file, "
      "or buff_engine_path to point to an existing engine.");
  }

  tools::infer::TrtOptions trt_options;
  trt_options.enable_fp16 = read_bool(yaml, "buff_trt_fp16", "trt_fp16", true);
  trt_options.force_rebuild = read_bool(yaml, "buff_trt_force_rebuild", "trt_force_rebuild", false);
  trt_options.workspace_size_bytes = read_workspace_size_bytes(yaml);

  const bool model_exists = !model_path_.empty() && std::filesystem::exists(model_path_);
  const bool engine_exists = !engine_path_.empty() && std::filesystem::exists(engine_path_);
  if ((!engine_exists || trt_options.force_rebuild) && !model_exists) {
    throw std::runtime_error(
      "TensorRT auto_buff requires an existing buff_engine_path or an existing ONNX "
      "buff_model_path for engine build. buff_model_path=" +
      display_path(model_path_) + ", buff_engine_path=" + display_path(engine_path_));
  }

  trt_engine_ = std::make_unique<tools::infer::TrtEngine>(
    model_path_, engine_path_, std::vector<int64_t>{1, 3, kInputSize, kInputSize}, trt_options);
}

std::vector<YOLO11_BUFF::Object> YOLO11_BUFF::get_multicandidateboxes(cv::Mat & image)
{
  const int64 start = cv::getTickCount();

  double scale_to_original = 1.0;
  cv::Mat output;
  std::vector<float> raw_output;
  if (!infer(image, scale_to_original, output, raw_output)) {
    return {};
  }

  auto object_result = parse_candidates(output, scale_to_original, false);
  draw_detections(image, object_result, false);

  const float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
  cv::putText(
    image, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0,
    cv::Scalar(255, 0, 0), 2, 8);

  return object_result;
}

std::vector<YOLO11_BUFF::Object> YOLO11_BUFF::get_onecandidatebox(cv::Mat & image)
{
  const int64 start = cv::getTickCount();

  double scale_to_original = 1.0;
  cv::Mat output;
  std::vector<float> raw_output;
  if (!infer(image, scale_to_original, output, raw_output)) {
    return {};
  }

  auto object_result = parse_candidates(output, scale_to_original, true);
  draw_detections(image, object_result, true);

  const float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
  cv::putText(
    image, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0,
    cv::Scalar(255, 0, 0), 2, 8);

  return object_result;
}

std::vector<float> YOLO11_BUFF::preprocess(
  const cv::Mat & input_image, double & scale_to_original) const
{
  const double scale = std::min(
    static_cast<double>(kInputSize) / input_image.rows,
    static_cast<double>(kInputSize) / input_image.cols);
  const int resized_h = static_cast<int>(input_image.rows * scale);
  const int resized_w = static_cast<int>(input_image.cols * scale);
  scale_to_original = 1.0 / scale;

  cv::Mat letterbox(kInputSize, kInputSize, CV_8UC3, cv::Scalar(0, 0, 0));
  const auto roi = cv::Rect(0, 0, resized_w, resized_h);
  cv::resize(input_image, letterbox(roi), {resized_w, resized_h});

  cv::Mat rgb;
  cv::cvtColor(letterbox, rgb, cv::COLOR_BGR2RGB);
  rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

  std::vector<float> chw(3 * kInputSize * kInputSize);
  std::vector<cv::Mat> channels(3);
  for (int c = 0; c < 3; ++c) {
    channels[c] =
      cv::Mat(kInputSize, kInputSize, CV_32F, chw.data() + c * kInputSize * kInputSize);
  }
  cv::split(rgb, channels);

  return chw;
}

bool YOLO11_BUFF::infer(
  cv::Mat & image, double & scale_to_original, cv::Mat & output, std::vector<float> & raw_output)
{
  if (image.empty()) {
    tools::logger()->warn("Empty img!, camera drop!");
    return false;
  }

  const auto input = preprocess(image, scale_to_original);
  if (!trt_engine_->infer(input.data(), input.size(), raw_output)) {
    tools::logger()->error("TensorRT auto_buff inference failed.");
    return false;
  }

  output = to_output_mat(raw_output, trt_engine_->output_shape());
  return true;
}

cv::Mat YOLO11_BUFF::to_output_mat(
  const std::vector<float> & raw, const std::vector<int64_t> & shape) const
{
  if (shape.size() == 3) {
    const int d1 = static_cast<int>(shape[1]);
    const int d2 = static_cast<int>(shape[2]);
    cv::Mat view(d1, d2, CV_32F, const_cast<float *>(raw.data()));

    if (d1 == kOutputRows) {
      return view;
    }

    if (d2 == kOutputRows) {
      cv::Mat transposed;
      cv::transpose(view, transposed);
      return transposed;
    }
  }

  if (shape.size() == 2) {
    const int d0 = static_cast<int>(shape[0]);
    const int d1 = static_cast<int>(shape[1]);
    cv::Mat view(d0, d1, CV_32F, const_cast<float *>(raw.data()));

    if (d0 == kOutputRows) {
      return view;
    }

    if (d1 == kOutputRows) {
      cv::Mat transposed;
      cv::transpose(view, transposed);
      return transposed;
    }
  }

  throw std::runtime_error(
    "Unexpected TensorRT output layout for auto_buff. Expected 17 output rows, got shape=" +
    shape_to_string(shape));
}

std::vector<YOLO11_BUFF::Object> YOLO11_BUFF::parse_candidates(
  const cv::Mat & output, double scale_to_original, bool only_best) const
{
  if (only_best) {
    int best_index = -1;
    float max_confidence = 0.0f;
    for (int i = 0; i < output.cols; ++i) {
      const float confidence = output.at<float>(4, i);
      if (confidence > max_confidence) {
        max_confidence = confidence;
        best_index = i;
      }
    }

    if (best_index < 0 || max_confidence <= ConfidenceThreshold) {
      return {};
    }

    return {build_object(output, best_index, max_confidence, scale_to_original)};
  }

  std::vector<cv::Rect> boxes;
  std::vector<float> confidences;
  std::vector<Object> objects;

  for (int i = 0; i < output.cols; ++i) {
    const float score = output.at<float>(4, i);
    if (score <= ConfidenceThreshold) {
      continue;
    }

    auto object = build_object(output, i, score, scale_to_original);
    boxes.emplace_back(object.rect);
    confidences.emplace_back(score);
    objects.emplace_back(std::move(object));
  }

  std::vector<int> indexes;
  cv::dnn::NMSBoxes(boxes, confidences, ConfidenceThreshold, IouThreshold, indexes);

  std::vector<Object> object_result;
  object_result.reserve(indexes.size());
  for (const auto index : indexes) {
    object_result.emplace_back(objects[static_cast<std::size_t>(index)]);
  }

  return object_result;
}

YOLO11_BUFF::Object YOLO11_BUFF::build_object(
  const cv::Mat & output, int index, float confidence, double scale_to_original) const
{
  const float cx = output.at<float>(0, index) * static_cast<float>(scale_to_original);
  const float cy = output.at<float>(1, index) * static_cast<float>(scale_to_original);
  const float ow = output.at<float>(2, index) * static_cast<float>(scale_to_original);
  const float oh = output.at<float>(3, index) * static_cast<float>(scale_to_original);

  Object object;
  object.rect = cv::Rect_<float>(cx - 0.5f * ow, cy - 0.5f * oh, ow, oh);
  object.label = 0;
  object.prob = confidence;

  object.kpt.reserve(kNumPoints);
  for (int i = 0; i < kNumPoints; ++i) {
    const int row = 5 + i * 2;
    const float x = output.at<float>(row, index) * static_cast<float>(scale_to_original);
    const float y = output.at<float>(row + 1, index) * static_cast<float>(scale_to_original);
    object.kpt.emplace_back(x, y);
  }

  return object;
}

void YOLO11_BUFF::draw_detections(
  cv::Mat & image, const std::vector<Object> & objects, bool draw_index) const
{
  for (const auto & obj : objects) {
    cv::rectangle(image, obj.rect, cv::Scalar(255, 255, 255), 1, 8);

    const std::string label = "buff:" + std::to_string(obj.prob).substr(0, 4);
    const cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, nullptr);
    const cv::Rect textBox(
      static_cast<int>(obj.rect.tl().x),
      static_cast<int>(obj.rect.tl().y - 15),
      textSize.width,
      textSize.height + 5);
    cv::rectangle(image, textBox, cv::Scalar(0, 255, 255), cv::FILLED);
    cv::putText(
      image,
      label,
      cv::Point(static_cast<int>(obj.rect.tl().x), static_cast<int>(obj.rect.tl().y - 5)),
      cv::FONT_HERSHEY_SIMPLEX,
      0.5,
      cv::Scalar(0, 0, 0));

    for (int i = 0; i < kNumPoints; ++i) {
      cv::circle(image, obj.kpt[i], 2, cv::Scalar(255, 255, 0), -1, cv::LINE_AA);
      if (draw_index) {
        cv::putText(
          image,
          std::to_string(i + 1),
          obj.kpt[i] + cv::Point2f(5, -5),
          cv::FONT_HERSHEY_SIMPLEX,
          0.5,
          cv::Scalar(255, 255, 0),
          1,
          cv::LINE_AA);
      }
    }
  }
}

void YOLO11_BUFF::save(const std::string & programName, const cv::Mat & image)
{
  const std::filesystem::path saveDir = "../result/";
  if (!std::filesystem::exists(saveDir)) {
    std::filesystem::create_directories(saveDir);
  }
  const std::filesystem::path savePath = saveDir / (programName + ".jpg");
  cv::imwrite(savePath.string(), image);
}

}  // namespace auto_buff
