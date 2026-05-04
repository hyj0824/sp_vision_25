#include "yolo.hpp"

#include <fmt/chrono.h>
#include <opencv2/core/hal/interface.h>
#include <yaml-cpp/yaml.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <stdexcept>

#include "tools/img_tools.hpp"
#include "tools/logger.hpp"

namespace auto_aim
{
namespace
{

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

std::uint16_t float_to_half_bits(float value)
{
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));

  const auto sign = static_cast<std::uint16_t>((bits >> 16) & 0x8000);
  auto exponent = static_cast<int>((bits >> 23) & 0xff) - 127 + 15;
  auto mantissa = bits & 0x7fffff;

  if (exponent <= 0) {
    if (exponent < -10) return sign;
    mantissa |= 0x800000;
    const auto shift = 14 - exponent;
    return static_cast<std::uint16_t>(sign | ((mantissa + (1u << (shift - 1))) >> shift));
  }

  if (exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7c00);

  mantissa += 0x1000;
  if (mantissa & 0x800000) {
    mantissa = 0;
    ++exponent;
    if (exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7c00);
  }

  return static_cast<std::uint16_t>(sign | (exponent << 10) | (mantissa >> 13));
}

const std::array<std::uint16_t, 256> & normalized_half_lut()
{
  static const auto lut = [] {
    std::array<std::uint16_t, 256> values;
    for (std::size_t i = 0; i < values.size(); ++i) {
      values[i] = float_to_half_bits(static_cast<float>(i) / 255.0f);
    }
    return values;
  }();
  return lut;
}

void write_chw_rgb_normalized_fp16(const cv::Mat & bgr, std::vector<std::uint16_t> & chw)
{
  const auto & lut = normalized_half_lut();
  const auto pixels = static_cast<std::size_t>(bgr.rows) * bgr.cols;
  chw.resize(3 * pixels);
  auto * r = chw.data();
  auto * g = r + pixels;
  auto * b = g + pixels;

  for (int y = 0; y < bgr.rows; ++y) {
    const auto * row = bgr.ptr<cv::Vec3b>(y);
    const auto row_offset = static_cast<std::size_t>(y) * bgr.cols;
    for (int x = 0; x < bgr.cols; ++x) {
      const auto offset = row_offset + x;
      const auto & pixel = row[x];
      r[offset] = lut[pixel[2]];
      g[offset] = lut[pixel[1]];
      b[offset] = lut[pixel[0]];
    }
  }
}

cv::Mat to_output_mat(const std::vector<float> & raw, const std::vector<int64_t> & shape)
{
  if (shape.size() != 3) {
    throw std::runtime_error("Unexpected TensorRT output rank for yolov5.");
  }

  const int d1 = static_cast<int>(shape[1]);
  const int d2 = static_cast<int>(shape[2]);
  cv::Mat view(d1, d2, CV_32F, const_cast<float *>(raw.data()));

  // Keep parse() unchanged: it expects rows=num_boxes and cols=num_attrs(22).
  if (d2 == 22) {
    return view;
  }

  if (d1 == 22) {
    cv::Mat transposed;
    cv::transpose(view, transposed);
    return transposed;
  }

  throw std::runtime_error("Unexpected yolov5 output layout, cannot map to parse format.");
}

}  // namespace

YOLO::YOLO(const std::string & config_path, bool debug)
: debug_(debug), detector_(config_path, false)
{
  auto yaml = YAML::LoadFile(config_path);

  if (!yaml["yolov5_engine_path"]) {
    throw std::runtime_error("TensorRT yolov5 requires yolov5_engine_path.");
  }
  engine_path_ = yaml["yolov5_engine_path"].as<std::string>();
  if (engine_path_.empty()) {
    throw std::runtime_error("TensorRT yolov5_engine_path must not be empty.");
  }

  binary_threshold_ = yaml["threshold"].as<double>();
  min_confidence_ = yaml["min_confidence"].as<double>();
  int x = 0, y = 0, width = 0, height = 0;
  x = yaml["roi"]["x"].as<int>();
  y = yaml["roi"]["y"].as<int>();
  width = yaml["roi"]["width"].as<int>();
  height = yaml["roi"]["height"].as<int>();
  use_roi_ = yaml["use_roi"].as<bool>();
  use_traditional_ = yaml["use_traditional"].as<bool>();
  roi_ = cv::Rect(x, y, width, height);
  offset_ = cv::Point2f(x, y);

  save_path_ = "imgs";
  std::filesystem::create_directory(save_path_);

  trt_engine_ = std::make_unique<tools::infer::TrtEngine>(engine_path_);
}

std::list<Armor> YOLO::detect(const cv::Mat & raw_img, int frame_count)
{
  if (raw_img.empty()) {
    tools::logger()->warn("Empty img!, camera drop!");
    return std::list<Armor>();
  }

  cv::Mat bgr_img;
  if (use_roi_) {
    if (roi_.width == -1) {  // -1 表示该维度不裁切
      roi_.width = raw_img.cols;
    }
    if (roi_.height == -1) {  // -1 表示该维度不裁切
      roi_.height = raw_img.rows;
    }
    bgr_img = raw_img(roi_);
  } else {
    bgr_img = raw_img;
  }

  auto x_scale = static_cast<double>(640) / bgr_img.rows;
  auto y_scale = static_cast<double>(640) / bgr_img.cols;
  auto scale = std::min(x_scale, y_scale);
  auto h = static_cast<int>(bgr_img.rows * scale);
  auto w = static_cast<int>(bgr_img.cols * scale);

  // preprocess: letterbox to 640x640, then BGR->RGB and HWC->CHW.
  if (input_buffer_.empty()) {
    input_buffer_ = cv::Mat(640, 640, CV_8UC3, cv::Scalar(0, 0, 0));
  }
  if (letterbox_size_ != cv::Size(w, h)) {
    input_buffer_.setTo(cv::Scalar(0, 0, 0));
    letterbox_size_ = cv::Size(w, h);
  }
  auto roi = cv::Rect(0, 0, w, h);
  cv::resize(bgr_img, input_buffer_(roi), {w, h});

  if (trt_engine_->input_is_fp16()) {
    write_chw_rgb_normalized_fp16(input_buffer_, fp16_input_);
    raw_output_.resize(trt_engine_->output_elements());
    if (!trt_engine_->infer(
        fp16_input_.data(),
        fp16_input_.size() * sizeof(fp16_input_[0]),
        raw_output_.data(),
        raw_output_.size() * sizeof(raw_output_[0])))
    {
      tools::logger()->error("TensorRT yolov5 inference failed.");
      return std::list<Armor>();
    }

    auto output = to_output_mat(raw_output_, trt_engine_->output_shape());
    return parse(scale, output, raw_img, frame_count);
  }

  auto chw_input = to_chw_rgb_normalized(input_buffer_);
  raw_output_.resize(trt_engine_->output_elements());
  if (!trt_engine_->infer(
      chw_input.data(),
      chw_input.size() * sizeof(chw_input[0]),
      raw_output_.data(),
      raw_output_.size() * sizeof(raw_output_[0])))
  {
    tools::logger()->error("TensorRT yolov5 inference failed.");
    return std::list<Armor>();
  }

  auto output = to_output_mat(raw_output_, trt_engine_->output_shape());
  return parse(scale, output, raw_img, frame_count);
}

std::list<Armor> YOLO::parse(
  double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  // for each row: xywh + classess
  std::vector<int> color_ids, num_ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;
  std::vector<std::vector<cv::Point2f>> armors_key_points;
  for (int r = 0; r < output.rows; r++) {
    double score = output.at<float>(r, 8);
    score = sigmoid(score);

    if (score < score_threshold_) continue;

    std::vector<cv::Point2f> armor_key_points;

    //颜色和类别独热向量
    cv::Mat color_scores = output.row(r).colRange(9, 13);     //color
    cv::Mat classes_scores = output.row(r).colRange(13, 22);  //num
    cv::Point class_id, color_id;
    int _class_id, _color_id;
    double score_color, score_num;
    cv::minMaxLoc(classes_scores, NULL, &score_num, NULL, &class_id);
    cv::minMaxLoc(color_scores, NULL, &score_color, NULL, &color_id);
    _class_id = class_id.x;
    _color_id = color_id.x;

    armor_key_points.push_back(
      cv::Point2f(output.at<float>(r, 0) / scale, output.at<float>(r, 1) / scale));
    armor_key_points.push_back(
      cv::Point2f(output.at<float>(r, 6) / scale, output.at<float>(r, 7) / scale));
    armor_key_points.push_back(
      cv::Point2f(output.at<float>(r, 4) / scale, output.at<float>(r, 5) / scale));
    armor_key_points.push_back(
      cv::Point2f(output.at<float>(r, 2) / scale, output.at<float>(r, 3) / scale));

    float min_x = armor_key_points[0].x;
    float max_x = armor_key_points[0].x;
    float min_y = armor_key_points[0].y;
    float max_y = armor_key_points[0].y;

    for (int i = 1; i < armor_key_points.size(); i++) {
      if (armor_key_points[i].x < min_x) min_x = armor_key_points[i].x;
      if (armor_key_points[i].x > max_x) max_x = armor_key_points[i].x;
      if (armor_key_points[i].y < min_y) min_y = armor_key_points[i].y;
      if (armor_key_points[i].y > max_y) max_y = armor_key_points[i].y;
    }

    cv::Rect rect(min_x, min_y, max_x - min_x, max_y - min_y);

    color_ids.emplace_back(_color_id);
    num_ids.emplace_back(_class_id);
    boxes.emplace_back(rect);
    confidences.emplace_back(score);
    armors_key_points.emplace_back(armor_key_points);
  }

  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, score_threshold_, nms_threshold_, indices);

  std::list<Armor> armors;
  for (const auto & i : indices) {
    if (use_roi_) {
      armors.emplace_back(
        color_ids[i], num_ids[i], confidences[i], boxes[i], armors_key_points[i], offset_);
    } else {
      armors.emplace_back(color_ids[i], num_ids[i], confidences[i], boxes[i], armors_key_points[i]);
    }
  }

  tmp_img_ = bgr_img;
  for (auto it = armors.begin(); it != armors.end();) {
    if (!check_name(*it)) {
      it = armors.erase(it);
      continue;
    }

    if (!check_type(*it)) {
      it = armors.erase(it);
      continue;
    }
    // 使用传统方法二次矫正角点
    if (use_traditional_) detector_.detect(*it, bgr_img);

    it->center_norm = get_center_norm(bgr_img, it->center);
    ++it;
  }

  if (debug_) draw_detections(bgr_img, armors, frame_count);

  return armors;
}

bool YOLO::check_name(const Armor & armor) const
{
  auto name_ok = armor.name != ArmorName::not_armor;
  auto confidence_ok = armor.confidence > min_confidence_;

  // 保存不确定的图案，用于神经网络的迭代
  // if (name_ok && !confidence_ok) save(armor);

  return name_ok && confidence_ok;
}

bool YOLO::check_type(const Armor & armor) const
{
  auto name_ok = (armor.type == ArmorType::small)
                   ? (armor.name != ArmorName::one && armor.name != ArmorName::base)
                   : (armor.name != ArmorName::two && armor.name != ArmorName::sentry &&
                      armor.name != ArmorName::outpost);

  // 保存异常的图案，用于神经网络的迭代
  // if (!name_ok) save(armor);

  return name_ok;
}

cv::Point2f YOLO::get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const
{
  auto h = bgr_img.rows;
  auto w = bgr_img.cols;
  return {center.x / w, center.y / h};
}

void YOLO::draw_detections(
  const cv::Mat & img, const std::list<Armor> & armors, int frame_count) const
{
  auto detection = img.clone();
  tools::draw_text(detection, fmt::format("[{}]", frame_count), {10, 30}, {255, 255, 255});
  for (const auto & armor : armors) {
    auto info = fmt::format(
      "{:.2f} {} {} {}", armor.confidence, COLORS[armor.color], ARMOR_NAMES[armor.name],
      ARMOR_TYPES[armor.type]);
    tools::draw_points(detection, armor.points, {0, 255, 0});
    tools::draw_text(detection, info, armor.center, {0, 255, 0});
  }

  if (use_roi_) {
    cv::Scalar green(0, 255, 0);
    cv::rectangle(detection, roi_, green, 2);
  }
  cv::resize(detection, detection, {}, 0.5, 0.5);  // 显示时缩小图片尺寸
  cv::imshow("detection", detection);
}

void YOLO::save(const Armor & armor) const
{
  auto file_name = fmt::format("{:%Y-%m-%d_%H-%M-%S}", std::chrono::system_clock::now());
  auto img_path = fmt::format("{}/{}_{}.jpg", save_path_, armor.name, file_name);
  cv::imwrite(img_path, tmp_img_);
}

double YOLO::sigmoid(double x)
{
  if (x > 0)
    return 1.0 / (1.0 + exp(-x));
  else
    return exp(x) / (1.0 + exp(x));
}

std::list<Armor> YOLO::postprocess(
  double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  return parse(scale, output, bgr_img, frame_count);
}

}  // namespace auto_aim
