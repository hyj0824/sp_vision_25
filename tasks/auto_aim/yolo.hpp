#ifndef AUTO_AIM__YOLO_HPP
#define AUTO_AIM__YOLO_HPP

#include <list>
#include <memory>
#include <opencv2/opencv.hpp>
#include <cstdint>
#include <string>
#include <vector>

#include "armor.hpp"
#include "detector.hpp"
#include "tools/infer/trt_engine.hpp"

namespace auto_aim
{
class YOLO
{
public:
  YOLO(const std::string & config_path, bool debug = true);

  std::list<Armor> detect(const cv::Mat & img, int frame_count = -1);

  std::list<Armor> postprocess(
    double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count);

private:
  std::string engine_path_;
  std::string save_path_, debug_path_;
  bool debug_, use_roi_, use_traditional_;

  const float nms_threshold_ = 0.3;
  const float score_threshold_ = 0.7;
  double min_confidence_, binary_threshold_;

  std::unique_ptr<tools::infer::TrtEngine> trt_engine_;
  cv::Mat input_buffer_;
  cv::Size letterbox_size_;
  std::vector<std::uint16_t> fp16_input_;
  std::vector<float> raw_output_;

  cv::Rect roi_;
  cv::Point2f offset_;
  cv::Mat tmp_img_;

  Detector detector_;

  bool check_name(const Armor & armor) const;
  bool check_type(const Armor & armor) const;

  cv::Point2f get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const;

  std::list<Armor> parse(double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count);

  void save(const Armor & armor) const;
  void draw_detections(const cv::Mat & img, const std::list<Armor> & armors, int frame_count) const;
  double sigmoid(double x);
};

}  // namespace auto_aim

#endif  // AUTO_AIM__YOLO_HPP
