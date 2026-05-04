#ifndef AUTO_BUFF__YOLO11_BUFF_HPP
#define AUTO_BUFF__YOLO11_BUFF_HPP

#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "tools/infer/trt_engine.hpp"

namespace auto_buff
{
const std::vector<std::string> class_names = {"buff", "r"};

class YOLO11_BUFF
{
public:
  struct Object
  {
    cv::Rect_<float> rect;
    int label;
    float prob;
    std::vector<cv::Point2f> kpt;
  };

  YOLO11_BUFF(const std::string & config);

  // 使用NMS，用来获取多个框
  std::vector<Object> get_multicandidateboxes(cv::Mat & image);

  // 寻找置信度最高的框
  std::vector<Object> get_onecandidatebox(cv::Mat & image);

private:
  static constexpr int kInputSize = 640;
  static constexpr int kNumPoints = 6;
  static constexpr int kOutputRows = 5 + kNumPoints * 2;

  std::string engine_path_;
  std::unique_ptr<tools::infer::TrtEngine> trt_engine_;

  std::vector<float> preprocess(const cv::Mat & input_image, double & scale_to_original) const;

  bool infer(
    cv::Mat & image, double & scale_to_original, cv::Mat & output,
    std::vector<float> & raw_output);

  cv::Mat to_output_mat(const std::vector<float> & raw, const std::vector<int64_t> & shape) const;

  std::vector<Object> parse_candidates(
    const cv::Mat & output, double scale_to_original, bool only_best) const;

  Object build_object(
    const cv::Mat & output, int index, float confidence, double scale_to_original) const;

  void draw_detections(cv::Mat & image, const std::vector<Object> & objects, bool draw_index) const;

  // 将image保存为"../result/$${programName}.jpg"
  void save(const std::string & programName, const cv::Mat & image);
};
}  // namespace auto_buff
#endif
