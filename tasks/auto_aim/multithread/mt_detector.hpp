#ifndef AUTO_AIM__MT_DETECTOR_HPP
#define AUTO_AIM__MT_DETECTOR_HPP

#include <chrono>
#include <cstdint>
#include <memory>
#include <opencv2/opencv.hpp>
#include <tuple>
#include <vector>

#include "tasks/auto_aim/yolo.hpp"
#include "tools/infer/trt_engine.hpp"
#include "tools/logger.hpp"
#include "tools/thread_safe_queue.hpp"

namespace auto_aim
{
namespace multithread
{

class MultiThreadDetector
{
public:
  MultiThreadDetector(const std::string & config_path, bool debug = false);

  void push(cv::Mat img, std::chrono::steady_clock::time_point t);

  std::tuple<std::list<Armor>, std::chrono::steady_clock::time_point> pop();

  std::tuple<cv::Mat, std::list<Armor>, std::chrono::steady_clock::time_point> debug_pop();

private:
  std::unique_ptr<tools::infer::TrtEngine> trt_engine_;
  YOLO yolo_;
  cv::Mat input_buffer_;
  cv::Size letterbox_size_;
  std::vector<std::uint16_t> fp16_input_;

  using QueueItem =
    std::tuple<cv::Mat, std::chrono::steady_clock::time_point, std::vector<float>>;

  tools::ThreadSafeQueue<QueueItem> queue_ {
    16,
    [] { tools::logger()->debug("[MultiThreadDetector] queue is full!"); }};
};

}  // namespace multithread

}  // namespace auto_aim

#endif  // AUTO_AIM__MT_DETECTOR_HPP
