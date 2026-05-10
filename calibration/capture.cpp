#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <memory>
#include <opencv2/opencv.hpp>
#include <thread>
#include <utility>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

const std::string keys =
  "{help h usage ?  |                          | 输出命令行参数说明}"
  "{@config-path c  | configs/calibration.yaml | yaml配置文件路径 }"
  "{output-folder o |      assets/img_with_q   | 输出文件夹路径   }";

namespace
{
class ChessboardDetector
{
public:
  struct Result
  {
    bool success = false;
    bool timed_out = false;
    std::vector<cv::Point2f> corners;
  };

  explicit ChessboardDetector(const cv::Size & pattern_size, std::chrono::milliseconds timeout)
  : pattern_size_(pattern_size),
    timeout_(timeout),
    busy_(std::make_shared<std::atomic_bool>(false))
  {
  }

  Result detect(const cv::Mat & img)
  {
    Result no_result;
    if (img.empty()) return no_result;

    bool expected = false;
    if (!busy_->compare_exchange_strong(expected, true)) {
      no_result.timed_out = true;
      return no_result;
    }

    auto promise = std::make_shared<std::promise<Result>>();
    auto future = promise->get_future();
    auto busy = busy_;
    auto pattern_size = pattern_size_;
    auto detection_img = img.clone();

    try {
      std::thread([promise, busy, pattern_size, detection_img = std::move(detection_img)]() mutable {
        Result result;
        try {
          constexpr int flags =
            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK;
          result.success =
            cv::findChessboardCorners(detection_img, pattern_size, result.corners, flags);
        } catch (...) {
          result = Result{};
        }

        busy->store(false);
        try {
          promise->set_value(std::move(result));
        } catch (...) {
        }
      }).detach();
    } catch (...) {
      busy_->store(false);
      return no_result;
    }

    if (future.wait_for(timeout_) != std::future_status::ready) {
      no_result.timed_out = true;
      return no_result;
    }

    return future.get();
  }

  const cv::Size pattern_size_;
  const std::chrono::milliseconds timeout_;
  std::shared_ptr<std::atomic_bool> busy_;
};
}  // namespace

void write_q(const std::string q_path, const Eigen::Quaterniond & q)
{
  std::ofstream q_file(q_path);
  Eigen::Vector4d xyzw = q.coeffs();
  // 输出顺序为wxyz
  q_file << fmt::format("{} {} {} {}", xyzw[3], xyzw[0], xyzw[1], xyzw[2]);
  q_file.close();
}

void capture_loop(const std::string & config_path, const std::string & output_folder)
{
  auto yaml = YAML::LoadFile(config_path);
  auto chessboard_corner_cols = yaml["chessboard_corner_cols"].as<int>();
  auto chessboard_corner_rows = yaml["chessboard_corner_rows"].as<int>();
  auto find_chessboard_timeout_ms =
    yaml["find_chessboard_timeout_ms"] ? yaml["find_chessboard_timeout_ms"].as<int>() : 40;
  find_chessboard_timeout_ms = std::max(find_chessboard_timeout_ms, 1);
  cv::Size pattern_size(chessboard_corner_cols, chessboard_corner_rows);
  ChessboardDetector chessboard_detector(pattern_size, std::chrono::milliseconds(find_chessboard_timeout_ms));

  io::Gimbal gimbal(config_path);
  io::Camera camera(config_path);
  cv::Mat img;
  std::chrono::steady_clock::time_point timestamp;

  int count = 0;
  while (true) {
    camera.read(img, timestamp);
    if (img.empty()) continue;

    Eigen::Quaterniond q = gimbal.q(timestamp);

    // 在图像上显示欧拉角，用来判断imuabs系的xyz正方向，同时判断imu是否存在零漂
    auto img_with_ypr = img.clone();
    auto chessboard_result = chessboard_detector.detect(img);

    Eigen::Vector3d zyx = tools::eulers(q, 2, 1, 0) * 57.3;  // degree
    tools::draw_text(img_with_ypr, fmt::format("Z {:.2f}", zyx[0]), {40, 40}, {0, 0, 255});
    tools::draw_text(img_with_ypr, fmt::format("Y {:.2f}", zyx[1]), {40, 80}, {0, 0, 255});
    tools::draw_text(img_with_ypr, fmt::format("X {:.2f}", zyx[2]), {40, 120}, {0, 0, 255});
    if (chessboard_result.timed_out) {
      tools::draw_text(img_with_ypr, "Chessboard timeout", {40, 160}, {0, 0, 255});
    }

    cv::drawChessboardCorners(
      img_with_ypr, pattern_size, chessboard_result.corners, chessboard_result.success);  // 显示识别结果
    cv::resize(img_with_ypr, img_with_ypr, {}, 0.5, 0.5);  // 显示时缩小图片尺寸

    // 按“s”保存图片和对应四元数，按“q”退出程序
    cv::imshow("Press s to save, q to quit", img_with_ypr);
    auto key = cv::waitKey(1);
    if (key == 'q')
      break;
    else if (key != 's')
      continue;

    // 保存图片和四元数
    count++;
    auto img_path = fmt::format("{}/{}.jpg", output_folder, count);
    auto q_path = fmt::format("{}/{}.txt", output_folder, count);
    cv::imwrite(img_path, img);
    write_q(q_path, q);
    tools::logger()->info("[{}] Saved in {}", count, output_folder);
  }

  // 离开该作用域时，camera和gimbal会自动关闭
}

int main(int argc, char * argv[])
{
  // 读取命令行参数
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  auto config_path = cli.get<std::string>(0);
  auto output_folder = cli.get<std::string>("output-folder");

  // 新建输出文件夹
  std::filesystem::create_directory(output_folder);

  // 主循环，保存图片和对应四元数
  capture_loop(config_path, output_folder);

  tools::logger()->warn("注意四元数输出顺序为wxyz");

  return 0;
}
