#include <fmt/core.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <list>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"

namespace
{
struct BenchmarkFrame
{
  int frame_count;
  cv::Mat img;
  double t;
  double w;
  double x;
  double y;
  double z;
};

struct BenchmarkTiming
{
  double core_s = 0;
  double yolo_s = 0;
  double tracker_s = 0;
  double aimer_s = 0;
};

double percentile(const std::vector<double> & sorted_values, double p)
{
  if (sorted_values.empty()) return 0;

  const auto index =
    static_cast<std::size_t>(p * static_cast<double>(sorted_values.size() - 1));
  return sorted_values[index];
}

std::vector<BenchmarkFrame> load_benchmark_frames(
  cv::VideoCapture & video, std::ifstream & text, int start_index, int end_index)
{
  std::vector<BenchmarkFrame> frames;
  if (!video.isOpened() || !text.is_open()) return frames;

  video.set(cv::CAP_PROP_POS_FRAMES, start_index);
  for (int i = 0; i < start_index; i++) {
    double t, w, x, y, z;
    text >> t >> w >> x >> y >> z;
  }

  for (int frame_count = start_index;; frame_count++) {
    if (end_index > 0 && frame_count > end_index) break;

    cv::Mat img;
    video.read(img);
    if (img.empty()) break;

    double t, w, x, y, z;
    if (!(text >> t >> w >> x >> y >> z)) break;

    frames.push_back({frame_count, img.clone(), t, w, x, y, z});
  }

  return frames;
}

BenchmarkTiming run_benchmark_frame(
  const BenchmarkFrame & frame, auto_aim::YOLO & yolo, auto_aim::Solver & solver,
  auto_aim::Tracker & tracker, auto_aim::Aimer & aimer, io::Command & last_command,
  std::chrono::steady_clock::time_point t0)
{
  auto timestamp = t0 + std::chrono::microseconds(static_cast<int64_t>(frame.t * 1e6));

  auto core_start = std::chrono::steady_clock::now();
  auto yolo_start = std::chrono::steady_clock::now();
  auto armors = yolo.detect(frame.img, frame.frame_count);

  auto tracker_start = std::chrono::steady_clock::now();
  solver.set_R_gimbal2world({frame.w, frame.x, frame.y, frame.z});
  auto targets = tracker.track(armors, timestamp);

  auto aimer_start = std::chrono::steady_clock::now();
  auto command = aimer.aim(targets, timestamp, 27, false);

  if (
    !targets.empty() && aimer.debug_aim_point.valid &&
    std::abs(command.yaw - last_command.yaw) * 57.3 < 2)
    command.shoot = true;

  if (command.control) last_command = command;

  auto finish = std::chrono::steady_clock::now();
  return {
    tools::delta_time(finish, core_start),
    tools::delta_time(tracker_start, yolo_start),
    tools::delta_time(aimer_start, tracker_start),
    tools::delta_time(finish, aimer_start)};
}

void add_timing(BenchmarkTiming & total, const BenchmarkTiming & timing)
{
  total.core_s += timing.core_s;
  total.yolo_s += timing.yolo_s;
  total.tracker_s += timing.tracker_s;
  total.aimer_s += timing.aimer_s;
}

void log_stage_average(const BenchmarkTiming & total, std::size_t frames)
{
  tools::logger()->info(
    "Stage avg: yolo {:.2f} ms, tracker {:.2f} ms, aimer {:.2f} ms",
    total.yolo_s * 1e3 / frames, total.tracker_s * 1e3 / frames,
    total.aimer_s * 1e3 / frames);
}

void log_latency_summary(const std::string & label, std::vector<double> values, double avg_ms)
{
  std::sort(values.begin(), values.end());
  tools::logger()->info(
    "{}: avg {:.2f} ms, p50 {:.2f} ms, p90 {:.2f} ms, p99 {:.2f} ms, "
    "min {:.2f} ms, max {:.2f} ms",
    label, avg_ms, percentile(values, 0.50), percentile(values, 0.90), percentile(values, 0.99),
    values.front(), values.back());
}

}  // namespace

const std::string keys =
  "{help h usage ? |                   | 输出命令行参数说明 }"
  "{config-path c  | configs/demo.yaml | yaml配置文件的路径}"
  "{start-index s  | 0                 | 视频起始帧下标    }"
  "{end-index e    | 0                 | 视频结束帧下标    }"
  "{benchmark b    |                   | 启用benchmark，预加载输入排除IO }"
  "{@input-path    | assets/demo/demo  | avi和txt文件的路径}";

int main(int argc, char * argv[])
{
  // 读取命令行参数
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  auto input_path = cli.get<std::string>(0);
  auto config_path = cli.get<std::string>("config-path");
  auto start_index = cli.get<int>("start-index");
  auto end_index = cli.get<int>("end-index");
  const bool benchmark_mode = cli.has("benchmark");
  const bool enable_visualization = !benchmark_mode && std::getenv("DISPLAY") != nullptr;

  tools::Plotter plotter;
  tools::Exiter exiter;

  auto video_path = fmt::format("{}.avi", input_path);
  auto text_path = fmt::format("{}.txt", input_path);
  cv::VideoCapture video(video_path);
  std::ifstream text(text_path);

  auto_aim::YOLO yolo(config_path, enable_visualization);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Aimer aimer(config_path);

  cv::Mat img, drawing;
  auto t0 = std::chrono::steady_clock::now();

  auto_aim::Target last_target;
  io::Command last_command;
  double last_t = -1;

  if (benchmark_mode) {
    auto preload_start = std::chrono::steady_clock::now();
    auto benchmark_frames = load_benchmark_frames(video, text, start_index, end_index);
    auto preload_finish = std::chrono::steady_clock::now();

    if (benchmark_frames.empty()) {
      tools::logger()->warn("Benchmark input is empty. Check video/text path and frame range.");
      return 1;
    }

    tools::logger()->info(
      "Preloaded {} frames in {:.3f}s. This IO time is excluded from benchmark.",
      benchmark_frames.size(), tools::delta_time(preload_finish, preload_start));

    BenchmarkTiming total_timing;
    std::vector<double> core_latencies_ms;
    core_latencies_ms.reserve(benchmark_frames.size());

    auto benchmark_start = std::chrono::steady_clock::now();
    std::size_t processed_frames = 0;
    for (const auto & frame : benchmark_frames) {
      if (exiter.exit()) break;

      auto timing = run_benchmark_frame(frame, yolo, solver, tracker, aimer, last_command, t0);
      add_timing(total_timing, timing);
      processed_frames++;
      core_latencies_ms.push_back(timing.core_s * 1e3);
    }
    auto benchmark_finish = std::chrono::steady_clock::now();

    if (processed_frames == 0) {
      tools::logger()->warn("Benchmark stopped before processing frames.");
      return 1;
    }

    const auto process_elapsed_s = tools::delta_time(benchmark_finish, benchmark_start);
    tools::logger()->info(
      "Benchmark (IO excluded): {} frames, {:.3f}s, {:.2f} FPS, {:.2f} ms/frame",
      processed_frames, process_elapsed_s, processed_frames / process_elapsed_s,
      process_elapsed_s * 1e3 / processed_frames);
    log_latency_summary(
      "Core compute latency", core_latencies_ms, total_timing.core_s * 1e3 / processed_frames);
    log_stage_average(total_timing, processed_frames);

    return 0;
  }

  video.set(cv::CAP_PROP_POS_FRAMES, start_index);
  for (int i = 0; i < start_index; i++) {
    double t, w, x, y, z;
    text >> t >> w >> x >> y >> z;
  }

  for (int frame_count = start_index; !exiter.exit(); frame_count++) {
    if (end_index > 0 && frame_count > end_index) break;

    video.read(img);
    if (img.empty()) break;

    double t, w, x, y, z;
    text >> t >> w >> x >> y >> z;
    auto timestamp = t0 + std::chrono::microseconds(int(t * 1e6));

    /// 自瞄核心逻辑

    solver.set_R_gimbal2world({w, x, y, z});

    auto yolo_start = std::chrono::steady_clock::now();
    auto armors = yolo.detect(img, frame_count);

    auto tracker_start = std::chrono::steady_clock::now();
    auto targets = tracker.track(armors, timestamp);

    auto aimer_start = std::chrono::steady_clock::now();
    auto command = aimer.aim(targets, timestamp, 27, false);

    if (
      !targets.empty() && aimer.debug_aim_point.valid &&
      std::abs(command.yaw - last_command.yaw) * 57.3 < 2)
      command.shoot = true;

    if (command.control) last_command = command;
    /// 调试输出

    auto finish = std::chrono::steady_clock::now();
    tools::logger()->info(
      "[{}] yolo: {:.1f}ms, tracker: {:.1f}ms, aimer: {:.1f}ms", frame_count,
      tools::delta_time(tracker_start, yolo_start) * 1e3,
      tools::delta_time(aimer_start, tracker_start) * 1e3,
      tools::delta_time(finish, aimer_start) * 1e3);

    tools::draw_text(
      img,
      fmt::format(
        "command is {},{:.2f},{:.2f},shoot:{}", command.control, command.yaw * 57.3,
        command.pitch * 57.3, command.shoot),
      {10, 60}, {154, 50, 205});

    Eigen::Quaternion gimbal_q = {w, x, y, z};
    tools::draw_text(
      img,
      fmt::format(
        "gimbal yaw{:.2f}", (tools::eulers(gimbal_q.toRotationMatrix(), 2, 1, 0) * 57.3)[0]),
      {10, 90}, {255, 255, 255});

    nlohmann::json data;

    // 装甲板原始观测数据
    data["armor_num"] = armors.size();
    if (!armors.empty()) {
      const auto & armor = armors.front();
      data["armor_x"] = armor.xyz_in_world[0];
      data["armor_y"] = armor.xyz_in_world[1];
      data["armor_yaw"] = armor.ypr_in_world[0] * 57.3;
      data["armor_yaw_raw"] = armor.yaw_raw * 57.3;
      data["armor_center_x"] = armor.center_norm.x;
      data["armor_center_y"] = armor.center_norm.y;
    }

    Eigen::Quaternion q{w, x, y, z};
    auto yaw = tools::eulers(q, 2, 1, 0)[0];
    data["gimbal_yaw"] = yaw * 57.3;
    data["cmd_yaw"] = command.yaw * 57.3;
    data["shoot"] = command.shoot;

    if (!targets.empty()) {
      auto target = targets.front();

      if (last_t == -1) {
        last_target = target;
        last_t = t;
        continue;
      }

      std::vector<Eigen::Vector4d> armor_xyza_list;

      // 当前帧target更新后
      armor_xyza_list = target.armor_xyza_list();
      for (const Eigen::Vector4d & xyza : armor_xyza_list) {
        auto image_points =
          solver.reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);
        tools::draw_points(img, image_points, {0, 255, 0});
      }

      // aimer瞄准位置
      auto aim_point = aimer.debug_aim_point;
      Eigen::Vector4d aim_xyza = aim_point.xyza;
      auto image_points =
        solver.reproject_armor(aim_xyza.head(3), aim_xyza[3], target.armor_type, target.name);
      if (aim_point.valid) tools::draw_points(img, image_points, {0, 0, 255});

      // 观测器内部数据
      Eigen::VectorXd x = target.ekf_x();
      data["x"] = x[0];
      data["vx"] = x[1];
      data["y"] = x[2];
      data["vy"] = x[3];
      data["z"] = x[4];
      data["vz"] = x[5];
      data["a"] = x[6] * 57.3;
      data["w"] = x[7];
      data["r"] = x[8];
      data["l"] = x[9];
      data["h"] = x[10];
      data["last_id"] = target.last_id;

      // 卡方检验数据
      data["residual_yaw"] = target.ekf().data.at("residual_yaw");
      data["residual_pitch"] = target.ekf().data.at("residual_pitch");
      data["residual_distance"] = target.ekf().data.at("residual_distance");
      data["residual_angle"] = target.ekf().data.at("residual_angle");
      data["nis"] = target.ekf().data.at("nis");
      data["nees"] = target.ekf().data.at("nees");
      data["nis_fail"] = target.ekf().data.at("nis_fail");
      data["nees_fail"] = target.ekf().data.at("nees_fail");
      data["recent_nis_failures"] = target.ekf().data.at("recent_nis_failures");
    }

    plotter.plot(data);

    if (enable_visualization) {
      cv::resize(img, img, {}, 0.5, 0.5);  // 显示时缩小图片尺寸
      cv::imshow("reprojection", img);
      auto key = cv::waitKey(30);
      if (key == 'q') break;
    }
  }

  return 0;
}
