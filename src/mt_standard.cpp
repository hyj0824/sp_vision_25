#include <atomic>
#include <chrono>
#include <fmt/format.h>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/multithread/commandgener.hpp"
#include "tasks/auto_aim/shooter.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tasks/auto_buff/ceres_rune_predictor.hpp"
#include "tasks/auto_buff/rune_detector.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"

const std::string keys =
  "{help h usage ? | | 输出命令行参数说明}"
  "{@config-path   | | yaml配置文件路径 }";

using namespace std::chrono_literals;

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>("@config-path");
  if (cli.has("help") || !cli.has("@config-path")) {
    cli.printMessage();
    return 0;
  }

  tools::Exiter exiter;
  tools::Plotter plotter;
  tools::Recorder recorder;

  io::Camera camera(config_path);
  io::Gimbal gimbal(config_path);

  auto_aim::YOLO yolo(config_path, false);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Aimer aimer(config_path);
  auto_aim::Shooter shooter(config_path);

  auto_buff::RuneDetector rune_detector(config_path);
  auto_buff::CeresRunePredictor rune_predictor(config_path);

  auto_aim::multithread::CommandGener commandgener(shooter, aimer, gimbal, plotter);

  const auto yaml = YAML::LoadFile(config_path);
  const bool rune_show_window = yaml["rune_show_window"] && yaml["rune_show_window"].as<bool>();
  std::atomic<io::GimbalMode> mode{io::GimbalMode::IDLE};
  auto last_mode{io::GimbalMode::IDLE};
  bool rune_window_ok = rune_show_window;

  while (!exiter.exit()) {
    mode = gimbal.mode();

    if (last_mode != mode) {
      tools::logger()->info("Switch to {}", gimbal.str(mode));
      last_mode = mode.load();
    }

    /// 自瞄
    auto gs = gimbal.state();
    if (mode.load() == io::GimbalMode::AUTO_AIM) {
      cv::Mat img;
      std::chrono::steady_clock::time_point t;
      camera.read(img, t);
      auto armors = yolo.detect(img);
      Eigen::Quaterniond q = gimbal.q(t - 1ms);

      // recorder.record(img, q, t);

      solver.set_R_gimbal2world(q);

      Eigen::Vector3d ypr = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);

      auto targets = tracker.track(armors, t);

      commandgener.push(targets, t, gs.bullet_speed, ypr);  // 发送给决策线程

    }

    /// 打符
    else if (mode.load() == io::GimbalMode::SMALL_BUFF || mode.load() == io::GimbalMode::BIG_BUFF) {
      cv::Mat img;
      Eigen::Quaterniond q;
      std::chrono::steady_clock::time_point t;

      camera.read(img, t);
      q = gimbal.q(t - 1ms);

      // recorder.record(img, q, t);

      rune_predictor.set_R_gimbal2world(q);
      auto detection = rune_detector.detect(img, t);
      rune_predictor.update(detection, t);
      const auto rune_mode =
        mode.load() == io::GimbalMode::BIG_BUFF ? auto_buff::RuneMode::BIG : auto_buff::RuneMode::SMALL;
      auto command = rune_predictor.aim(rune_mode, t, gs.bullet_speed, true);
      gimbal.send(command);

      if (rune_window_ok) {
        if (detection.has_value()) {
          for (std::size_t i = 0; i < detection->keypoints.size(); ++i) {
            tools::draw_point(img, detection->keypoints[i]);
            cv::putText(
              img, std::to_string(i), detection->keypoints[i], cv::FONT_HERSHEY_SIMPLEX, 0.8,
              cv::Scalar(255, 255, 255), 2);
          }
          const cv::Point label_pos{
            static_cast<int>(detection->rect.x), static_cast<int>(detection->rect.y)};
          cv::putText(
            img, fmt::format("rune {:.2f}", detection->confidence), label_pos,
            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
        }
        const auto & debug = rune_predictor.debug();
        cv::putText(
          img,
          fmt::format(
            "buff {} ctrl:{} shoot:{} yaw:{:.2f} pitch:{:.2f}",
            rune_mode == auto_buff::RuneMode::BIG ? "BIG" : "SMALL", command.control ? 1 : 0,
            command.shoot ? 1 : 0, command.yaw * 57.3, command.pitch * 57.3),
          {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
        if (debug.valid) {
          cv::putText(
            img,
            fmt::format(
              "pred:{:.2f} fly:{:.3f} fit:{} dir:{}", debug.predict_rotation * 57.3,
              debug.fly_time, debug.fit_data_size, debug.direction),
            {10, 60}, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
        }

        cv::resize(img, img, {}, 0.5, 0.5);
        try {
          cv::imshow("rune", img);
          cv::waitKey(1);
        } catch (const cv::Exception & e) {
          tools::logger()->warn("Disable rune camera window: {}", e.what());
          rune_window_ok = false;
        }
      }

    } else
      continue;
  }

  return 0;
}
