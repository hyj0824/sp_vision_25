#include <atomic>
#include <chrono>
#include <opencv2/opencv.hpp>

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

  std::atomic<io::GimbalMode> mode{io::GimbalMode::IDLE};
  auto last_mode{io::GimbalMode::IDLE};

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

    } else
      continue;
  }

  return 0;
}
