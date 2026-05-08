#include <fmt/format.h>

#include <chrono>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <string>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_buff/ceres_rune_predictor.hpp"
#include "tasks/auto_buff/rune_detector.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"

const std::string keys =
  "{help h usage ? | | show usage}"
  "{@config-path   | | yaml config path}";

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  tools::Plotter plotter;
  tools::Recorder recorder;
  tools::Exiter exiter;

  io::Gimbal gimbal(config_path);
  io::Camera camera(config_path);

  auto_buff::RuneDetector detector(config_path);
  auto_buff::CeresRunePredictor predictor(config_path);

  cv::Mat img;
  Eigen::Quaterniond q;
  std::chrono::steady_clock::time_point timestamp;

  while (!exiter.exit()) {
    camera.read(img, timestamp);
    q = gimbal.q(timestamp);
    auto gs = gimbal.state();

    predictor.set_R_gimbal2world(q);

    auto detection = detector.detect(img, timestamp);
    auto observation = predictor.update(detection, timestamp);

    const auto rune_mode =
      gimbal.mode() == io::GimbalMode::BIG_BUFF ? auto_buff::RuneMode::BIG : auto_buff::RuneMode::SMALL;
    auto command = predictor.aim(rune_mode, timestamp, gs.bullet_speed, true);
    gimbal.send(command);

    nlohmann::json data;
    data["gimbal_yaw"] = gs.yaw * 57.3;
    data["gimbal_pitch"] = gs.pitch * 57.3;
    data["gimbal_yaw_vel"] = gs.yaw_vel * 57.3;
    data["gimbal_pitch_vel"] = gs.pitch_vel * 57.3;
    data["bullet_speed"] = gs.bullet_speed;

    if (detection.has_value()) {
      data["rune_conf"] = detection->confidence;
      for (std::size_t i = 0; i < detection->keypoints.size(); ++i) {
        tools::draw_point(img, detection->keypoints[i]);
        cv::putText(
          img, std::to_string(i), detection->keypoints[i], cv::FONT_HERSHEY_SIMPLEX, 0.8,
          cv::Scalar(255, 255, 255), 2);
      }
    }

    if (observation.has_value()) {
      data["rune_dis"] = observation->distance;
      data["rune_x"] = observation->target_in_world.x();
      data["rune_y"] = observation->target_in_world.y();
      data["rune_z"] = observation->target_in_world.z();
    }

    const auto & debug = predictor.debug();
    if (debug.valid) {
      data["aim_yaw"] = debug.yaw * 57.3;
      data["aim_pitch"] = debug.pitch * 57.3;
      data["predict_angle"] = debug.predict_rotation * 57.3;
      data["fly_time"] = debug.fly_time;
      data["fit_size"] = debug.fit_data_size;
      data["direction"] = debug.direction;
    }

    if (command.control) {
      data["cmd_yaw"] = command.yaw * 57.3;
      data["cmd_pitch"] = command.pitch * 57.3;
      data["shoot"] = command.shoot ? 1 : 0;
    }

    plotter.plot(data);

    cv::resize(img, img, {}, 0.5, 0.5);
    cv::imshow("result", img);

    auto key = cv::waitKey(1);
    if (key == 'q') break;
  }

  return 0;
}
