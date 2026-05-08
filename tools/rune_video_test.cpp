#include <fmt/core.h>

#include <Eigen/Geometry>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "tasks/auto_buff/ceres_rune_predictor.hpp"

namespace
{

constexpr double kRadToDeg = 180.0 / CV_PI;

struct Options
{
  std::string config = "configs/standard3.yaml";
  std::string video;
  std::string out_dir = "outputs/rune_trt_video_test";
  auto_buff::RuneMode mode = auto_buff::RuneMode::SMALL;
  std::string pose_mode = "identity";
  double bullet_speed = 24.0;
  int max_frames = 0;
  int stride = 1;
  int save_every = 120;
  bool save_video = false;
};

std::string read_next(int argc, char ** argv, int & i, const std::string & name)
{
  if (i + 1 >= argc) throw std::runtime_error("Missing value for " + name);
  return argv[++i];
}

void print_usage(const char * argv0)
{
  fmt::print(
    "Usage: {} --video VIDEO [--config configs/standard3.yaml] "
    "[--out-dir outputs/rune_trt_video_test] [--mode small|big] "
    "[--pose-mode identity|feedback] [--bullet-speed 24] [--max-frames 0] "
    "[--stride 1] [--save-every 120] [--save-video]\n",
    argv0);
}

Options parse_args(int argc, char ** argv)
{
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    } else if (arg == "--config") {
      options.config = read_next(argc, argv, i, arg);
    } else if (arg == "--video") {
      options.video = read_next(argc, argv, i, arg);
    } else if (arg == "--out-dir") {
      options.out_dir = read_next(argc, argv, i, arg);
    } else if (arg == "--mode") {
      const auto mode = read_next(argc, argv, i, arg);
      if (mode == "small") {
        options.mode = auto_buff::RuneMode::SMALL;
      } else if (mode == "big") {
        options.mode = auto_buff::RuneMode::BIG;
      } else {
        throw std::runtime_error("--mode must be small or big");
      }
    } else if (arg == "--pose-mode") {
      options.pose_mode = read_next(argc, argv, i, arg);
      if (options.pose_mode != "identity" && options.pose_mode != "feedback") {
        throw std::runtime_error("--pose-mode must be identity or feedback");
      }
    } else if (arg == "--bullet-speed") {
      options.bullet_speed = std::stod(read_next(argc, argv, i, arg));
    } else if (arg == "--max-frames") {
      options.max_frames = std::stoi(read_next(argc, argv, i, arg));
    } else if (arg == "--stride") {
      options.stride = std::max(1, std::stoi(read_next(argc, argv, i, arg)));
    } else if (arg == "--save-every") {
      options.save_every = std::max(0, std::stoi(read_next(argc, argv, i, arg)));
    } else if (arg == "--save-video") {
      options.save_video = true;
    } else {
      throw std::runtime_error("Unknown argument: " + arg);
    }
  }

  if (options.video.empty()) {
    throw std::runtime_error("--video is required");
  }
  return options;
}

Eigen::Quaterniond q_from_yaw_pitch(double yaw, double pitch)
{
  const Eigen::Matrix3d R =
    Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix() *
    Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()).toRotationMatrix();
  return Eigen::Quaterniond(R).normalized();
}

void draw_detection(cv::Mat & img, const auto_buff::RuneDetection & detection)
{
  std::vector<cv::Point> polygon;
  polygon.reserve(detection.keypoints.size());
  for (std::size_t i = 0; i < detection.keypoints.size(); ++i) {
    const auto & p = detection.keypoints[i];
    polygon.emplace_back(cvRound(p.x), cvRound(p.y));
    cv::circle(img, p, 5, cv::Scalar(0, 0, 255), -1);
    const cv::Point label_origin(cvRound(p.x + 6.0f), cvRound(p.y - 6.0f));
    cv::putText(
      img, std::to_string(i), label_origin, cv::FONT_HERSHEY_SIMPLEX, 0.7,
      cv::Scalar(255, 255, 255), 2);
  }
  if (polygon.size() == 4) {
    cv::polylines(img, polygon, true, cv::Scalar(0, 255, 0), 2);
  }
}

void draw_aim(
  cv::Mat & img, auto_buff::CeresRunePredictor & predictor,
  const auto_buff::RuneObservation & observation, const auto_buff::RuneAimDebug & debug)
{
  if (!debug.valid) return;

  const auto projected = predictor.reproject(observation, debug.predict_target_in_world);
  if (projected.empty()) return;

  cv::Point2f center(0.0f, 0.0f);
  std::vector<cv::Point> polygon;
  polygon.reserve(projected.size());
  for (const auto & p : projected) {
    center += p;
    polygon.emplace_back(cvRound(p.x), cvRound(p.y));
  }
  center *= 1.0f / static_cast<float>(projected.size());
  cv::drawMarker(
    img, center, cv::Scalar(255, 0, 255), cv::MARKER_CROSS, 28, 2, cv::LINE_AA);
  if (polygon.size() == 4) {
    cv::polylines(img, polygon, true, cv::Scalar(255, 0, 255), 2);
  }
}

void draw_text(
  cv::Mat & img, int frame, double time, const std::optional<auto_buff::RuneDetection> & detection,
  const std::optional<auto_buff::RuneObservation> & observation, const io::Command & command,
  const auto_buff::RuneAimDebug & debug)
{
  std::string text = fmt::format("frame={} t={:.2f}s", frame, time);
  if (detection.has_value()) {
    text += fmt::format(" conf={:.3f}", detection->confidence);
  }
  if (observation.has_value()) {
    text += fmt::format(" dist={:.2f}m", observation->distance);
  }
  if (command.control && debug.valid) {
    text += fmt::format(
      " yaw={:.2f} pitch={:.2f}", command.yaw * kRadToDeg, command.pitch * kRadToDeg);
  }
  cv::putText(
    img, text, cv::Point(20, 38), cv::FONT_HERSHEY_SIMPLEX, 0.85, cv::Scalar(0, 255, 255), 2,
    cv::LINE_AA);
}

void write_csv_header(std::ofstream & csv)
{
  csv
    << "frame,time,valid,control,shoot,conf,distance,reproj_error,cam_x,cam_y,cam_z,world_x,world_y,world_z,"
       "yaw_deg,pitch_deg,predict_rotation_deg,fly_time,fit_size,direction,lost_count,"
       "raw_angle_deg,filtered_angle_deg,angle_velocity_deg_s,model_time,fit_angle_deg,"
       "fit_residual_deg,param_a,param_w,param_t0,param_b,param_c,valid_params,"
       "p0x,p0y,p1x,p1y,p2x,p2y,p3x,p3y\n";
}

void write_csv_row(
  std::ofstream & csv, int frame, double time,
  const std::optional<auto_buff::RuneDetection> & detection,
  const std::optional<auto_buff::RuneObservation> & observation, const io::Command & command,
  const auto_buff::RuneAimDebug & debug)
{
  csv << frame << ',' << time << ',' << (observation.has_value() ? 1 : 0) << ','
      << (command.control ? 1 : 0) << ',' << (command.shoot ? 1 : 0) << ',';

  if (detection.has_value()) {
    csv << detection->confidence;
  }
  csv << ',';

  if (observation.has_value()) {
    csv << observation->distance << ',' << observation->reprojection_error << ','
        << observation->target_in_camera.x() << ',' << observation->target_in_camera.y() << ','
        << observation->target_in_camera.z() << ',' << observation->target_in_world.x() << ','
        << observation->target_in_world.y() << ',' << observation->target_in_world.z() << ',';
  } else {
    csv << ",,,,,,,,";
  }

  if (command.control && debug.valid) {
    csv << command.yaw * kRadToDeg << ',' << command.pitch * kRadToDeg << ','
        << debug.predict_rotation * kRadToDeg << ',' << debug.fly_time << ','
        << debug.fit_data_size << ',' << debug.direction << ',' << debug.lost_count << ','
        << debug.raw_relative_angle * kRadToDeg << ',' << debug.filtered_relative_angle * kRadToDeg
        << ',' << debug.angle_velocity * kRadToDeg << ',' << debug.model_time << ','
        << debug.fit_angle * kRadToDeg << ',' << debug.fit_residual * kRadToDeg << ','
        << debug.params[0] << ',' << debug.params[1] << ',' << debug.params[2] << ','
        << debug.params[3] << ',' << debug.params[4] << ',' << (debug.valid_params ? 1 : 0);
  } else {
    csv << ",,,,,,,,,,,,,,,,,,,,";
  }

  if (detection.has_value() && detection->keypoints.size() == 4) {
    for (const auto & p : detection->keypoints) {
      csv << ',' << p.x << ',' << p.y;
    }
  } else {
    csv << ",,,,,,,,";
  }
  csv << '\n';
}

}  // namespace

int main(int argc, char ** argv)
{
  try {
    const auto options = parse_args(argc, argv);

    cv::VideoCapture cap(options.video);
    if (!cap.isOpened()) {
      throw std::runtime_error("Failed to open video: " + options.video);
    }

    const double fps = cap.get(cv::CAP_PROP_FPS) > 1e-3 ? cap.get(cv::CAP_PROP_FPS) : 60.0;
    const int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    const int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    const int total = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

    const std::filesystem::path out_dir(options.out_dir);
    const auto frames_dir = out_dir / "frames";
    std::filesystem::create_directories(frames_dir);

    std::ofstream csv(out_dir / "rune_trt_video.csv");
    if (!csv.is_open()) {
      throw std::runtime_error("Failed to open output CSV.");
    }
    write_csv_header(csv);

    cv::VideoWriter video_writer;
    if (options.save_video) {
      const auto overlay_path = (out_dir / "overlay.mp4").string();
      const int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
      video_writer.open(overlay_path, fourcc, fps / options.stride, cv::Size(width, height));
      if (!video_writer.isOpened()) {
        throw std::runtime_error("Failed to open output video: " + overlay_path);
      }
    }

    auto_buff::RuneDetector detector(options.config);
    auto_buff::CeresRunePredictor predictor(options.config);
    const auto t0 = std::chrono::steady_clock::now();

    double yaw_feedback = 0.0;
    double pitch_feedback = 0.0;
    int processed = 0;
    int valid = 0;
    int control = 0;
    int frame_idx = 0;

    cv::Mat frame;
    while (cap.read(frame)) {
      if (options.max_frames > 0 && processed >= options.max_frames) break;
      if (frame_idx % options.stride != 0) {
        ++frame_idx;
        continue;
      }

      const double video_time = static_cast<double>(frame_idx) / fps;
      const auto timestamp =
        t0 + std::chrono::microseconds(static_cast<int64_t>(video_time * 1e6));

      if (options.pose_mode == "feedback") {
        predictor.set_R_gimbal2world(q_from_yaw_pitch(yaw_feedback, pitch_feedback));
      } else {
        predictor.set_R_gimbal2world(Eigen::Quaterniond::Identity());
      }

      const auto detection = detector.detect(frame, timestamp);
      const auto observation = predictor.update(detection, timestamp);
      const auto command = predictor.aim(options.mode, timestamp, options.bullet_speed, false);
      const auto & debug = predictor.debug();

      if (observation.has_value()) ++valid;
      if (command.control) {
        yaw_feedback = command.yaw;
        pitch_feedback = command.pitch;
        ++control;
      }

      write_csv_row(csv, frame_idx, video_time, detection, observation, command, debug);

      if ((options.save_every > 0 && processed % options.save_every == 0) || options.save_video) {
        cv::Mat vis = frame.clone();
        if (detection.has_value()) draw_detection(vis, detection.value());
        if (observation.has_value()) draw_aim(vis, predictor, observation.value(), debug);
        draw_text(vis, frame_idx, video_time, detection, observation, command, debug);

        if (options.save_every > 0 && processed % options.save_every == 0) {
          const auto image_path = frames_dir / fmt::format("frame_{:06d}.jpg", frame_idx);
          cv::imwrite(image_path.string(), vis);
        }
        if (video_writer.isOpened()) {
          video_writer.write(vis);
        }
      }

      ++processed;
      if (processed % 100 == 0) {
        fmt::print(
          "processed={} valid={} control={} frame={}/{}\n", processed, valid, control, frame_idx,
          total);
      }
      ++frame_idx;
    }

    fmt::print(
      "rune_video_test done: processed={} valid={} control={} out_dir={}\n", processed, valid,
      control, out_dir.string());
    fmt::print("csv={}\n", (out_dir / "rune_trt_video.csv").string());
    fmt::print("frames={}\n", frames_dir.string());
    if (options.save_video) {
      fmt::print("video={}\n", (out_dir / "overlay.mp4").string());
    }
    return 0;
  } catch (const std::exception & e) {
    fmt::print(stderr, "rune_video_test failed: {}\n", e.what());
    return 1;
  }
}
