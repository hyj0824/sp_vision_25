#include <fmt/core.h>

#include <Eigen/Geometry>
#include <chrono>
#include <cctype>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "tasks/auto_buff/ceres_rune_predictor.hpp"

namespace
{

std::vector<std::string> split_csv_line(const std::string & line)
{
  std::vector<std::string> fields;
  std::stringstream ss(line);
  std::string field;
  while (std::getline(ss, field, ',')) {
    while (!field.empty() && std::isspace(static_cast<unsigned char>(field.back()))) {
      field.pop_back();
    }
    while (!field.empty() && std::isspace(static_cast<unsigned char>(field.front()))) {
      field.erase(field.begin());
    }
    fields.push_back(field);
  }
  if (!line.empty() && line.back() == ',') {
    fields.emplace_back();
  }
  return fields;
}

std::unordered_map<std::string, std::size_t> header_index(const std::vector<std::string> & header)
{
  std::unordered_map<std::string, std::size_t> index;
  for (std::size_t i = 0; i < header.size(); ++i) {
    index.emplace(header[i], i);
  }
  return index;
}

const std::string & value(
  const std::vector<std::string> & row, const std::unordered_map<std::string, std::size_t> & index,
  const std::string & key)
{
  const auto it = index.find(key);
  if (it == index.end() || it->second >= row.size()) {
    throw std::runtime_error("CSV is missing field: " + key);
  }
  return row[it->second];
}

double number(
  const std::vector<std::string> & row, const std::unordered_map<std::string, std::size_t> & index,
  const std::string & key)
{
  const auto & text = value(row, index, key);
  if (text.empty()) return 0.0;
  return std::stod(text);
}

Eigen::Quaterniond q_from_yaw_pitch(double yaw, double pitch)
{
  const Eigen::Matrix3d R =
    Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix() *
    Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()).toRotationMatrix();
  return Eigen::Quaterniond(R).normalized();
}

struct Options
{
  std::string config = "configs/standard3.yaml";
  std::string csv;
  std::string output = "outputs/rune_offline_cpp_predict.csv";
  auto_buff::RuneMode mode = auto_buff::RuneMode::SMALL;
  std::string pose_mode = "feedback";
  double bullet_speed = 24.0;
};

Options parse_args(int argc, char ** argv)
{
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto read_next = [&](const std::string & name) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("Missing value for " + name);
      return argv[++i];
    };

    if (arg == "--config") {
      options.config = read_next(arg);
    } else if (arg == "--csv") {
      options.csv = read_next(arg);
    } else if (arg == "--output") {
      options.output = read_next(arg);
    } else if (arg == "--mode") {
      const auto mode = read_next(arg);
      if (mode == "small") {
        options.mode = auto_buff::RuneMode::SMALL;
      } else if (mode == "big") {
        options.mode = auto_buff::RuneMode::BIG;
      } else {
        throw std::runtime_error("--mode must be small or big");
      }
    } else if (arg == "--pose-mode") {
      options.pose_mode = read_next(arg);
      if (options.pose_mode != "feedback" && options.pose_mode != "identity") {
        throw std::runtime_error("--pose-mode must be feedback or identity");
      }
    } else if (arg == "--bullet-speed") {
      options.bullet_speed = std::stod(read_next(arg));
    } else if (arg == "--help" || arg == "-h") {
      fmt::print(
        "Usage: {} --csv detections.csv [--config configs/standard3.yaml] "
        "[--output out.csv] [--mode small|big] [--pose-mode feedback|identity]\n",
        argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown argument: " + arg);
    }
  }

  if (options.csv.empty()) {
    throw std::runtime_error("--csv is required");
  }
  return options;
}

}  // namespace

int main(int argc, char ** argv)
{
  try {
    const auto options = parse_args(argc, argv);
    std::ifstream input(options.csv);
    if (!input.is_open()) {
      throw std::runtime_error("Failed to open CSV: " + options.csv);
    }

    std::ofstream output(options.output);
    if (!output.is_open()) {
      throw std::runtime_error("Failed to open output CSV: " + options.output);
    }

    std::string line;
    if (!std::getline(input, line)) {
      throw std::runtime_error("Empty CSV: " + options.csv);
    }
    const auto header = split_csv_line(line);
    const auto index = header_index(header);

    auto_buff::CeresRunePredictor predictor(options.config);
    const auto t0 = std::chrono::steady_clock::now();
    double yaw_feedback = 0.0;
    double pitch_feedback = 0.0;

    output
      << "frame,time,valid,control,shoot,distance,world_x,world_y,world_z,yaw_deg,pitch_deg,"
         "predict_rotation_deg,fly_time,fit_size,direction\n";

    int rows = 0;
    int valid_rows = 0;
    int control_rows = 0;
    while (std::getline(input, line)) {
      if (line.empty()) continue;
      const auto row = split_csv_line(line);
      const int frame = static_cast<int>(number(row, index, "frame"));
      const double time = number(row, index, "time");
      const bool valid = value(row, index, "valid") == "1";
      const auto timestamp = t0 + std::chrono::microseconds(static_cast<int64_t>(time * 1e6));

      if (options.pose_mode == "feedback") {
        predictor.set_R_gimbal2world(q_from_yaw_pitch(yaw_feedback, pitch_feedback));
      } else {
        predictor.set_R_gimbal2world(Eigen::Quaterniond::Identity());
      }

      std::optional<auto_buff::RuneDetection> detection;
      if (valid) {
        auto_buff::RuneDetection current;
        current.timestamp = timestamp;
        current.confidence = static_cast<float>(number(row, index, "conf"));
        current.keypoints = {
          cv::Point2f(number(row, index, "p0x"), number(row, index, "p0y")),
          cv::Point2f(number(row, index, "p1x"), number(row, index, "p1y")),
          cv::Point2f(number(row, index, "p2x"), number(row, index, "p2y")),
          cv::Point2f(number(row, index, "p3x"), number(row, index, "p3y"))};
        detection = current;
      }

      const auto observation = predictor.update(detection, timestamp);
      const auto command = predictor.aim(options.mode, timestamp, options.bullet_speed, false);
      const auto & debug = predictor.debug();
      if (command.control) {
        yaw_feedback = command.yaw;
        pitch_feedback = command.pitch;
        ++control_rows;
      }
      if (observation.has_value()) ++valid_rows;

      output << frame << ',' << time << ',' << (observation.has_value() ? 1 : 0) << ','
             << (command.control ? 1 : 0) << ',' << (command.shoot ? 1 : 0) << ',';
      if (observation.has_value()) {
        output << observation->distance << ',' << observation->target_in_world.x() << ','
               << observation->target_in_world.y() << ',' << observation->target_in_world.z()
               << ',';
      } else {
        output << ",,,,";
      }
      if (command.control && debug.valid) {
        output << command.yaw * 57.3 << ',' << command.pitch * 57.3 << ','
               << debug.predict_rotation * 57.3 << ',' << debug.fly_time << ','
               << debug.fit_data_size << ',' << debug.direction;
      } else {
        output << ",,,,,";
      }
      output << '\n';
      ++rows;
    }

    fmt::print(
      "offline rune predict done: rows={} valid={} control={} output={}\n", rows, valid_rows,
      control_rows, options.output);
    return 0;
  } catch (const std::exception & e) {
    fmt::print(stderr, "offline rune predict failed: {}\n", e.what());
    return 1;
  }
}
