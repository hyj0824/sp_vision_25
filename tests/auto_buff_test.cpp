#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

#include <Eigen/Geometry>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <vector>

#include "tasks/auto_buff/ceres_rune_predictor.hpp"

namespace
{

const std::string keys =
  "{help h usage ? |                       | print this message}"
  "{config-path c  | configs/standard3.yaml | yaml config path}";

template<typename T>
T read_yaml(
  const YAML::Node & yaml, const char * primary, const char * fallback, const T & default_value)
{
  if (yaml[primary]) return yaml[primary].as<T>();
  if (yaml[fallback]) return yaml[fallback].as<T>();
  return default_value;
}

cv::Mat camera_matrix_from_yaml(const YAML::Node & yaml)
{
  const auto data = yaml["camera_matrix"].as<std::vector<double>>();
  cv::Mat camera_matrix(3, 3, CV_64F);
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      camera_matrix.at<double>(r, c) = data[static_cast<std::size_t>(r * 3 + c)];
    }
  }
  return camera_matrix;
}

cv::Mat distort_coeffs_from_yaml(const YAML::Node & yaml)
{
  const auto data = yaml["distort_coeffs"].as<std::vector<double>>();
  cv::Mat distort_coeffs(1, static_cast<int>(data.size()), CV_64F);
  for (std::size_t i = 0; i < data.size(); ++i) {
    distort_coeffs.at<double>(0, static_cast<int>(i)) = data[i];
  }
  return distort_coeffs;
}

}  // namespace

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }

  const auto config_path = cli.get<std::string>("config-path");
  const auto yaml = YAML::LoadFile(config_path);
  auto_buff::CeresRunePredictor predictor(config_path);
  predictor.set_R_gimbal2world(Eigen::Quaterniond::Identity());

  const double target_radius = read_yaml(yaml, "rune_target_radius", "target_radius", 0.145);
  const std::vector<cv::Point3f> object_points = {
    {0.0f, static_cast<float>(+target_radius), 0.0f},
    {static_cast<float>(-target_radius), 0.0f, 0.0f},
    {0.0f, static_cast<float>(-target_radius), 0.0f},
    {static_cast<float>(+target_radius), 0.0f, 0.0f}};

  const cv::Mat camera_matrix = camera_matrix_from_yaml(yaml);
  const cv::Mat distort_coeffs = distort_coeffs_from_yaml(yaml);
  const cv::Vec3d truth_rvec(0.08, -0.04, 0.20);
  const cv::Vec3d truth_tvec(0.04, -0.03, 5.0);

  std::vector<cv::Point2f> image_points;
  cv::projectPoints(
    object_points, truth_rvec, truth_tvec, camera_matrix, distort_coeffs, image_points);

  const auto timestamp = std::chrono::steady_clock::now();
  auto_buff::RuneDetection detection;
  detection.timestamp = timestamp;
  detection.keypoints = image_points;
  detection.confidence = 1.0f;

  auto_buff::RuneObservation observation;
  if (!predictor.solve_pnp(detection, observation)) {
    fmt::print(stderr, "auto_buff_test: solve_pnp failed\n");
    return 1;
  }

  const double t_error = (observation.target_in_camera - Eigen::Vector3d(0.04, -0.03, 5.0)).norm();
  if (!std::isfinite(observation.distance) || observation.target_in_camera.z() <= 0.0 || t_error > 0.1) {
    fmt::print(
      stderr,
      "auto_buff_test: unexpected PnP result, t=({:.4f}, {:.4f}, {:.4f}), distance={:.4f}, "
      "t_error={:.4f}\n",
      observation.target_in_camera.x(), observation.target_in_camera.y(),
      observation.target_in_camera.z(), observation.distance, t_error);
    return 1;
  }

  const auto reprojection = predictor.reproject(observation, observation.target_in_world);
  double max_reprojection_error = 0.0;
  for (std::size_t i = 0; i < reprojection.size(); ++i) {
    max_reprojection_error =
      std::max(max_reprojection_error, static_cast<double>(cv::norm(reprojection[i] - image_points[i])));
  }

  if (max_reprojection_error > 1.0) {
    fmt::print(
      stderr, "auto_buff_test: reprojection error too large: {:.4f}px\n",
      max_reprojection_error);
    return 1;
  }

  const auto updated = predictor.update(detection, timestamp);
  if (!updated.has_value()) {
    fmt::print(stderr, "auto_buff_test: predictor.update failed\n");
    return 1;
  }

  fmt::print(
    "auto_buff_test passed: z={:.3f}m, distance={:.3f}m, reprojection={:.3f}px\n",
    observation.target_in_camera.z(), observation.distance, max_reprojection_error);
  return 0;
}
