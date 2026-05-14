#include "ceres_rune_predictor.hpp"

#include <ceres/ceres.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <stdexcept>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/trajectory.hpp"

namespace auto_buff
{
namespace
{

constexpr double kAngleBetweenFanBlades = 72.0 * CV_PI / 180.0;

bool is_finite_point(const std::pair<double, double> & point)
{
  return std::isfinite(point.first) && std::isfinite(point.second);
}

std::vector<std::pair<double, double>> sanitized(std::vector<std::pair<double, double>> data)
{
  data.erase(
    std::remove_if(data.begin(), data.end(), [](const auto & point) {
      return !is_finite_point(point);
    }),
    data.end());
  return data;
}

template<typename T>
T read_yaml(
  const YAML::Node & yaml, const char * primary, const char * fallback, const T & default_value)
{
  if (yaml[primary]) return yaml[primary].as<T>();
  if (yaml[fallback]) return yaml[fallback].as<T>();
  return default_value;
}

std::string read_yaml_string(
  const YAML::Node & yaml, const char * primary, const char * fallback, const std::string & default_value)
{
  if (yaml[primary]) return yaml[primary].as<std::string>();
  if (yaml[fallback]) return yaml[fallback].as<std::string>();
  return default_value;
}

class PriorCost final : public ceres::SizedCostFunction<1, 5>
{
public:
  PriorCost(double truth, int id) : truth_(truth), id_(id) {}

  bool Evaluate(double const * const * parameters, double * residuals, double ** jacobians)
    const override
  {
    residuals[0] = parameters[0][id_] - truth_;
    if (jacobians != nullptr && jacobians[0] != nullptr) {
      for (int i = 0; i < 5; ++i) {
        jacobians[0][i] = (i == id_) ? 1.0 : 0.0;
      }
    }
    return true;
  }

private:
  double truth_;
  int id_;
};

class FitCost final : public ceres::SizedCostFunction<1, 5>
{
public:
  FitCost(double t, double y) : t_(t), y_(y) {}

  bool Evaluate(double const * const * parameters, double * residuals, double ** jacobians)
    const override
  {
    const double a = parameters[0][0];
    const double w = parameters[0][1];
    const double t0 = parameters[0][2];
    const double b = parameters[0][3];
    const double c = parameters[0][4];

    if (
      !std::isfinite(t_) || !std::isfinite(y_) || !std::isfinite(a) || !std::isfinite(w) ||
      !std::isfinite(t0) || !std::isfinite(b) || !std::isfinite(c))
    {
      residuals[0] = 1e6;
      if (jacobians != nullptr && jacobians[0] != nullptr) {
        std::fill(jacobians[0], jacobians[0] + 5, 0.0);
      }
      return true;
    }

    const double phase = w * (t_ + t0);
    const double cs = std::cos(phase);
    const double sn = std::sin(phase);
    residuals[0] = -a * cs + b * t_ + c - y_;
    if (!std::isfinite(residuals[0])) residuals[0] = 1e6;

    if (jacobians != nullptr && jacobians[0] != nullptr) {
      jacobians[0][0] = -cs;
      jacobians[0][1] = a * (t_ + t0) * sn;
      jacobians[0][2] = a * w * sn;
      jacobians[0][3] = t_;
      jacobians[0][4] = 1.0;
      for (int i = 0; i < 5; ++i) {
        if (!std::isfinite(jacobians[0][i])) jacobians[0][i] = 0.0;
      }
    }
    return true;
  }

private:
  double t_;
  double y_;
};

}  // namespace

CeresRunePredictor::CeresRunePredictor(const std::string & config_path)
{
  const auto yaml = YAML::LoadFile(config_path);

  const auto R_gimbal2imubody_data = yaml["R_gimbal2imubody"].as<std::vector<double>>();
  const auto R_camera2gimbal_data = yaml["R_camera2gimbal"].as<std::vector<double>>();
  const auto t_camera2gimbal_data = yaml["t_camera2gimbal"].as<std::vector<double>>();
  R_gimbal2imubody_ =
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(R_gimbal2imubody_data.data());
  R_camera2gimbal_ =
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(R_camera2gimbal_data.data());
  t_camera2gimbal_ = Eigen::Matrix<double, 3, 1>(t_camera2gimbal_data.data());

  const auto camera_matrix_data = yaml["camera_matrix"].as<std::vector<double>>();
  const auto distort_coeffs_data = yaml["distort_coeffs"].as<std::vector<double>>();
  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> camera_matrix(camera_matrix_data.data());
  cv::eigen2cv(camera_matrix, camera_matrix_);
  distort_coeffs_ =
    cv::Mat(1, static_cast<int>(distort_coeffs_data.size()), CV_64F, cv::Scalar(0.0));
  for (std::size_t i = 0; i < distort_coeffs_data.size(); ++i) {
    distort_coeffs_.at<double>(0, static_cast<int>(i)) = distort_coeffs_data[i];
  }

  yaw_offset_ = read_yaml(yaml, "rune_yaw_offset", "yaw_offset", 0.0) / 57.3;
  pitch_offset_ = read_yaml(yaml, "rune_pitch_offset", "pitch_offset", 0.0) / 57.3;
  delay_time_ = read_yaml(yaml, "rune_delay_time", "predict_time", 0.1);
  if (yaml["delay"]) delay_time_ = yaml["delay"].as<double>();
  fire_gap_time_ = read_yaml(yaml, "rune_fire_gap_time", "fire_gap_time", 0.6);
  fan_len_ = read_yaml(yaml, "rune_fan_len", "fan_len", 0.7);
  target_radius_ = read_yaml(yaml, "rune_target_radius", "target_radius", 0.145);
  small_rune_speed_ = read_yaml(yaml, "rune_small_speed", "small_rune_speed", CV_PI / 3.0);
  stale_reset_time_ = read_yaml(yaml, "rune_stale_reset_time", "stale_reset_time", 2.0);
  center_fallback_time_ = read_yaml(yaml, "rune_center_fallback_time", "center_fallback_time", 0.1);
  auto_fire_ = read_yaml(yaml, "rune_auto_fire", "auto_fire", true);
  lost_count_threshold_ = read_yaml(yaml, "rune_lost_count", "lostCnt", 20);
  direction_window_ = read_yaml(yaml, "rune_direction_window", "direction_window", 10);
  min_fit_data_size_ = read_yaml(yaml, "rune_min_fit_data_size", "min_fit_data_size", 20);
  max_fit_data_size_ = read_yaml(yaml, "rune_max_fit_data_size", "max_fit_data_size", 1200);
  min_shooting_size_ = read_yaml(yaml, "rune_min_shooting_size", "min_shooting_size", 30);
  fit_interval_frames_ = read_yaml(yaml, "rune_fit_interval_frames", "fit_interval_frames", 3);
  Qw_ = read_yaml(yaml, "rune_Qw", "Qw", 0.2);
  Qtheta_ = read_yaml(yaml, "rune_Qtheta", "Qtheta", 0.08);
  Rtheta_ = read_yaml(yaml, "rune_Rtheta", "Rtheta", 0.4);
  reject_pnp_outliers_ = read_yaml(yaml, "rune_reject_pnp_outliers", "reject_pnp_outliers", true);
  log_rejected_pnp_ = read_yaml(yaml, "rune_log_rejected_pnp", "log_rejected_pnp", false);
  fit_use_raw_angle_ = read_yaml(yaml, "rune_fit_use_raw_angle", "fit_use_raw_angle", false);
  const auto angle_filter_name =
    read_yaml_string(yaml, "rune_angle_filter", "angle_filter", "simple");
  if (angle_filter_name == "legacy_ekf" || angle_filter_name == "legacy") {
    angle_filter_mode_ = AngleFilterMode::LEGACY_EKF;
  } else if (angle_filter_name == "simple" || angle_filter_name == "linear") {
    angle_filter_mode_ = AngleFilterMode::SIMPLE;
  } else {
    throw std::runtime_error(
      "Unsupported rune_angle_filter: " + angle_filter_name + " (expected simple or legacy_ekf)");
  }
  pnp_min_distance_ = read_yaml(yaml, "rune_pnp_min_distance", "pnp_min_distance", 1.0);
  pnp_max_distance_ = read_yaml(yaml, "rune_pnp_max_distance", "pnp_max_distance", 15.0);
  pnp_min_depth_ = read_yaml(yaml, "rune_pnp_min_depth", "pnp_min_depth", 0.5);
  pnp_max_reprojection_error_ =
    read_yaml(yaml, "rune_pnp_max_reprojection_error", "pnp_max_reprojection_error", 8.0);
  pnp_max_distance_jump_ =
    read_yaml(yaml, "rune_pnp_max_distance_jump", "pnp_max_distance_jump", 2.5);
  pnp_max_angle_rate_ = read_yaml(yaml, "rune_pnp_max_angle_rate", "pnp_max_angle_rate", 8.0);
  pnp_angle_gate_max_gap_ =
    read_yaml(yaml, "rune_pnp_angle_gate_max_gap", "pnp_angle_gate_max_gap", 0.25);

  fit_data_.reserve(static_cast<std::size_t>(std::max(max_fit_data_size_, min_fit_data_size_)));
  last_fire_time_ = std::chrono::steady_clock::now();
}

void CeresRunePredictor::set_R_gimbal2world(const Eigen::Quaterniond & q)
{
  const Eigen::Matrix3d R_imubody2imuabs = q.toRotationMatrix();
  R_gimbal2world_ = R_gimbal2imubody_.transpose() * R_imubody2imuabs * R_gimbal2imubody_;
}

std::optional<RuneObservation> CeresRunePredictor::update(
  const std::optional<RuneDetection> & detection, std::chrono::steady_clock::time_point timestamp)
{
  if (!detection.has_value()) {
    ++lost_count_;
    if (
      latest_observation_.has_value() &&
      tools::delta_time(timestamp, latest_observation_->timestamp) > stale_reset_time_)
    {
      reset(timestamp);
    }
    return std::nullopt;
  }

  RuneObservation observation;
  if (!solve_pnp(detection.value(), observation)) {
    ++lost_count_;
    return std::nullopt;
  }

  std::string reject_reason;
  if (reject_pnp_outliers_ && reject_observation(observation, reject_reason)) {
    ++lost_count_;
    if (log_rejected_pnp_) {
      tools::logger()->debug("Reject rune PnP observation: {}", reject_reason);
    }
    return std::nullopt;
  }

  lost_count_ = 0;
  if (first_detect_) {
    reset(observation.timestamp);
    set_first_detect(observation);
  }

  update_angle(observation);
  latest_observation_ = observation;
  last_detection_time_ = observation.timestamp;
  return observation;
}

io::Command CeresRunePredictor::aim(
  RuneMode mode, std::chrono::steady_clock::time_point timestamp, double bullet_speed, bool to_now)
{
  debug_ = {};
  debug_.mode = mode;
  if (bullet_speed < 10.0 || bullet_speed > 35.0) bullet_speed = 24.0;

  if (!latest_observation_.has_value()) {
    return {false, false, 0.0, 0.0};
  }

  const double stale_time = tools::delta_time(timestamp, latest_observation_->timestamp);
  if (stale_time > stale_reset_time_) {
    reset(timestamp);
    return {false, false, 0.0, 0.0};
  }

  if (
    lost_count_ > 0 && stale_time > center_fallback_time_ &&
    center_positions_.size() >= static_cast<std::size_t>(std::max(1, direction_window_)))
  {
    double fly_time = 0.0;
    const Eigen::Vector3d center = fallback_center();
    const auto yaw_pitch = solve_yaw_pitch(center, bullet_speed, fly_time);
    if (!yaw_pitch.has_value()) return {false, false, 0.0, 0.0};
    debug_.valid = true;
    debug_.center_in_world = center;
    debug_.predict_target_in_world = center;
    debug_.fly_time = fly_time;
    debug_.yaw = yaw_pitch->x();
    debug_.pitch = yaw_pitch->y();
    debug_.lost_count = lost_count_;
    fill_debug_model(tools::delta_time(latest_observation_->timestamp, start_time_));
    return {true, false, yaw_pitch->x(), yaw_pitch->y()};
  }

  if (direction_ == Direction::UNKNOWN) {
    return {false, false, 0.0, 0.0};
  }

  const auto & observation = latest_observation_.value();
  const double detect_to_now = to_now ? std::max(0.0, tools::delta_time(timestamp, observation.timestamp)) : 0.0;
  const double current_time = tools::delta_time(observation.timestamp, start_time_);

  double fly_time = 0.0;
  double rotation = 0.0;
  Eigen::Vector3d aim_world = observation.target_in_world;

  for (int iter = 0; iter < 3; ++iter) {
    const auto yaw_pitch = solve_yaw_pitch(aim_world, bullet_speed, fly_time);
    if (!yaw_pitch.has_value()) return {false, false, 0.0, 0.0};

    const double future_dt = detect_to_now + fly_time + delay_time_;
    if (direction_ == Direction::STABLE) {
      rotation = 0.0;
    } else if (mode == RuneMode::BIG) {
      if (!valid_params_) return {false, false, 0.0, 0.0};
      rotation = get_big_rotation(current_time, future_dt);
    } else {
      rotation = get_small_rotation(future_dt);
    }
    aim_world = predict_point_in_world(observation, rotation);
  }

  const auto yaw_pitch = solve_yaw_pitch(aim_world, bullet_speed, fly_time);
  if (!yaw_pitch.has_value()) return {false, false, 0.0, 0.0};

  bool fire = false;
  const bool big_ready =
    mode == RuneMode::SMALL ||
    fit_data_.size() > static_cast<std::size_t>(std::max(min_shooting_size_, min_fit_data_size_));
  if (
    auto_fire_ && big_ready &&
    tools::delta_time(timestamp, last_fire_time_) > fire_gap_time_)
  {
    fire = true;
    last_fire_time_ = timestamp;
  }

  debug_.valid = true;
  debug_.current_target_in_world = observation.target_in_world;
  debug_.predict_target_in_world = aim_world;
  debug_.center_in_world = predict_point_in_world(observation, 0.0) +
                           observation.R_target2world * Eigen::Vector3d(0.0, fan_len_, 0.0);
  debug_.relative_angle = angle_rel_;
  debug_.raw_relative_angle = raw_angle_rel_;
  debug_.filtered_relative_angle = angle_rel_;
  debug_.angle_velocity = angle_state_(0);
  debug_.predict_rotation = rotation;
  debug_.fly_time = fly_time;
  debug_.yaw = yaw_pitch->x();
  debug_.pitch = yaw_pitch->y();
  debug_.direction = static_cast<int>(direction_sign());
  debug_.lost_count = lost_count_;
  fill_debug_model(current_time);

  return {true, fire, yaw_pitch->x(), yaw_pitch->y()};
}

io::Command CeresRunePredictor::aim_static(
  std::chrono::steady_clock::time_point timestamp, double bullet_speed, bool fire)
{
  debug_ = {};
  debug_.mode = RuneMode::SMALL;
  if (bullet_speed < 10.0 || bullet_speed > 35.0) bullet_speed = 24.0;

  if (!latest_observation_.has_value()) {
    return {false, false, 0.0, 0.0};
  }

  const auto & observation = latest_observation_.value();
  const double stale_time = tools::delta_time(timestamp, observation.timestamp);
  if (stale_time > stale_reset_time_) {
    reset(timestamp);
    return {false, false, 0.0, 0.0};
  }

  double fly_time = 0.0;
  const auto yaw_pitch = solve_yaw_pitch(observation.target_in_world, bullet_speed, fly_time);
  if (!yaw_pitch.has_value()) return {false, false, 0.0, 0.0};

  debug_.valid = true;
  debug_.current_target_in_world = observation.target_in_world;
  debug_.predict_target_in_world = observation.target_in_world;
  debug_.relative_angle = angle_rel_;
  debug_.raw_relative_angle = raw_angle_rel_;
  debug_.filtered_relative_angle = angle_rel_;
  debug_.angle_velocity = angle_state_(0);
  debug_.predict_rotation = 0.0;
  debug_.fly_time = fly_time;
  debug_.yaw = yaw_pitch->x();
  debug_.pitch = yaw_pitch->y();
  debug_.direction = static_cast<int>(direction_sign());
  debug_.lost_count = lost_count_;
  fill_debug_model(tools::delta_time(observation.timestamp, start_time_));

  bool shoot = false;
  if (fire && tools::delta_time(timestamp, last_fire_time_) > fire_gap_time_) {
    shoot = true;
    last_fire_time_ = timestamp;
  }

  return {true, shoot, yaw_pitch->x(), yaw_pitch->y()};
}

bool CeresRunePredictor::solve_pnp(
  const RuneDetection & detection, RuneObservation & observation) const
{
  if (detection.keypoints.size() != 4) return false;

  const std::vector<cv::Point3d> object_points = {
    {0.0, +target_radius_, 0.0},
    {-target_radius_, 0.0, 0.0},
    {0.0, -target_radius_, 0.0},
    {+target_radius_, 0.0, 0.0}};

  cv::Vec3d rvec;
  cv::Vec3d tvec;
  const bool ok = cv::solvePnP(
    object_points, detection.keypoints, camera_matrix_, distort_coeffs_, rvec, tvec, false,
    cv::SOLVEPNP_IPPE);
  if (!ok) return false;

  for (int i = 0; i < 3; ++i) {
    if (!std::isfinite(rvec[i]) || !std::isfinite(tvec[i])) return false;
  }

  observation.timestamp = detection.timestamp;
  observation.keypoints = detection.keypoints;
  observation.target_in_camera = Eigen::Vector3d(tvec[0], tvec[1], tvec[2]);
  observation.target_in_gimbal = R_camera2gimbal_ * observation.target_in_camera + t_camera2gimbal_;
  observation.target_in_world = R_gimbal2world_ * observation.target_in_gimbal;
  observation.distance = observation.target_in_world.norm();

  cv::Mat rmat_cv;
  cv::Rodrigues(rvec, rmat_cv);
  cv::cv2eigen(rmat_cv, observation.R_target2camera);
  observation.R_target2gimbal = R_camera2gimbal_ * observation.R_target2camera;
  observation.R_target2world = R_gimbal2world_ * observation.R_target2gimbal;

  std::vector<cv::Point2d> projected_points;
  cv::projectPoints(object_points, rvec, tvec, camera_matrix_, distort_coeffs_, projected_points);
  double squared_error = 0.0;
  for (std::size_t i = 0; i < projected_points.size(); ++i) {
    const cv::Point2d detected(detection.keypoints[i].x, detection.keypoints[i].y);
    const cv::Point2d diff = projected_points[i] - detected;
    squared_error += diff.x * diff.x + diff.y * diff.y;
  }
  observation.reprojection_error =
    std::sqrt(squared_error / std::max<std::size_t>(1, projected_points.size()));

  return true;
}

std::vector<cv::Point2f> CeresRunePredictor::reproject(
  const RuneObservation & observation, const Eigen::Vector3d & target_in_world) const
{
  const Eigen::Matrix3d R_target2camera =
    R_camera2gimbal_.transpose() * R_gimbal2world_.transpose() * observation.R_target2world;
  const Eigen::Vector3d target_in_camera =
    R_camera2gimbal_.transpose() * (R_gimbal2world_.transpose() * target_in_world - t_camera2gimbal_);

  cv::Mat rmat_cv;
  cv::eigen2cv(R_target2camera, rmat_cv);
  cv::Vec3d rvec;
  cv::Rodrigues(rmat_cv, rvec);
  cv::Vec3d tvec(target_in_camera.x(), target_in_camera.y(), target_in_camera.z());

  const std::vector<cv::Point3f> object_points = {
    {0.0f, static_cast<float>(+target_radius_), 0.0f},
    {static_cast<float>(-target_radius_), 0.0f, 0.0f},
    {0.0f, static_cast<float>(-target_radius_), 0.0f},
    {static_cast<float>(+target_radius_), 0.0f, 0.0f}};
  std::vector<cv::Point2f> image_points;
  cv::projectPoints(object_points, rvec, tvec, camera_matrix_, distort_coeffs_, image_points);
  return image_points;
}

void CeresRunePredictor::reset(std::chrono::steady_clock::time_point timestamp)
{
  first_detect_ = true;
  angle_filter_initialized_ = false;
  lost_count_ = 0;
  total_shift_ = 0;
  direction_ = Direction::UNKNOWN;
  convexity_ = Convexity::UNKNOWN;
  angle_abs_last_ = 0.0;
  angle_rel_ = 0.0;
  raw_angle_rel_ = 0.0;
  start_time_ = timestamp;
  last_detection_time_ = timestamp;
  fit_data_.clear();
  direction_data_.clear();
  center_positions_.clear();
  valid_params_ = false;
  params_ = {0.470, 1.942, 0.0, 1.178, 0.0};
  angle_state_ = Eigen::Vector2d(0.0, small_rune_speed_);
  angle_cov_ = Eigen::Matrix2d::Identity();
  legacy_angle_state_ = Eigen::Vector3d(0.0, small_rune_speed_, 0.0);
  legacy_angle_cov_ = Eigen::Matrix3d::Identity();
}

void CeresRunePredictor::set_first_detect(const RuneObservation & observation)
{
  first_detect_ = false;
  R_target2world_base_ = observation.R_target2world;
  start_time_ = observation.timestamp;
  angle_abs_last_ = 0.0;
  angle_rel_ = 0.0;
  raw_angle_rel_ = 0.0;
  total_shift_ = 0;
  angle_state_ = Eigen::Vector2d(small_rune_speed_, 0.0);
  legacy_angle_state_ = Eigen::Vector3d(0.0, small_rune_speed_, 0.0);
}

void CeresRunePredictor::update_angle(const RuneObservation & observation)
{
  const Eigen::Matrix3d R_rel = R_target2world_base_.transpose() * observation.R_target2world;
  const double angle_abs = -std::atan2(R_rel(1, 0), R_rel(0, 0));

  const double delta_abs = angle_abs - angle_abs_last_;
  angle_abs_last_ = angle_abs;
  total_shift_ += static_cast<int>(std::round(delta_abs / kAngleBetweenFanBlades));
  const double unwrapped_angle = angle_abs - total_shift_ * kAngleBetweenFanBlades;
  raw_angle_rel_ = unwrapped_angle;

  const double dt = last_detection_time_.time_since_epoch().count() == 0
                      ? 1.0 / 100.0
                      : std::max(1e-3, tools::delta_time(observation.timestamp, last_detection_time_));
  angle_rel_ = filter_angle(unwrapped_angle, dt, RuneMode::BIG);

  const double time = tools::delta_time(observation.timestamp, start_time_);
  const double fitting_angle = fit_use_raw_angle_ ? raw_angle_rel_ : angle_rel_;
  if (std::isfinite(time) && std::isfinite(fitting_angle)) {
    fit_data_.emplace_back(time, std::abs(fitting_angle));
    if (fit_data_.size() > static_cast<std::size_t>(max_fit_data_size_)) {
      fit_data_.erase(fit_data_.begin(), fit_data_.begin() + fit_data_.size() / 2);
    }
  }

  update_direction();

  ++fit_frame_count_;
  if (
    fit_interval_frames_ <= 1 || fit_frame_count_ % fit_interval_frames_ == 0 ||
    !valid_params_)
  {
    valid_params_ = fit_once();
  }

  const Eigen::Vector3d center =
    observation.target_in_world + observation.R_target2world * Eigen::Vector3d(0.0, fan_len_, 0.0);
  center_positions_.push_back(center);
  const auto max_centers = static_cast<std::size_t>(std::max(1, direction_window_ * 2));
  while (center_positions_.size() > max_centers) center_positions_.pop_front();
}

double CeresRunePredictor::filter_angle(double measurement, double dt, RuneMode)
{
  if (angle_filter_mode_ == AngleFilterMode::LEGACY_EKF) {
    return legacy_ekf_filter_angle(measurement, dt, RuneMode::BIG);
  }
  return simple_filter_angle(measurement, dt);
}

double CeresRunePredictor::simple_filter_angle(double measurement, double dt)
{
  if (!angle_filter_initialized_) {
    angle_state_ = Eigen::Vector2d(small_rune_speed_, measurement);
    angle_cov_ = Eigen::Matrix2d::Identity();
    angle_filter_initialized_ = true;
    return measurement;
  }

  Eigen::Matrix2d F;
  F << 1.0, 0.0, dt, 1.0;
  Eigen::Matrix2d Q;
  Q << Qw_, 0.0, 0.0, Qtheta_;
  angle_state_ = F * angle_state_;
  angle_cov_ = F * angle_cov_ * F.transpose() + Q;

  Eigen::RowVector2d H;
  H << 0.0, 1.0;
  const double residual = measurement - angle_state_(1);
  const double S = (H * angle_cov_ * H.transpose())(0, 0) + Rtheta_;
  const Eigen::Vector2d K = angle_cov_ * H.transpose() / S;
  angle_state_ += K * residual;
  angle_cov_ = (Eigen::Matrix2d::Identity() - K * H) * angle_cov_;
  return angle_state_(1);
}

double CeresRunePredictor::legacy_ekf_filter_angle(double measurement, double dt, RuneMode mode)
{
  if (!angle_filter_initialized_) {
    legacy_angle_state_ = Eigen::Vector3d(0.0, small_rune_speed_, measurement);
    legacy_angle_cov_ = Eigen::Matrix3d::Identity();
    angle_state_ = Eigen::Vector2d(small_rune_speed_, measurement);
    angle_cov_ = Eigen::Matrix2d::Identity();
    angle_filter_initialized_ = true;
    return measurement;
  }

  const double direction = direction_sign();
  const double phase = params_[1] * (legacy_angle_state_(0) + params_[2]);
  double model_omega = legacy_angle_state_(1);
  if (mode == RuneMode::SMALL) {
    model_omega = direction == 0.0 ? small_rune_speed_ : direction * small_rune_speed_;
  } else if (direction != 0.0 && std::abs(params_[1]) > 1e-6) {
    model_omega = (params_[0] / params_[1] * std::sin(phase) + params_[3]) * direction;
  }

  Eigen::Vector3d predicted = legacy_angle_state_;
  predicted(0) += dt;
  predicted(1) = model_omega;
  predicted(2) += predicted(1) * dt;

  Eigen::Matrix3d F = Eigen::Matrix3d::Zero();
  if (mode == RuneMode::SMALL) {
    F << 1.0, 0.0, 0.0,
         0.0, 1.0, 0.0,
         0.0, dt, 1.0;
  } else {
    const double domega_dt =
      direction == 0.0 ? 0.0 : params_[0] * std::cos(phase) * direction;
    F << 1.0, 0.0, 0.0,
         domega_dt, 0.0, 0.0,
         0.0, dt, 1.0;
  }

  Eigen::Matrix3d Q = Eigen::Matrix3d::Zero();
  Q(1, 1) = Qw_;
  Q(2, 2) = Qtheta_;
  legacy_angle_state_ = predicted;
  legacy_angle_cov_ = F * legacy_angle_cov_ * F.transpose() + Q;

  Eigen::RowVector3d H;
  H << 0.0, 0.0, 1.0;
  const double residual = measurement - legacy_angle_state_(2);
  const double S = (H * legacy_angle_cov_ * H.transpose())(0, 0) + Rtheta_;
  const Eigen::Vector3d K = legacy_angle_cov_ * H.transpose() / S;
  legacy_angle_state_ += K * residual;
  legacy_angle_cov_ = (Eigen::Matrix3d::Identity() - K * H) * legacy_angle_cov_;

  angle_state_ = Eigen::Vector2d(legacy_angle_state_(1), legacy_angle_state_(2));
  angle_cov_ << legacy_angle_cov_(1, 1), legacy_angle_cov_(1, 2),
    legacy_angle_cov_(2, 1), legacy_angle_cov_(2, 2);
  return legacy_angle_state_(2);
}

void CeresRunePredictor::update_direction()
{
  if (direction_ != Direction::UNKNOWN && direction_ != Direction::STABLE) return;

  direction_data_.push_back(angle_rel_);
  if (direction_data_.size() < static_cast<std::size_t>(std::max(2, direction_window_))) return;

  int stable = 0;
  int anti = 0;
  int clockwise = 0;
  const std::size_t half = direction_data_.size() / 2;
  for (std::size_t i = 0; i < half; ++i) {
    const double diff = direction_data_[i + half] - direction_data_[i];
    if (diff > 1.5e-2) {
      ++clockwise;
    } else if (diff < -1.5e-2) {
      ++anti;
    } else {
      ++stable;
    }
  }

  const int best = std::max({stable, clockwise, anti});
  if (best == clockwise) {
    direction_ = Direction::CLOCKWISE;
  } else if (best == anti) {
    direction_ = Direction::ANTI_CLOCKWISE;
  } else {
    direction_ = Direction::STABLE;
  }
}

bool CeresRunePredictor::fit_once()
{
  auto data = sanitized(fit_data_);
  if (data.size() < static_cast<std::size_t>(min_fit_data_size_)) return false;
  if (data.size() < static_cast<std::size_t>(2 * min_fit_data_size_)) {
    convexity_ = get_convexity(data);
  }
  params_ = ransac_fitting(data, convexity_);
  return std::all_of(params_.begin(), params_.end(), [](double value) {
    return std::isfinite(value);
  });
}

CeresRunePredictor::Convexity CeresRunePredictor::get_convexity(
  const std::vector<std::pair<double, double>> & data) const
{
  if (data.size() < 2) return Convexity::UNKNOWN;
  const auto first = data.front();
  const auto last = data.back();
  if (std::abs(last.first - first.first) < 1e-9) return Convexity::UNKNOWN;

  const double slope = (last.second - first.second) / (last.first - first.first);
  const double offset =
    (first.second * last.first - last.second * first.first) / (last.first - first.first);
  if (!std::isfinite(slope) || !std::isfinite(offset)) return Convexity::UNKNOWN;

  int concave = 0;
  int convex = 0;
  for (const auto & point : data) {
    if (slope * point.first + offset > point.second) {
      ++concave;
    } else {
      ++convex;
    }
  }

  const int threshold = static_cast<int>(data.size() * 0.75);
  if (concave > threshold) return Convexity::CONCAVE;
  if (convex > threshold) return Convexity::CONVEX;
  return Convexity::UNKNOWN;
}

std::array<double, 5> CeresRunePredictor::ransac_fitting(
  const std::vector<std::pair<double, double>> & data, Convexity convexity) const
{
  std::vector<std::pair<double, double>> inliers(data.begin(), data.end());
  std::vector<std::pair<double, double>> outliers;
  std::array<double, 5> params{0.470, 1.942, 0.0, 1.178, 0.0};
  const int iter_times = data.size() < 400 ? 200 : 20;

  for (int iter = 0; iter < iter_times; ++iter) {
    std::vector<std::pair<double, double>> sample;
    if (inliers.size() > 400) {
      sample.assign(inliers.end() - 200, inliers.end());
    } else {
      sample.assign(inliers.begin(), inliers.end());
    }

    params = least_square_estimate(sample, params, convexity);

    if (data.size() <= 800) continue;

    std::vector<double> errors;
    errors.reserve(inliers.size());
    for (const auto & inlier : inliers) {
      errors.push_back(std::abs(inlier.second - get_angle_big(inlier.first, params)));
    }
    std::sort(errors.begin(), errors.end());
    const auto threshold = errors[static_cast<std::size_t>(errors.size() * 0.95)];

    for (std::size_t i = 0; i + 100 < inliers.size();) {
      if (std::abs(inliers[i].second - get_angle_big(inliers[i].first, params)) > threshold) {
        outliers.push_back(inliers[i]);
        inliers.erase(inliers.begin() + static_cast<std::ptrdiff_t>(i));
      } else {
        ++i;
      }
    }

    for (std::size_t i = 0; i < outliers.size();) {
      if (std::abs(outliers[i].second - get_angle_big(outliers[i].first, params)) < threshold) {
        inliers.emplace(inliers.begin(), outliers[i]);
        outliers.erase(outliers.begin() + static_cast<std::ptrdiff_t>(i));
      } else {
        ++i;
      }
    }
  }

  return least_square_estimate(inliers, params, convexity);
}

std::array<double, 5> CeresRunePredictor::least_square_estimate(
  const std::vector<std::pair<double, double>> & points,
  const std::array<double, 5> & initial_params, Convexity convexity) const
{
  std::array<double, 5> result = initial_params;
  if (points.empty()) return result;

  ceres::Problem problem;
  std::size_t valid_count = 0;
  for (const auto & point : points) {
    if (!is_finite_point(point)) continue;
    auto * cost = new FitCost(point.first, point.second);
    auto * loss = new ceres::SoftLOneLoss(0.1);
    problem.AddResidualBlock(cost, loss, result.begin());
    ++valid_count;
  }
  if (valid_count == 0) return result;

  std::array<double, 3> omega{};
  if (points.size() < 100) {
    if (convexity == Convexity::CONCAVE) {
      problem.SetParameterUpperBound(result.begin(), 2, -2.8);
      problem.SetParameterLowerBound(result.begin(), 2, -4.0);
    } else if (convexity == Convexity::CONVEX) {
      problem.SetParameterUpperBound(result.begin(), 2, -1.1);
      problem.SetParameterLowerBound(result.begin(), 2, -2.3);
    }
    omega = {10.0, 1.0, 1.0};
  } else {
    omega = {60.0, 50.0, 50.0};
  }

  problem.AddResidualBlock(
    new PriorCost(result[0], 0),
    new ceres::ScaledLoss(new ceres::HuberLoss(0.1), omega[0], ceres::TAKE_OWNERSHIP),
    result.begin());
  problem.AddResidualBlock(
    new PriorCost(result[1], 1),
    new ceres::ScaledLoss(new ceres::HuberLoss(0.1), omega[1], ceres::TAKE_OWNERSHIP),
    result.begin());
  problem.AddResidualBlock(
    new PriorCost(result[3], 3),
    new ceres::ScaledLoss(new ceres::HuberLoss(0.1), omega[2], ceres::TAKE_OWNERSHIP),
    result.begin());

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.max_num_iterations = 50;
  options.minimizer_progress_to_stdout = false;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  return result;
}

double CeresRunePredictor::get_angle_big(
  double time, const std::array<double, 5> & params) const
{
  return -params[0] * std::cos(params[1] * (time + params[2])) + params[3] * time + params[4];
}

double CeresRunePredictor::get_big_rotation(double current_time, double future_dt) const
{
  const double delta =
    get_angle_big(current_time + future_dt, params_) - get_angle_big(current_time, params_);
  return direction_sign() * delta;
}

double CeresRunePredictor::get_small_rotation(double future_dt) const
{
  return direction_sign() * small_rune_speed_ * future_dt;
}

double CeresRunePredictor::direction_sign() const
{
  if (direction_ == Direction::CLOCKWISE) return 1.0;
  if (direction_ == Direction::ANTI_CLOCKWISE) return -1.0;
  return 0.0;
}

Eigen::Vector3d CeresRunePredictor::predict_point_in_world(
  const RuneObservation & observation, double rotation_angle) const
{
  const Eigen::Vector3d target_shift_in_target(
    fan_len_ * std::sin(rotation_angle),
    fan_len_ - fan_len_ * std::cos(rotation_angle),
    0.0);
  return observation.R_target2world * target_shift_in_target + observation.target_in_world;
}

std::optional<Eigen::Vector2d> CeresRunePredictor::solve_yaw_pitch(
  const Eigen::Vector3d & target_in_world, double bullet_speed, double & fly_time) const
{
  const double distance = std::hypot(target_in_world.x(), target_in_world.y());
  const double height = target_in_world.z();
  tools::Trajectory trajectory(bullet_speed, distance, height);
  if (trajectory.unsolvable) {
    tools::logger()->debug(
      "[CeresRunePredictor] Unsolvable trajectory: speed={:.2f}, d={:.2f}, h={:.2f}",
      bullet_speed,
      distance,
      height);
    return std::nullopt;
  }

  fly_time = trajectory.fly_time;
  const double yaw = std::atan2(target_in_world.y(), target_in_world.x()) + yaw_offset_;
  const double pitch = -(trajectory.pitch + pitch_offset_);
  return Eigen::Vector2d(yaw, pitch);
}

Eigen::Vector3d CeresRunePredictor::fallback_center() const
{
  if (center_positions_.empty()) return Eigen::Vector3d::Zero();
  Eigen::Vector3d sum = Eigen::Vector3d::Zero();
  for (const auto & center : center_positions_) {
    sum += center;
  }
  return sum / static_cast<double>(center_positions_.size());
}

void CeresRunePredictor::fill_debug_model(double current_time)
{
  debug_.raw_relative_angle = raw_angle_rel_;
  debug_.filtered_relative_angle = angle_rel_;
  debug_.relative_angle = angle_rel_;
  debug_.angle_velocity = angle_state_(0);
  debug_.model_time = current_time;
  debug_.params = params_;
  debug_.valid_params = valid_params_;
  debug_.fit_data_size = static_cast<int>(fit_data_.size());
  debug_.fit_angle = get_angle_big(current_time, params_);
  const double fitting_angle = fit_use_raw_angle_ ? raw_angle_rel_ : angle_rel_;
  debug_.fit_residual = std::abs(fitting_angle) - debug_.fit_angle;
}

bool CeresRunePredictor::reject_observation(
  const RuneObservation & observation, std::string & reason) const
{
  const double depth = observation.target_in_camera.z();
  if (!std::isfinite(observation.distance) || !std::isfinite(depth)) {
    reason = "non-finite pose";
    return true;
  }
  if (pnp_min_distance_ > 0.0 && observation.distance < pnp_min_distance_) {
    reason = "distance too small";
    return true;
  }
  if (pnp_max_distance_ > 0.0 && observation.distance > pnp_max_distance_) {
    reason = "distance too large";
    return true;
  }
  if (pnp_min_depth_ > 0.0 && depth < pnp_min_depth_) {
    reason = "depth too small";
    return true;
  }
  if (
    pnp_max_reprojection_error_ > 0.0 &&
    observation.reprojection_error > pnp_max_reprojection_error_)
  {
    reason = "reprojection error too large";
    return true;
  }

  if (first_detect_ || !latest_observation_.has_value()) return false;

  const double dt = tools::delta_time(observation.timestamp, latest_observation_->timestamp);
  if (
    !std::isfinite(dt) || dt <= 1e-3 ||
    (pnp_angle_gate_max_gap_ > 0.0 && dt > pnp_angle_gate_max_gap_))
  {
    return false;
  }

  const double distance_jump = std::abs(observation.distance - latest_observation_->distance);
  if (pnp_max_distance_jump_ > 0.0 && distance_jump > pnp_max_distance_jump_) {
    reason = "distance jump too large";
    return true;
  }

  const double raw_angle = raw_relative_angle_of(observation);
  const double angle_rate = std::abs((raw_angle - raw_angle_rel_) / dt);
  if (pnp_max_angle_rate_ > 0.0 && angle_rate > pnp_max_angle_rate_) {
    reason = "angle rate too large";
    return true;
  }

  return false;
}

double CeresRunePredictor::raw_relative_angle_of(const RuneObservation & observation) const
{
  const Eigen::Matrix3d R_rel = R_target2world_base_.transpose() * observation.R_target2world;
  const double angle_abs = -std::atan2(R_rel(1, 0), R_rel(0, 0));
  const double delta_abs = angle_abs - angle_abs_last_;
  const int shift = total_shift_ + static_cast<int>(std::round(delta_abs / kAngleBetweenFanBlades));
  return angle_abs - shift * kAngleBetweenFanBlades;
}

}  // namespace auto_buff
