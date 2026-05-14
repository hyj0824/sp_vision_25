#ifndef AUTO_BUFF__CERES_RUNE_PREDICTOR_HPP
#define AUTO_BUFF__CERES_RUNE_PREDICTOR_HPP

#include <Eigen/Dense>
#include <array>
#include <chrono>
#include <deque>
#include <optional>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "io/command.hpp"
#include "rune_detector.hpp"

namespace auto_buff
{

enum class RuneMode { SMALL, BIG };

struct RuneObservation
{
  std::chrono::steady_clock::time_point timestamp;
  std::vector<cv::Point2f> keypoints;

  Eigen::Vector3d target_in_camera = Eigen::Vector3d::Zero();
  Eigen::Vector3d target_in_gimbal = Eigen::Vector3d::Zero();
  Eigen::Vector3d target_in_world = Eigen::Vector3d::Zero();

  Eigen::Matrix3d R_target2camera = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d R_target2gimbal = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d R_target2world = Eigen::Matrix3d::Identity();

  double relative_angle = 0.0;
  double distance = 0.0;
  double reprojection_error = 0.0;
};

struct RuneAimDebug
{
  bool valid = false;
  RuneMode mode = RuneMode::SMALL;
  Eigen::Vector3d current_target_in_world = Eigen::Vector3d::Zero();
  Eigen::Vector3d predict_target_in_world = Eigen::Vector3d::Zero();
  Eigen::Vector3d center_in_world = Eigen::Vector3d::Zero();
  double relative_angle = 0.0;
  double raw_relative_angle = 0.0;
  double filtered_relative_angle = 0.0;
  double angle_velocity = 0.0;
  double model_time = 0.0;
  double fit_angle = 0.0;
  double fit_residual = 0.0;
  double predict_rotation = 0.0;
  double fly_time = 0.0;
  double yaw = 0.0;
  double pitch = 0.0;
  std::array<double, 5> params{0.0, 0.0, 0.0, 0.0, 0.0};
  bool valid_params = false;
  int fit_data_size = 0;
  int direction = 0;
  int lost_count = 0;
};

class CeresRunePredictor
{
public:
  explicit CeresRunePredictor(const std::string & config_path);

  void set_R_gimbal2world(const Eigen::Quaterniond & q);

  std::optional<RuneObservation> update(
    const std::optional<RuneDetection> & detection, std::chrono::steady_clock::time_point timestamp);

  io::Command aim(
    RuneMode mode, std::chrono::steady_clock::time_point timestamp, double bullet_speed,
    bool to_now = true);
  io::Command aim_static(
    std::chrono::steady_clock::time_point timestamp, double bullet_speed, bool fire = false);

  const RuneAimDebug & debug() const { return debug_; }

  bool solve_pnp(const RuneDetection & detection, RuneObservation & observation) const;

  std::vector<cv::Point2f> reproject(
    const RuneObservation & observation, const Eigen::Vector3d & target_in_world) const;

private:
  enum class AngleFilterMode { SIMPLE, LEGACY_EKF };
  enum class Direction { UNKNOWN, STABLE, ANTI_CLOCKWISE, CLOCKWISE };
  enum class Convexity { UNKNOWN, CONCAVE, CONVEX };

  cv::Mat camera_matrix_;
  cv::Mat distort_coeffs_;
  Eigen::Matrix3d R_gimbal2imubody_ = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d R_camera2gimbal_ = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t_camera2gimbal_ = Eigen::Vector3d::Zero();
  Eigen::Matrix3d R_gimbal2world_ = Eigen::Matrix3d::Identity();

  double yaw_offset_ = 0.0;
  double pitch_offset_ = 0.0;
  double delay_time_ = 0.0;
  double fire_gap_time_ = 0.6;
  double fan_len_ = 0.7;
  double target_radius_ = 0.145;
  double small_rune_speed_ = CV_PI / 3.0;
  double stale_reset_time_ = 2.0;
  double center_fallback_time_ = 0.1;
  bool auto_fire_ = true;
  bool reject_pnp_outliers_ = true;
  bool log_rejected_pnp_ = false;
  bool fit_use_raw_angle_ = false;
  AngleFilterMode angle_filter_mode_ = AngleFilterMode::SIMPLE;
  int lost_count_threshold_ = 20;
  int direction_window_ = 10;
  int min_fit_data_size_ = 20;
  int max_fit_data_size_ = 1200;
  int min_shooting_size_ = 30;
  int fit_interval_frames_ = 3;

  double Qw_ = 0.2;
  double Qtheta_ = 0.08;
  double Rtheta_ = 0.4;
  double pnp_min_distance_ = 1.0;
  double pnp_max_distance_ = 15.0;
  double pnp_min_depth_ = 0.5;
  double pnp_max_reprojection_error_ = 8.0;
  double pnp_max_distance_jump_ = 2.5;
  double pnp_max_angle_rate_ = 8.0;
  double pnp_angle_gate_max_gap_ = 0.25;

  bool first_detect_ = true;
  bool angle_filter_initialized_ = false;
  int lost_count_ = 0;
  int total_shift_ = 0;
  int fit_frame_count_ = 0;
  Direction direction_ = Direction::UNKNOWN;
  Convexity convexity_ = Convexity::UNKNOWN;

  std::chrono::steady_clock::time_point start_time_{};
  std::chrono::steady_clock::time_point last_detection_time_{};
  std::chrono::steady_clock::time_point last_fire_time_{};

  Eigen::Matrix3d R_target2world_base_ = Eigen::Matrix3d::Identity();
  double angle_abs_last_ = 0.0;
  double angle_rel_ = 0.0;
  double raw_angle_rel_ = 0.0;

  Eigen::Vector2d angle_state_ = Eigen::Vector2d::Zero();  // omega, theta
  Eigen::Matrix2d angle_cov_ = Eigen::Matrix2d::Identity();
  Eigen::Vector3d legacy_angle_state_ = Eigen::Vector3d::Zero();  // t, omega, theta
  Eigen::Matrix3d legacy_angle_cov_ = Eigen::Matrix3d::Identity();

  std::array<double, 5> params_{0.470, 1.942, 0.0, 1.178, 0.0};
  bool valid_params_ = false;
  std::vector<std::pair<double, double>> fit_data_;
  std::vector<double> direction_data_;
  std::deque<Eigen::Vector3d> center_positions_;

  std::optional<RuneObservation> latest_observation_;
  RuneAimDebug debug_;

  void reset(std::chrono::steady_clock::time_point timestamp);
  void set_first_detect(const RuneObservation & observation);
  void update_angle(const RuneObservation & observation);
  double filter_angle(double measurement, double dt, RuneMode mode);
  double simple_filter_angle(double measurement, double dt);
  double legacy_ekf_filter_angle(double measurement, double dt, RuneMode mode);
  void update_direction();

  bool fit_once();
  Convexity get_convexity(const std::vector<std::pair<double, double>> & data) const;
  std::array<double, 5> ransac_fitting(
    const std::vector<std::pair<double, double>> & data, Convexity convexity) const;
  std::array<double, 5> least_square_estimate(
    const std::vector<std::pair<double, double>> & points,
    const std::array<double, 5> & initial_params, Convexity convexity) const;

  double get_angle_big(double time, const std::array<double, 5> & params) const;
  double get_big_rotation(double current_time, double future_dt) const;
  double get_small_rotation(double future_dt) const;
  double direction_sign() const;

  Eigen::Vector3d predict_point_in_world(
    const RuneObservation & observation, double rotation_angle) const;
  std::optional<Eigen::Vector2d> solve_yaw_pitch(
    const Eigen::Vector3d & target_in_world, double bullet_speed, double & fly_time) const;
  Eigen::Vector3d fallback_center() const;
  void fill_debug_model(double current_time);
  bool reject_observation(const RuneObservation & observation, std::string & reason) const;
  double raw_relative_angle_of(const RuneObservation & observation) const;
};

}  // namespace auto_buff

#endif  // AUTO_BUFF__CERES_RUNE_PREDICTOR_HPP
