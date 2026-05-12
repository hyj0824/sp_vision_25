#include "target.hpp"

#include <algorithm>
#include <array>
#include <numeric>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_aim
{
namespace
{
constexpr double OUTPOST_ARMOR_RADIUS = 0.275;       // m, 550mm rotation diameter
constexpr double OUTPOST_ARMOR_HEIGHT_STEP = 0.102;  // m, 1114/1216/1318mm centers
constexpr int OUTPOST_HEIGHT_MODEL_COUNT = 6;
constexpr double OUTPOST_HEIGHT_MODEL_SCORE_GATE = 0.08;    // m
constexpr double OUTPOST_HEIGHT_MODEL_SCORE_MARGIN = 0.04;  // m
constexpr int OUTPOST_HEIGHT_MODEL_MIN_VOTES = 2;

bool is_outpost(const ArmorName name) { return name == ArmorName::outpost; }

int wrap_index(int value, int size)
{
  auto r = value % size;
  return r < 0 ? r + size : r;
}

double outpost_armor_z_offset(const int id, const int phase, const int direction)
{
  // 前哨站三块板不共高，且实物“顺时针编号递增”与代码里 yaw 正方向不一定一致，
  // 因此这里把“相位 + 方向”作为可自动纠偏的离散几何模型。
  constexpr std::array<double, 3> z_offsets{
    -OUTPOST_ARMOR_HEIGHT_STEP, 0.0, OUTPOST_ARMOR_HEIGHT_STEP};
  return z_offsets[wrap_index(phase + direction * id, static_cast<int>(z_offsets.size()))];
}
}  // namespace

Target::Target(
  const Armor & armor, std::chrono::steady_clock::time_point t, double radius, int armor_num,
  Eigen::VectorXd P0_dig)
: name(armor.name),
  armor_type(armor.type),
  jumped(false),
  last_id(0),
  update_count_(0),
  armor_num_(armor_num),
  t_(t),
  is_switch_(false),
  is_converged_(false),
  switch_count_(0)
{
  auto r = is_outpost(armor.name) ? OUTPOST_ARMOR_RADIUS : radius;
  priority = armor.priority;
  const Eigen::VectorXd & xyz = armor.xyz_in_world;
  const Eigen::VectorXd & ypr = armor.ypr_in_world;

  // 旋转中心的坐标
  auto center_x = xyz[0] + r * std::cos(ypr[0]);
  auto center_y = xyz[1] + r * std::sin(ypr[0]);
  // outpost 只有相对高度已知，绝对中心高度由 EKF 的 x[4] 负责吸收。
  auto center_z = xyz[2]
                  - (is_outpost(armor.name) ? outpost_armor_z_offset(0, 0, 1) : 0.0);

  // x vx y vy z vz a w r l h
  // a: angle
  // w: angular velocity
  // l: r2 - r1
  // h: z2 - z1
  Eigen::VectorXd x0{{center_x, 0, center_y, 0, center_z, 0, ypr[0], 0, r, 0, 0}};  //初始化预测量
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();

  // 防止夹角求和出现异常值
  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[6] = tools::limit_rad(c[6]);
    return c;
  };

  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);  //初始化滤波器（预测量、预测量协方差）
}

Target::Target(double x, double vyaw, double radius, double h) : armor_num_(4)
{
  Eigen::VectorXd x0{{x, 0, 0, 0, 0, 0, 0, vyaw, radius, 0, h}};
  Eigen::VectorXd P0_dig{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();

  // 防止夹角求和出现异常值
  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[6] = tools::limit_rad(c[6]);
    return c;
  };

  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);  //初始化滤波器（预测量、预测量协方差）
}

void Target::predict(std::chrono::steady_clock::time_point t)
{
  auto dt = tools::delta_time(t, t_);
  predict(dt);
  t_ = t;
}

void Target::predict(double dt)
{
  // 状态转移矩阵
  // clang-format off
  Eigen::MatrixXd F{
    {1, dt,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    {0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    {0,  0,  1, dt,  0,  0,  0,  0,  0,  0,  0},
    {0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0},
    {0,  0,  0,  0,  1, dt,  0,  0,  0,  0,  0},
    {0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0},
    {0,  0,  0,  0,  0,  0,  1, dt,  0,  0,  0},
    {0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0},
    {0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0},
    {0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0},
    {0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1}
  };
  // clang-format on

  // Piecewise White Noise Model
  // https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/07-Kalman-Filter-Math.ipynb
  double v1, v2;
  if (is_outpost(name)) {
    v1 = 10;   // 前哨站加速度方差
    v2 = 0.1;  // 前哨站角加速度方差
  } else {
    v1 = 100;  // 加速度方差
    v2 = 400;  // 角加速度方差
  }
  auto a = dt * dt * dt * dt / 4;
  auto b = dt * dt * dt / 2;
  auto c = dt * dt;
  // 预测过程噪声偏差的方差
  // clang-format off
  Eigen::MatrixXd Q{
    {a * v1, b * v1,      0,      0,      0,      0,      0,      0, 0, 0, 0},
    {b * v1, c * v1,      0,      0,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0, a * v1, b * v1,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0, b * v1, c * v1,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0, a * v1, b * v1,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0, b * v1, c * v1,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0, a * v2, b * v2, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0, b * v2, c * v2, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0}
  };
  // clang-format on

  // 防止夹角求和出现异常值
  auto f = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
    Eigen::VectorXd x_prior = F * x;
    x_prior[6] = tools::limit_rad(x_prior[6]);
    return x_prior;
  };

  // 前哨站转速特判
  if (this->convergened() && is_outpost(this->name) && std::abs(this->ekf_.x[7]) > 2)
    this->ekf_.x[7] = this->ekf_.x[7] > 0 ? 2.51 : -2.51;

  ekf_.predict(F, Q, f);
}

void Target::update(const Armor & armor)
{
  // 装甲板匹配
  int id = 0;
  auto min_angle_error = 1e10;
  const std::vector<Eigen::Vector4d> & xyza_list = armor_xyza_list();

  std::vector<std::pair<Eigen::Vector4d, int>> xyza_i_list;
  for (int i = 0; i < armor_num_; i++) {
    xyza_i_list.push_back({xyza_list[i], i});
  }

  std::sort(
    xyza_i_list.begin(), xyza_i_list.end(),
    [](const std::pair<Eigen::Vector4d, int> & a, const std::pair<Eigen::Vector4d, int> & b) {
      Eigen::Vector3d ypd1 = tools::xyz2ypd(a.first.head(3));
      Eigen::Vector3d ypd2 = tools::xyz2ypd(b.first.head(3));
      return ypd1[2] < ypd2[2];
    });

  // 取最多前3个distance最小的装甲板
  const int candidate_count = std::min<int>(3, xyza_i_list.size());
  for (int i = 0; i < candidate_count; i++) {
    const auto & xyza = xyza_i_list[i].first;
    Eigen::Vector3d ypd = tools::xyz2ypd(xyza.head(3));
    auto angle_error = std::abs(tools::limit_rad(armor.ypr_in_world[0] - xyza[3])) +
                       std::abs(tools::limit_rad(armor.ypd_in_world[0] - ypd[0]));

    if (std::abs(angle_error) < std::abs(min_angle_error)) {
      id = xyza_i_list[i].second;
      min_angle_error = angle_error;
    }
  }

  if (id != 0) jumped = true;

  if (id != last_id) {
    is_switch_ = true;
  } else {
    is_switch_ = false;
  }

  if (is_switch_) switch_count_++;

  if (is_outpost(name)) update_outpost_height_model(armor, id);

  last_id = id;
  update_count_++;

  update_ypda(armor, id);
}

void Target::update_outpost_height_model(const Armor & armor, int id)
{
  if (!is_outpost(name)) return;

  if (last_outpost_observed_id_.has_value() && last_outpost_observed_z_.has_value()) {
    auto last_id = last_outpost_observed_id_.value();
    auto measured_dz = armor.xyz_in_world[2] - last_outpost_observed_z_.value();

    std::array<double, OUTPOST_HEIGHT_MODEL_COUNT> scores{};
    scores.fill(1e9);
    for (int phase = 0; phase < 3; phase++) {
      for (int direction : {1, -1}) {
        auto model = phase + (direction < 0 ? 3 : 0);
        auto expected_dz =
          outpost_armor_z_offset(id, phase, direction) -
          outpost_armor_z_offset(last_id, phase, direction);
        auto score = std::abs(measured_dz - expected_dz);
        scores[model] = score;
      }
    }

    auto best_it = std::min_element(scores.begin(), scores.end());
    auto best_model = static_cast<int>(std::distance(scores.begin(), best_it));
    auto best_score = *best_it;
    auto second_score = 1e9;
    for (int i = 0; i < OUTPOST_HEIGHT_MODEL_COUNT; i++) {
      if (i == best_model) continue;
      second_score = std::min(second_score, scores[i]);
    }

    if (
      best_score < OUTPOST_HEIGHT_MODEL_SCORE_GATE &&
      second_score - best_score > OUTPOST_HEIGHT_MODEL_SCORE_MARGIN) {
      outpost_height_model_votes_[best_model]++;
      tools::logger()->debug(
        "[Target] outpost height model vote: phase={}, direction={}, score={:.3f}",
        best_model % 3, best_model < 3 ? 1 : -1, best_score);
    }

    auto voted_it =
      std::max_element(outpost_height_model_votes_.begin(), outpost_height_model_votes_.end());
    auto voted_model = static_cast<int>(std::distance(outpost_height_model_votes_.begin(), voted_it));
    if (
      *voted_it >= OUTPOST_HEIGHT_MODEL_MIN_VOTES &&
      voted_model != (outpost_height_phase_ + (outpost_height_direction_ < 0 ? 3 : 0))) {
      auto old_offset = outpost_armor_z_offset(id, outpost_height_phase_, outpost_height_direction_);
      outpost_height_phase_ = voted_model % 3;
      outpost_height_direction_ = voted_model < 3 ? 1 : -1;
      auto new_offset = outpost_armor_z_offset(id, outpost_height_phase_, outpost_height_direction_);

      // 切换离散高度模型时，补偿中心高度，避免当前目标连续性被打断。
      ekf_.x[4] += old_offset - new_offset;
      outpost_height_model_votes_.fill(0);
      outpost_height_model_votes_[voted_model] = 1;
      tools::logger()->info(
        "[Target] outpost height model corrected: phase={}, direction={}", outpost_height_phase_,
        outpost_height_direction_);
    }
  }

  last_outpost_observed_id_ = id;
  last_outpost_observed_z_ = armor.xyz_in_world[2];
}

void Target::update_ypda(const Armor & armor, int id)
{
  //观测jacobi
  Eigen::MatrixXd H = h_jacobian(ekf_.x, id);
  // Eigen::VectorXd R_dig{{4e-3, 4e-3, 1, 9e-2}};
  auto center_yaw = std::atan2(armor.xyz_in_world[1], armor.xyz_in_world[0]);
  auto delta_angle = tools::limit_rad(armor.ypr_in_world[0] - center_yaw);
  Eigen::VectorXd R_dig{
    {4e-3, 4e-3, log(std::abs(delta_angle) + 1) + 1,
     log(std::abs(armor.ypd_in_world[2]) + 1) / 200 + 9e-2}};

  //测量过程噪声偏差的方差
  Eigen::MatrixXd R = R_dig.asDiagonal();

  // 定义非线性转换函数h: x -> z
  auto h = [&](const Eigen::VectorXd & x) -> Eigen::Vector4d {
    Eigen::VectorXd xyz = h_armor_xyz(x, id);
    Eigen::VectorXd ypd = tools::xyz2ypd(xyz);
    auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
    return {ypd[0], ypd[1], ypd[2], angle};
  };

  // 防止夹角求差出现异常值
  auto z_subtract = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a - b;
    c[0] = tools::limit_rad(c[0]);
    c[1] = tools::limit_rad(c[1]);
    c[3] = tools::limit_rad(c[3]);
    return c;
  };

  const Eigen::VectorXd & ypd = armor.ypd_in_world;
  const Eigen::VectorXd & ypr = armor.ypr_in_world;
  Eigen::VectorXd z{{ypd[0], ypd[1], ypd[2], ypr[0]}};  //获得观测量

  ekf_.update(z, H, R, h, z_subtract);
}

Eigen::VectorXd Target::ekf_x() const { return ekf_.x; }

const tools::ExtendedKalmanFilter & Target::ekf() const { return ekf_; }

std::vector<Eigen::Vector4d> Target::armor_xyza_list() const
{
  std::vector<Eigen::Vector4d> _armor_xyza_list;

  for (int i = 0; i < armor_num_; i++) {
    auto angle = tools::limit_rad(ekf_.x[6] + i * 2 * CV_PI / armor_num_);
    Eigen::Vector3d xyz = h_armor_xyz(ekf_.x, i);
    _armor_xyza_list.push_back({xyz[0], xyz[1], xyz[2], angle});
  }
  return _armor_xyza_list;
}

bool Target::diverged() const
{
  auto r_ok = ekf_.x[8] > 0.05 && ekf_.x[8] < 0.5;
  auto l_ok = ekf_.x[8] + ekf_.x[9] > 0.05 && ekf_.x[8] + ekf_.x[9] < 0.5;

  if (r_ok && l_ok) return false;

  tools::logger()->debug("[Target] r={:.3f}, l={:.3f}", ekf_.x[8], ekf_.x[9]);
  return true;
}

bool Target::convergened()
{
  if (!is_outpost(this->name) && update_count_ > 3 && !this->diverged()) {
    is_converged_ = true;
  }

  //前哨站特殊判断
  if (is_outpost(this->name) && update_count_ > 10 && !this->diverged()) {
    is_converged_ = true;
  }

  return is_converged_;
}

// 计算出装甲板中心的坐标（考虑长短轴）
Eigen::Vector3d Target::h_armor_xyz(const Eigen::VectorXd & x, int id) const
{
  auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
  auto use_outpost_model = is_outpost(name);
  auto use_l_h = !use_outpost_model && (armor_num_ == 4) && (id == 1 || id == 3);

  auto r = use_outpost_model ? OUTPOST_ARMOR_RADIUS : ((use_l_h) ? x[8] + x[9] : x[8]);
  auto armor_x = x[0] - r * std::cos(angle);
  auto armor_y = x[2] - r * std::sin(angle);
  auto armor_z = use_outpost_model
                   ? x[4] + outpost_armor_z_offset(id, outpost_height_phase_, outpost_height_direction_)
                   : ((use_l_h) ? x[4] + x[10] : x[4]);

  return {armor_x, armor_y, armor_z};
}

Eigen::MatrixXd Target::h_jacobian(const Eigen::VectorXd & x, int id) const
{
  auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
  auto use_outpost_model = is_outpost(name);
  auto use_l_h = !use_outpost_model && (armor_num_ == 4) && (id == 1 || id == 3);

  auto r = use_outpost_model ? OUTPOST_ARMOR_RADIUS : ((use_l_h) ? x[8] + x[9] : x[8]);
  auto dx_da = r * std::sin(angle);
  auto dy_da = -r * std::cos(angle);

  auto dx_dr = use_outpost_model ? 0.0 : -std::cos(angle);
  auto dy_dr = use_outpost_model ? 0.0 : -std::sin(angle);
  auto dx_dl = (use_l_h) ? -std::cos(angle) : 0.0;
  auto dy_dl = (use_l_h) ? -std::sin(angle) : 0.0;

  // outpost 的三块板高度是固定常量，不把它们当成 EKF 状态去估计。
  auto dz_dh = (use_l_h) ? 1.0 : 0.0;

  // clang-format off
  Eigen::MatrixXd H_armor_xyza{
    {1, 0, 0, 0, 0, 0, dx_da, 0, dx_dr, dx_dl,     0},
    {0, 0, 1, 0, 0, 0, dy_da, 0, dy_dr, dy_dl,     0},
    {0, 0, 0, 0, 1, 0,     0, 0,     0,     0, dz_dh},
    {0, 0, 0, 0, 0, 0,     1, 0,     0,     0,     0}
  };
  // clang-format on

  Eigen::VectorXd armor_xyz = h_armor_xyz(x, id);
  Eigen::MatrixXd H_armor_ypd = tools::xyz2ypd_jacobian(armor_xyz);
  // clang-format off
  Eigen::MatrixXd H_armor_ypda{
    {H_armor_ypd(0, 0), H_armor_ypd(0, 1), H_armor_ypd(0, 2), 0},
    {H_armor_ypd(1, 0), H_armor_ypd(1, 1), H_armor_ypd(1, 2), 0},
    {H_armor_ypd(2, 0), H_armor_ypd(2, 1), H_armor_ypd(2, 2), 0},
    {                0,                 0,                 0, 1}
  };
  // clang-format on

  return H_armor_ypda * H_armor_xyza;
}

bool Target::checkinit() { return isinit; }

}  // namespace auto_aim
