#include "io/gimbal/gimbal.hpp"
#include "tools/crc.hpp"
#include "tools/math_tools.hpp"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace io::protocol::legacy {

constexpr uint8_t SOF = 0xA5;

// MCU/legacy 串口的姿态约定来自老视觉 artinx-hub:
//   src/Sensor/CameraBase.cpp::clcTfRobot2Camera
// 即 Y-up 坐标系：yaw 绕 +Y, pitch 绕 +X, roll 绕 +Z；前 = -Z, 左 = -X。
// 新视觉是 FLU (X 前 / Y 左 / Z 上)：yaw 绕 +Z, pitch 绕 +Y, roll 绕 +X。
// 换基矩阵 v_new = kMcuToFlu * v_old，行：new_X=-old_Z, new_Y=-old_X, new_Z=+old_Y。
inline const Eigen::Matrix3d kMcuToFlu = (Eigen::Matrix3d() <<
     0,  0, -1,
    -1,  0,  0,
     0,  1,  0).finished();

inline float decompress_float(uint16_t raw, float min, float precision) {
  return static_cast<float>(raw) * precision + min;
}

inline uint16_t compress_float(float data, float min, float precision) {
  assert(data >= min);
  assert(data <= min + 65535.0f * precision);
  return static_cast<uint16_t>((data - min) / precision);
}

struct __attribute__((packed)) PacketHead {
  uint8_t magic;
  uint8_t payload_len;
  uint16_t reserved;
  uint8_t crc8;

  bool is_valid() const {
    return magic == SOF &&
           tools::check_crc8(reinterpret_cast<const uint8_t *>(this),
                             sizeof(PacketHead));
  }
};

template <uint8_t payload_len>
constexpr PacketHead make_head() {
  PacketHead head{SOF, payload_len, 0, 0};
  head.crc8 = tools::get_crc8(reinterpret_cast<const uint8_t *>(&head), 4);
  return head;
}

struct __attribute__((packed)) GimbalToVision {
  static constexpr uint8_t CMD_ID = 0x0A;
  static constexpr uint8_t PAYLOAD_LEN = 27 - 5 - 1 - 2;

  PacketHead head{make_head<PAYLOAD_LEN>()};
  uint8_t cmd_id{CMD_ID};

  uint16_t yaw;    // raw * 0.0005 - 4.0
  uint16_t pitch;  // raw * 0.0005 - 4.0
  uint16_t roll;   // raw * 0.0005 - 4.0
  uint16_t speedx; // raw * 0.01 - 20.0
  uint16_t speedy; // raw * 0.01 - 20.0
  uint8_t mask;
  uint16_t bullet_speed;  // raw * 0.005 - 1.0
  uint16_t cap_energy;    // raw * 0.1 - 1.0
  uint16_t chassis_power; // raw * 0.01 - 1.0
  uint16_t reserved;      // 预留字段，暂未使用
  uint16_t crc16;

  uint8_t color() const { return mask & 0x01; }

  uint8_t energy_mode() const {
    if ((mask >> 1) & 0x01)
      return 1;
    if ((mask >> 2) & 0x01)
      return 2;
    return 0;
  }

  uint8_t prior_num() const { return mask >> 3; }
};

static_assert(sizeof(GimbalToVision) == 27);

struct __attribute__((packed)) VisionToGimbal {
  static constexpr uint8_t CMD_ID = 0x0F;
  static constexpr uint8_t PAYLOAD_LEN = 15 - 5 - 1 - 2;

  PacketHead head{make_head<PAYLOAD_LEN>()};
  uint8_t cmd_id{CMD_ID};

  uint16_t yaw;             // raw = (yaw + 4.0) / 0.0005
  uint16_t pitch;           // raw = (pitch + 4.0) / 0.0005
  uint16_t horizontal_dist; // raw = (dist + 4.0) / 0.0005
  uint8_t flags;
  uint16_t crc16;

  void set_flags(bool fire, bool has_target, uint8_t target_type) {
    flags = static_cast<uint8_t>((fire ? 1 : 0) | ((has_target ? 1 : 0) << 1) |
                                 (target_type << 2));
  }
};

static_assert(sizeof(VisionToGimbal) == 15);

class LegacyProtocol : public BaseProtocol {
public:
  void serialize(uint8_t *buffer, size_t len, bool control, bool fire,
                 float yaw, float yaw_vel, float yaw_acc, float pitch,
                 float pitch_vel, float pitch_acc) const override {
    assert(len >= sizeof(VisionToGimbal));

    // 视觉给的 yaw/pitch 是 FLU 下的 ZYX；先在 FLU 构造旋转，
    // 再共轭回 MCU 的 Y-up 帧，最后用 (Y, X, Z) 顺序拆出 MCU 的 yaw/pitch。
    const Eigen::Matrix3d R_flu =
        Eigen::AngleAxisd(yaw,   Eigen::Vector3d::UnitZ()).toRotationMatrix() *
        Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()).toRotationMatrix();
    const Eigen::Matrix3d R_mcu = kMcuToFlu.transpose() * R_flu * kMcuToFlu;
    const Eigen::Vector3d ypr_mcu = tools::eulers(R_mcu, 1, 0, 2, false);
    const float old_yaw = static_cast<float>(ypr_mcu[0]);
    const float old_pitch = static_cast<float>(ypr_mcu[1]);

    VisionToGimbal tx_data{};
    tx_data.yaw = compress_float(old_yaw, -4.0f, 0.0005f);
    tx_data.pitch = compress_float(old_pitch, -4.0f, 0.0005f);

    // 当前 BaseProtocol 参数里没有目标距离；不要留 raw=0，否则下位机会解成 -4.0。
    tx_data.horizontal_dist = compress_float(0.0f, -4.0f, 0.0005f);

    tx_data.set_flags(fire, control, 0);

    tx_data.crc16 =
        tools::get_crc16(reinterpret_cast<const uint8_t *>(&tx_data),
                         sizeof(tx_data) - sizeof(tx_data.crc16));

    std::memcpy(buffer, &tx_data, sizeof(tx_data));
  }

  bool parse(const uint8_t *buffer, size_t len) override {
    if (len < sizeof(GimbalToVision)) {
      return false;
    }

    const auto &rx_data = reinterpret_cast<const GimbalToVision &>(*buffer);

    if (!rx_data.head.is_valid()) {
      return false;
    }

    const uint16_t expected_crc16 =
        tools::get_crc16(reinterpret_cast<const uint8_t *>(&rx_data),
                         sizeof(rx_data) - sizeof(rx_data.crc16));

    if (rx_data.crc16 != expected_crc16) {
      return false;
    }

    const float yaw = decompress_float(rx_data.yaw, -4.0f, 0.0005f);
    const float pitch = decompress_float(rx_data.pitch, -4.0f, 0.0005f);
    const float roll = decompress_float(rx_data.roll, -4.0f, 0.0005f);

    mode_ = static_cast<GimbalMode>(rx_data.color() + 1);

    // 用 MCU 的轴约定构建旋转：yaw→Y, pitch→X, roll→Z。
    const Eigen::Matrix3d R_mcu =
        Eigen::AngleAxisd(yaw,   Eigen::Vector3d::UnitY()).toRotationMatrix() *
        Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitX()).toRotationMatrix() *
        Eigen::AngleAxisd(roll,  Eigen::Vector3d::UnitZ()).toRotationMatrix();
    const Eigen::Matrix3d R_flu = kMcuToFlu * R_mcu * kMcuToFlu.transpose();

    // FLU 下用 intrinsic ZYX 拆出 yaw / pitch 给上层使用。
    const Eigen::Vector3d ypr_flu = tools::eulers(R_flu, 2, 1, 0, false);
    state_.yaw = ypr_flu[0];
    state_.pitch = ypr_flu[1];
    state_.bullet_speed = decompress_float(rx_data.bullet_speed, -1.0f, 0.005f);

    q_ = Eigen::Quaterniond(R_flu).normalized();

    return true;
  }
};

} // namespace io::protocol::legacy
