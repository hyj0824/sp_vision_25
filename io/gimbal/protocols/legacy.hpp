#include "io/gimbal/gimbal.hpp"
#include "tools/crc.hpp"
#include "tools/math_tools.hpp"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace io::protocol::legacy {

constexpr uint8_t SOF = 0xA5;

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

struct __attribute__((packed)) GimbalToVision {
  PacketHead head{SOF, sizeof(GimbalToVision) - sizeof(PacketHead) - 3, 0,
                  tools::get_crc8(reinterpret_cast<const uint8_t *>(this), 4)};

  uint8_t cmd_id = 0xFF;

  float yaw;
  float pitch;
  float roll;
  // 底盘速度，目前是全0
  float speedx;
  float speedy;

  uint8_t color : 1;         // 0: red, 1: blue
  uint8_t energy_mode : 2;   // 0: 无符, 1: 小符, 2: 大符
  uint8_t auto_aim_mode : 5; // reserved

  float bullet_speed;
  float cap_energy;
  float chassis_power;

  uint16_t crc16;
};

static_assert(sizeof(GimbalToVision) <= 64);

struct __attribute__((packed)) VisionToGimbal {
  PacketHead head{SOF, sizeof(VisionToGimbal) - sizeof(PacketHead) - 3, 0,
                  tools::get_crc8(reinterpret_cast<const uint8_t *>(this), 4)};

  uint8_t cmd_id = 0xFF;

  uint16_t yaw;            // raw = (yaw + 4.0) / 0.0005
  uint16_t pitch;          // raw = (pitch + 4.0) / 0.0005
  uint16_t horizontalDist; // reserved
  uint8_t isFire : 1;
  uint8_t hasTarget : 1;
  uint8_t targetType : 6;
  // bit0   isFire
  // bit1   hasTargets
  // bit2-7 targetType = flags >> 2 (reserved)

  uint16_t crc16;
};

static_assert(sizeof(VisionToGimbal) <= 64);

class LegacyProtocol : public BaseProtocol {
public:
  void serialize(uint8_t *buffer, size_t len, bool control, bool fire,
                 float yaw, float yaw_vel, float yaw_acc, float pitch,
                 float pitch_vel, float pitch_acc) const override {
    assert(len >= sizeof(VisionToGimbal));
    VisionToGimbal tx_data{};
    tx_data.yaw = yaw;
    tx_data.pitch = pitch;
    tx_data.hasTarget = control;
    tx_data.isFire = fire;
    tx_data.crc16 =
        tools::get_crc16(reinterpret_cast<const uint8_t *>(&tx_data),
                         sizeof(tx_data) - sizeof(tx_data.crc16));
    std::memcpy(buffer, &tx_data, sizeof(tx_data));
  }
  bool parse(const uint8_t *buffer, size_t len) override {
    assert(len >= sizeof(GimbalToVision));
    auto &rx_data = reinterpret_cast<const GimbalToVision &>(*buffer);
    if (!rx_data.head.is_valid()) {
      return false;
    }
    uint8_t mode = rx_data.color + 1;
    mode_ = static_cast<GimbalMode>(mode);
    state_.yaw = rx_data.yaw;
    state_.pitch = rx_data.pitch;
    state_.bullet_speed = rx_data.bullet_speed;

    q_ = Eigen::Quaterniond(tools::rotation_matrix(Eigen::Vector3d(
                                rx_data.yaw, rx_data.pitch, rx_data.roll)))
             .normalized();

    return true;
  }
};

} // namespace io::protocol::legacy
