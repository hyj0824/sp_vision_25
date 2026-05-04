#include "io/gimbal/gimbal.hpp"
#include "tools/crc.hpp"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace io::protocol::neo {

/**
 *   数据帧
 *   ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬─────────┬───────┬───────┐
 *   │ SOF  │LEN_L │LEN_H │ CRC8 │ID_L  │ ID_H │FLG_L │FLG_H │ DATA... │CRC16_L│CRC16_H│
 *   │ 0xA5 │      │      │      │      │      │      │      │ float[] │       │       │
 *   └──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴─────────┴───────┴───────┘
 *   CRC8  : Get_CRC8([0,1,2])
 *   CRC16 : Get_CRC16(buf[0..total-3], init=0xFFFF)，存于末 2 字节
 *   LEN (payload_len) = flags(2B) + float数组(N×4B)
 *                     = total_len - 8
 *
 *   CMD_ID_IMU     = 0x0002  下位机→视觉
 *   CMD_ID_VISION  = 0x0003  视觉→下位机
 */

constexpr uint8_t SOF = 0xA5;

struct __attribute__((packed)) PacketHead {
  uint8_t magic;
  uint16_t payload_len;
  uint8_t crc8;

  bool is_valid() const {
    return magic == SOF &&
           tools::check_crc8(reinterpret_cast<const uint8_t *>(this),
                             sizeof(PacketHead));
  }
};

template <uint16_t payload_len>
inline constexpr PacketHead make_head() {
  PacketHead head{SOF, payload_len, 0};
  head.crc8 = tools::get_crc8(reinterpret_cast<const uint8_t *>(&head), 3);
  return head;
}

struct __attribute__((packed)) GimbalToVision {
  PacketHead head = make_head<sizeof(GimbalToVision) - 8>();
  uint16_t cmd_id = 0x0002;

  uint8_t mode;         // 0: 空闲, 1: 自瞄, 2: 小符, 3: 大符
  uint8_t bullet_count; // 子弹累计发送次数
  float q[4];           // wxyz顺序
  float yaw;
  float yaw_vel;
  float pitch;
  float pitch_vel;
  float bullet_speed;

  uint16_t crc16;
};

static_assert(sizeof(GimbalToVision) == 46);

struct __attribute__((packed)) VisionToGimbal {
  PacketHead head = make_head<sizeof(VisionToGimbal) - 8>();
  uint16_t cmd_id = 0x0003;

  uint8_t mode; // 0: 不控制, 1: 控制云台但不开火，2: 控制云台且开火
  uint8_t reserved;
  float yaw;
  float yaw_vel;
  float yaw_acc;
  float pitch;
  float pitch_vel;
  float pitch_acc;

  uint16_t crc16;
};

static_assert(sizeof(VisionToGimbal) == 34);

class NeoProtocol : public BaseProtocol {
public:
  void serialize(uint8_t *buffer, size_t len, bool control, bool fire,
                 float yaw, float yaw_vel, float yaw_acc, float pitch,
                 float pitch_vel, float pitch_acc) const override {
    assert(len >= sizeof(VisionToGimbal));
    VisionToGimbal tx_data{};
    tx_data.mode = control ? (fire ? 2 : 1) : 0;
    tx_data.yaw = yaw;
    tx_data.yaw_vel = yaw_vel;
    tx_data.yaw_acc = yaw_acc;
    tx_data.pitch = pitch;
    tx_data.pitch_vel = pitch_vel;
    tx_data.pitch_acc = pitch_acc;
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

    if (!tools::check_crc16(buffer, sizeof(GimbalToVision))) {
      return false;
    }

    mode_ = static_cast<GimbalMode>(rx_data.mode);
    state_.yaw = rx_data.yaw;
    state_.yaw_vel = rx_data.yaw_vel;
    state_.pitch = rx_data.pitch;
    state_.pitch_vel = rx_data.pitch_vel;
    state_.bullet_speed = rx_data.bullet_speed;
    state_.bullet_count = rx_data.bullet_count;
    q_ = Eigen::Quaterniond(rx_data.q[0], rx_data.q[1], rx_data.q[2],
                            rx_data.q[3]);
    return true;
  }
};
} // namespace io::protocol::neo
