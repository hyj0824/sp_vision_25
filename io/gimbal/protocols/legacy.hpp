#include "io/gimbal/gimbal.hpp"
#include "tools/crc.hpp"
#include "tools/math_tools.hpp"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace io::protocol::legacy {

constexpr uint8_t SOF = 0xA5;

template <float min, float precision>
inline float decompress_float(uint16_t raw) {
  return static_cast<float>(raw) * precision + min;
}

template <float min, float precision>
inline uint16_t compress_float(float data) {
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

    VisionToGimbal tx_data{};
    tx_data.yaw = compress_float<-4.0f, 0.0005f>(yaw);
    tx_data.pitch = compress_float<-4.0f, 0.0005f>(pitch);

    // 当前 BaseProtocol 参数里没有目标距离；不要留 raw=0，否则下位机会解成
    // -4.0。
    tx_data.horizontal_dist = compress_float<-4.0f, 0.0005f>(0.0f);

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

    const float yaw = decompress_float<-4.0f, 0.0005f>(rx_data.yaw);
    const float pitch = decompress_float<-4.0f, 0.0005f>(rx_data.pitch);
    const float roll = decompress_float<-4.0f, 0.0005f>(rx_data.roll);

    mode_ = static_cast<GimbalMode>(rx_data.color() + 1);

    state_.yaw = yaw;
    state_.pitch = pitch;
    state_.bullet_speed = decompress_float<-1.0f, 0.005f>(rx_data.bullet_speed);

    q_ = Eigen::Quaterniond(
             tools::rotation_matrix(Eigen::Vector3d(yaw, pitch, roll)))
             .normalized();

    return true;
  }
};

} // namespace io::protocol::legacy