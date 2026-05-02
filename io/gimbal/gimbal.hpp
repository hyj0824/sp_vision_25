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

#ifndef IO__GIMBAL_HPP
#define IO__GIMBAL_HPP

#include <Eigen/Geometry>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <tuple>

#include "io/command.hpp"
#include "serial/serial.h"
#include "tools/crc.hpp"
#include "tools/thread_safe_queue.hpp"

namespace io
{

constexpr uint8_t SOF = 0xA5;

struct __attribute__((packed)) PacketHead
{
  uint8_t magic;
  uint16_t payload_len;
  uint8_t crc8;

  bool is_valid() const
  {
    return magic == SOF &&
           tools::check_crc8(reinterpret_cast<const uint8_t *>(this), sizeof(PacketHead));
  }
};

struct __attribute__((packed)) GimbalToVision
{
  PacketHead head{SOF, sizeof(GimbalToVision) - 8, tools::get_crc8(reinterpret_cast<const uint8_t *>(this), 3)};
  uint16_t cmd_id = 0x0002;

  uint8_t mode;          // 0: 空闲, 1: 自瞄, 2: 小符, 3: 大符
  uint8_t bullet_count;  // 子弹累计发送次数
  float q[4];            // wxyz顺序
  float yaw;
  float yaw_vel;
  float pitch;
  float pitch_vel;
  float bullet_speed;

  uint16_t crc16;
};

static_assert(sizeof(GimbalToVision) <= 64);

struct __attribute__((packed)) VisionToGimbal
{
  PacketHead head{SOF, sizeof(VisionToGimbal) - 8, tools::get_crc8(reinterpret_cast<const uint8_t *>(this), 3)};
  uint16_t cmd_id = 0x0003;

  uint8_t mode;  // 0: 不控制, 1: 控制云台但不开火，2: 控制云台且开火
  uint8_t reserved;
  float yaw;
  float yaw_vel;
  float yaw_acc;
  float pitch;
  float pitch_vel;
  float pitch_acc;

  uint16_t crc16;
};

static_assert(sizeof(VisionToGimbal) <= 64);

enum class GimbalMode
{
  IDLE,        // 空闲
  AUTO_AIM,    // 自瞄
  SMALL_BUFF,  // 小符
  BIG_BUFF     // 大符
};

struct GimbalState
{
  float yaw = 0;
  float yaw_vel = 0;
  float pitch = 0;
  float pitch_vel = 0;
  float bullet_speed = 0;
  uint16_t bullet_count = 0;
};

class Gimbal
{
public:
  Gimbal(const std::string & config_path);

  ~Gimbal();

  GimbalMode mode() const;
  GimbalState state() const;
  std::string str(GimbalMode mode) const;
  Eigen::Quaterniond q(std::chrono::steady_clock::time_point t);

  void send(
    bool control, bool fire, float yaw, float yaw_vel, float yaw_acc, float pitch, float pitch_vel,
    float pitch_acc);
  void send(const Command & command);

private:
  enum class ReadStatus { OK, TIMEOUT, INVALID, ERROR };

  serial::Serial serial_;
  mutable std::shared_mutex serial_mutex_;
  std::atomic<bool> serial_available_ = false;
  std::atomic<uint64_t> dropped_tx_count_ = 0;

  std::thread thread_;
  std::atomic<bool> quit_ = false;
  mutable std::mutex mutex_;

  GimbalToVision rx_data_;
  uint64_t bad_rx_packet_count_ = 0;

  GimbalMode mode_ = GimbalMode::IDLE;
  GimbalState state_;
  tools::ThreadSafeQueue<std::tuple<Eigen::Quaterniond, std::chrono::steady_clock::time_point>>
    queue_{1000};

  bool write(const uint8_t * buffer, size_t size);
  ReadStatus read(uint8_t * buffer, size_t size);
  ReadStatus read_header(PacketHead & head);
  ReadStatus read_packet(std::chrono::steady_clock::time_point & timestamp);
  void read_thread();
  void reconnect();
  void log_bad_rx_packet(const std::string & reason);
  void on_valid_rx_packet();
  void mark_serial_unavailable(const char * operation, const std::string & reason);
};

}  // namespace io

#endif  // IO__GIMBAL_HPP
