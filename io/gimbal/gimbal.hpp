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
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "io/command.hpp"
#include "serial/serial.h"
#include "tools/crc.hpp"
#include "tools/thread_safe_queue.hpp"

namespace io {

enum class GimbalMode {
  IDLE,       // 空闲
  AUTO_AIM,   // 自瞄
  SMALL_BUFF, // 小符
  BIG_BUFF    // 大符
};

struct GimbalState {
  float yaw = 0;
  float yaw_vel = 0;
  float pitch = 0;
  float pitch_vel = 0;
  float bullet_speed = 0;
  uint16_t bullet_count = 0;
};

class BaseProtocol {
protected:
  GimbalMode mode_{GimbalMode::IDLE};
  GimbalState state_{};
  Eigen::Quaterniond q_{};

public:
  virtual GimbalMode mode() const { return mode_; }
  virtual GimbalState state() const { return state_; }
  virtual Eigen::Quaterniond q() const { return q_; }
  virtual void serialize(uint8_t *buffer, size_t len, bool control, bool fire,
                         float yaw, float yaw_vel, float yaw_acc, float pitch,
                         float pitch_vel, float pitch_acc) const = 0;
  virtual bool parse(const uint8_t *buffer, size_t len) = 0;
  virtual ~BaseProtocol() = default;
};

class Gimbal {
public:
  Gimbal(const std::string &config_path);

  ~Gimbal();

  GimbalMode mode() const;
  GimbalState state() const;
  std::string str(GimbalMode mode) const;
  Eigen::Quaterniond q(std::chrono::steady_clock::time_point t);

  void send(bool control, bool fire, float yaw, float yaw_vel, float yaw_acc,
            float pitch, float pitch_vel, float pitch_acc);
  void send(const Command &command);

private:
  serial::Serial serial_;
  std::atomic<bool> serial_ok_ = false;

  std::thread thread_;
  std::atomic<bool> quit_ = false;

  tools::ThreadSafeQueue<
      std::tuple<Eigen::Quaterniond, std::chrono::steady_clock::time_point>>
      queue_{1000};
  
  mutable std::mutex mutex_;
  std::unique_ptr<BaseProtocol> protocol_;
  size_t tx_packet_size_{};
  size_t rx_packet_size_{};

  void write(const uint8_t *buffer, size_t size);
  void read_thread();
  void reconnect();
  void mark_serial_error(const char *operation, const std::string &reason);
};

} // namespace io

#endif // IO__GIMBAL_HPP
