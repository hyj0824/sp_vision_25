#include "gimbal.hpp"
#include "protocols/legacy.hpp"
#include "protocols/neo.hpp"

#include "tools/crc.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/yaml.hpp"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace io {
namespace {
using namespace std::chrono_literals;

constexpr size_t RX_BUFFER_SIZE = 16384;
constexpr size_t TX_BUFFER_SIZE = 64;
constexpr int RECONNECT_LOG_INTERVAL = 10;
constexpr uint32_t SERIAL_TIMEOUT_MS = 50;
constexpr auto RECONNECT_DELAY = 1s;

} // namespace

Gimbal::Gimbal(const std::string &config_path) {
  auto yaml = tools::load(config_path);
  auto com_port = tools::read<std::string>(yaml, "com_port");
  auto protocol = tools::read<std::string>(yaml, "protocol");

  if (protocol == "neo") {
    protocol_ = std::make_unique<protocol::neo::NeoProtocol>();
    rx_packet_size_ = sizeof(protocol::neo::GimbalToVision);
    tx_packet_size_ = sizeof(protocol::neo::VisionToGimbal);
  } else if (protocol == "legacy") {
    protocol_ = std::make_unique<protocol::legacy::LegacyProtocol>();
    rx_packet_size_ = sizeof(protocol::legacy::GimbalToVision);
    tx_packet_size_ = sizeof(protocol::legacy::VisionToGimbal);
  } else {
    throw std::runtime_error("Unsupported gimbal protocol: " + protocol);
  }

  auto timeout = serial::Timeout::simpleTimeout(SERIAL_TIMEOUT_MS);
  serial_.setPort(com_port);
  serial_.setTimeout(timeout);

  thread_ = std::thread(&Gimbal::read_thread, this);
}

Gimbal::~Gimbal() {
  quit_ = true;
  if (thread_.joinable())
    thread_.join();

  try {
    if (serial_.isOpen())
      serial_.close();
  } catch (const std::exception &e) {
    tools::logger()->debug("[Gimbal] Ignored serial close failure: {}",
                           e.what());
  }
}

GimbalMode Gimbal::mode() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return protocol_->mode();
}

GimbalState Gimbal::state() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return protocol_->state();
}

std::string Gimbal::str(GimbalMode mode) const {
  switch (mode) {
  case GimbalMode::IDLE:
    return "IDLE";
  case GimbalMode::AUTO_AIM:
    return "AUTO_AIM";
  case GimbalMode::SMALL_BUFF:
    return "SMALL_BUFF";
  case GimbalMode::BIG_BUFF:
    return "BIG_BUFF";
  default:
    return "INVALID";
  }
}

Eigen::Quaterniond Gimbal::q(std::chrono::steady_clock::time_point t) {
  if (queue_.empty()) {
    mark_serial_error("q", "no data in queue");
    return Eigen::Quaterniond::Identity();
  }
  while (true) {
    auto [q_a, t_a] = queue_.pop();
    auto [q_b, t_b] = queue_.front();
    auto t_ab = tools::delta_time(t_a, t_b);
    auto t_ac = tools::delta_time(t_a, t);
    auto k = t_ac / t_ab;
    Eigen::Quaterniond q_c = q_a.slerp(k, q_b).normalized();
    if (t < t_a)
      return q_c;
    if (!(t_a < t && t <= t_b))
      continue;

    return q_c;
  }
}

void Gimbal::send(bool control, bool fire, float yaw, float yaw_vel,
                  float yaw_acc, float pitch, float pitch_vel,
                  float pitch_acc) {
  std::vector<uint8_t> buffer(tx_packet_size_);
  protocol_->serialize(buffer.data(), tx_packet_size_, control, fire, yaw,
                       yaw_vel, yaw_acc, pitch, pitch_vel, pitch_acc);
  write(buffer.data(), tx_packet_size_);
}

void Gimbal::send(const Command &command) {
  send(command.control, command.shoot, command.yaw, 0, 0, command.pitch, 0, 0);
}

void Gimbal::write(const uint8_t *buffer, size_t size) {
  // 这是个回调，不应该在这里重试连接，重试连接应该在read_thread里做
  if (!serial_ok_ || !serial_.isOpen())
    return;

  try {
    const auto written = serial_.write(buffer, size);
    if (written != size) {
      mark_serial_error("write", "short write");
    }
  } catch (const std::exception &e) {
    mark_serial_error("write", e.what());
  }
}

void Gimbal::read_thread() {
  tools::logger()->info("[Gimbal] read_thread started.");
  std::vector<uint8_t> buffer(RX_BUFFER_SIZE);
  auto it = buffer.begin();

  while (!quit_) {
    if (!serial_ok_) {
      reconnect();
      continue;
    }

    try {
      std::vector<uint8_t> mbuff(rx_packet_size_ * 2);
      size_t read_size = serial_.read(mbuff, rx_packet_size_ * 2);

      if (buffer.size() + read_size > buffer.capacity()) {
        // 循环不会读非完整包
        std::vector<uint8_t> tmp(it, buffer.end());
        buffer.clear();
        buffer.insert(buffer.end(), tmp.begin(), tmp.end());
        it = buffer.begin();
      }
      buffer.insert(buffer.end(), mbuff.begin(), mbuff.begin() + read_size);

      for (; it + rx_packet_size_ <= buffer.end(); ++it) {
        bool valid = false;
        {
          std::lock_guard<std::mutex> lock(mutex_);
          valid = protocol_->parse(it.base(), rx_packet_size_);
        }
        if (valid) {
          queue_.push({protocol_->q(), std::chrono::steady_clock::now()});
          break;
        }
      }
    } catch (const std::exception &e) {
      mark_serial_error("read", e.what());
    }
  }

  tools::logger()->info("[Gimbal] read_thread stopped.");
}

void Gimbal::reconnect() {
  int attempt = 0;
  while (!quit_ && !serial_ok_) {
    ++attempt;

    try {
      if (serial_.isOpen())
        serial_.close();
      serial_.open();
      serial_.flushInput();

      serial_ok_ = true;
      queue_.clear();
      tools::logger()->info("[Gimbal] Serial reconnected.");
      return;
    } catch (const std::exception &e) {
      serial_ok_ = false;
      if (attempt == 1 || attempt % RECONNECT_LOG_INTERVAL == 0) {
        tools::logger()->warn("[Gimbal] Serial reconnect attempt {} failed: {}",
                              attempt, e.what());
      }
    }

    std::this_thread::sleep_for(RECONNECT_DELAY);
  }
}

void Gimbal::mark_serial_error(const char *operation,
                               const std::string &reason) {
  if (!serial_ok_.exchange(false))
    return;

  tools::logger()->error("[Gimbal] During {}: {}", operation,
                        reason);
}

} // namespace io
