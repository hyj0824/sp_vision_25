#include "gimbal.hpp"

#include "tools/crc.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/yaml.hpp"

#include <chrono>
#include <cstring>
#include <shared_mutex>
#include <string>
#include <thread>

namespace io {
namespace {
using namespace std::chrono_literals;

constexpr int RECONNECT_LOG_INTERVAL = 10;
constexpr uint64_t BAD_RX_PACKET_RECONNECT_THRESHOLD = 5;
constexpr uint32_t SERIAL_TIMEOUT_MS = 50;
constexpr auto RECONNECT_DELAY = 1s;

std::string compact_exception_message(const std::exception &e) {
  std::string message = e.what();
  const auto file_pos = message.find(", file ");
  if (file_pos != std::string::npos)
    message.erase(file_pos);
  return message;
}
} // namespace

Gimbal::Gimbal(const std::string &config_path) {
  auto yaml = tools::load(config_path);
  auto com_port = tools::read<std::string>(yaml, "com_port");

  try {
    auto timeout = serial::Timeout::simpleTimeout(SERIAL_TIMEOUT_MS);
    serial_.setPort(com_port);
    serial_.setTimeout(timeout);
    serial_.open();
    serial_.flushInput();
    serial_available_ = true;
    tools::logger()->info("[Gimbal] Serial opened: {}", com_port);
  } catch (const std::exception &e) {
    serial_available_ = false;
    tools::logger()->warn("[Gimbal] Serial {} is not available yet: {}",
                          com_port, compact_exception_message(e));
  }

  thread_ = std::thread(&Gimbal::read_thread, this);
}

Gimbal::~Gimbal() {
  quit_ = true;
  if (thread_.joinable())
    thread_.join();

  std::unique_lock<std::shared_mutex> lock(serial_mutex_);
  try {
    if (serial_.isOpen())
      serial_.close();
  } catch (const std::exception &e) {
    tools::logger()->debug("[Gimbal] Ignored serial close failure: {}",
                           compact_exception_message(e));
  }
}

GimbalMode Gimbal::mode() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return mode_;
}

GimbalState Gimbal::state() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return state_;
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
  VisionToGimbal tx_data;
  tx_data.mode = control ? (fire ? 2 : 1) : 0;
  tx_data.yaw = yaw;
  tx_data.yaw_vel = yaw_vel;
  tx_data.yaw_acc = yaw_acc;
  tx_data.pitch = pitch;
  tx_data.pitch_vel = pitch_vel;
  tx_data.pitch_acc = pitch_acc;
  tx_data.crc16 = tools::get_crc16(reinterpret_cast<const uint8_t *>(&tx_data),
                                   sizeof(tx_data) - sizeof(tx_data.crc16));

  write(reinterpret_cast<const uint8_t *>(&tx_data), sizeof(tx_data));
}

void Gimbal::send(const Command &command) {
  send(command.control, command.shoot, command.yaw, 0, 0, command.pitch, 0, 0);
}

bool Gimbal::write(const uint8_t *buffer, size_t size) {
  if (!serial_available_) {
    ++dropped_tx_count_;
    return false;
  }

  std::shared_lock<std::shared_mutex> lock(serial_mutex_);
  if (!serial_.isOpen()) {
    ++dropped_tx_count_;
    mark_serial_unavailable("write", "serial port is closed");
    return false;
  }

  try {
    const auto written = serial_.write(buffer, size);
    if (written == size)
      return true;

    ++dropped_tx_count_;
    mark_serial_unavailable("write", "short write");
    return false;
  } catch (const std::exception &e) {
    ++dropped_tx_count_;
    mark_serial_unavailable("write", compact_exception_message(e));
    return false;
  }
}

Gimbal::ReadStatus Gimbal::read(uint8_t *buffer, size_t size) {
  if (!serial_available_)
    return ReadStatus::ERROR;

  std::shared_lock<std::shared_mutex> lock(serial_mutex_);
  if (!serial_.isOpen()) {
    mark_serial_unavailable("read", "serial port is closed");
    return ReadStatus::ERROR;
  }

  try {
    const auto read_size = serial_.read(buffer, size);
    if (read_size == size)
      return ReadStatus::OK;
    return ReadStatus::TIMEOUT;
  } catch (const std::exception &e) {
    mark_serial_unavailable("read", compact_exception_message(e));
    return ReadStatus::ERROR;
  }
}

Gimbal::ReadStatus Gimbal::read_header(PacketHead &head) {
  uint8_t buffer[sizeof(PacketHead)]{};
  size_t buffered = 0;

  while (!quit_) {
    uint8_t byte = 0;
    auto status = read(&byte, 1);
    if (status != ReadStatus::OK)
      return status;

    if (buffered == 0) {
      if (byte != SOF)
        continue;
      buffer[buffered++] = byte;
      continue;
    }

    buffer[buffered++] = byte;
    if (buffered < sizeof(PacketHead))
      continue;

    std::memcpy(&head, buffer, sizeof(PacketHead));
    if (head.is_valid())
      return ReadStatus::OK;

    buffered = 0;
    for (size_t i = 1; i < sizeof(PacketHead); ++i) {
      if (buffer[i] == SOF) {
        buffered = sizeof(PacketHead) - i;
        std::memmove(buffer, buffer + i, buffered);
        break;
      }
    }
  }

  return ReadStatus::ERROR;
}

Gimbal::ReadStatus
Gimbal::read_packet(std::chrono::steady_clock::time_point &timestamp) {
  PacketHead head;
  auto status = read_header(head);
  if (status != ReadStatus::OK)
    return status;

  constexpr uint16_t expected_payload_len = sizeof(GimbalToVision) - 8;
  if (head.payload_len != expected_payload_len) {
    log_bad_rx_packet(
        "invalid payload_len: " + std::to_string(head.payload_len) +
        ", expected: " + std::to_string(expected_payload_len));
    return serial_available_ ? ReadStatus::INVALID : ReadStatus::ERROR;
  }

  rx_data_.head = head;
  timestamp = std::chrono::steady_clock::now();

  status = read(reinterpret_cast<uint8_t *>(&rx_data_) + sizeof(rx_data_.head),
                sizeof(rx_data_) - sizeof(rx_data_.head));
  if (status != ReadStatus::OK)
    return status;

  if (!tools::check_crc16(reinterpret_cast<uint8_t *>(&rx_data_),
                          sizeof(rx_data_))) {
    log_bad_rx_packet("CRC16 check failed");
    return serial_available_ ? ReadStatus::INVALID : ReadStatus::ERROR;
  }

  if (rx_data_.cmd_id != 0x0002) {
    log_bad_rx_packet("invalid cmd_id: " + std::to_string(rx_data_.cmd_id));
    return serial_available_ ? ReadStatus::INVALID : ReadStatus::ERROR;
  }

  return ReadStatus::OK;
}

void Gimbal::read_thread() {
  tools::logger()->info("[Gimbal] read_thread started.");

  while (!quit_) {
    if (!serial_available_) {
      reconnect();
      continue;
    }

    std::chrono::steady_clock::time_point t;
    auto status = read_packet(t);
    if (status == ReadStatus::ERROR) {
      reconnect();
      continue;
    }
    if (status == ReadStatus::TIMEOUT || status == ReadStatus::INVALID) {
      continue;
    }

    on_valid_rx_packet();
    Eigen::Quaterniond q(rx_data_.q[0], rx_data_.q[1], rx_data_.q[2],
                         rx_data_.q[3]);
    queue_.push({q, t});

    std::lock_guard<std::mutex> lock(mutex_);

    state_.yaw = rx_data_.yaw;
    state_.yaw_vel = rx_data_.yaw_vel;
    state_.pitch = rx_data_.pitch;
    state_.pitch_vel = rx_data_.pitch_vel;
    state_.bullet_speed = rx_data_.bullet_speed;
    state_.bullet_count = rx_data_.bullet_count;

    switch (rx_data_.mode) {
    case 0:
      mode_ = GimbalMode::IDLE;
      break;
    case 1:
      mode_ = GimbalMode::AUTO_AIM;
      break;
    case 2:
      mode_ = GimbalMode::SMALL_BUFF;
      break;
    case 3:
      mode_ = GimbalMode::BIG_BUFF;
      break;
    default:
      mode_ = GimbalMode::IDLE;
      tools::logger()->warn("[Gimbal] Invalid mode: {}", rx_data_.mode);
      break;
    }
  }

  tools::logger()->info("[Gimbal] read_thread stopped.");
}

void Gimbal::reconnect() {
  int attempt = 0;
  while (!quit_ && !serial_available_) {
    ++attempt;

    try {
      std::unique_lock<std::shared_mutex> lock(serial_mutex_);
      if (serial_.isOpen())
        serial_.close();
      serial_.open();
      serial_.flushInput();

      serial_available_ = true;
      bad_rx_packet_count_ = 0;
      queue_.clear();
      const auto dropped = dropped_tx_count_.exchange(0);
      if (dropped > 0) {
        tools::logger()->warn(
            "[Gimbal] Serial reconnected after dropping {} command frame(s).",
            dropped);
      } else {
        tools::logger()->info("[Gimbal] Serial reconnected.");
      }
      return;
    } catch (const std::exception &e) {
      serial_available_ = false;
      if (attempt == 1 || attempt % RECONNECT_LOG_INTERVAL == 0) {
        tools::logger()->warn("[Gimbal] Serial reconnect attempt {} failed: {}",
                              attempt, compact_exception_message(e));
      }
    }

    std::this_thread::sleep_for(RECONNECT_DELAY);
  }
}

void Gimbal::log_bad_rx_packet(const std::string &reason) {
  ++bad_rx_packet_count_;
  if (bad_rx_packet_count_ >= BAD_RX_PACKET_RECONNECT_THRESHOLD) {
    tools::logger()->warn("[Gimbal] Too many invalid RX packets ({}), "
                          "reconnecting serial. Last error: {}",
                          bad_rx_packet_count_, reason);
    bad_rx_packet_count_ = 0;
    mark_serial_unavailable("read", "invalid packet stream");
    return;
  }

  if (bad_rx_packet_count_ == 1) {
    tools::logger()->debug("[Gimbal] Dropped invalid RX packet: {}", reason);
  }
}

void Gimbal::on_valid_rx_packet() {
  if (bad_rx_packet_count_ == 0)
    return;

  tools::logger()->debug(
      "[Gimbal] RX stream recovered after {} invalid packet(s).",
      bad_rx_packet_count_);
  bad_rx_packet_count_ = 0;
}

void Gimbal::mark_serial_unavailable(const char *operation,
                                     const std::string &reason) {
  const bool was_available = serial_available_.exchange(false);
  if (was_available) {
    tools::logger()->warn("[Gimbal] Serial disconnected during {}: {}",
                          operation, reason);
  }
}

} // namespace io
