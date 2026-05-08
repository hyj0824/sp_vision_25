# Jetson Orin NX Auto Buff Environment Setup

This document records the environment needed by the migrated rune/auto-buff code on Jetson Orin NX. The main new runtime dependencies are TensorRT/CUDA for inference and Ceres for the big-rune predictor. The armor auto-aim path still uses the existing project dependencies.

## 1. Recommended Platform

- Hardware: Jetson Orin NX
- OS: Ubuntu from NVIDIA JetPack
- Recommended JetPack: 6.x if possible. JetPack 5.x can also work, but use the TensorRT/CUDA versions shipped by that JetPack.
- Compiler: GCC/G++ 9 or newer
- CMake: 3.16 or newer

Check the board:

```bash
cat /etc/nv_tegra_release
dpkg -l | grep -E 'nvinfer|cuda-toolkit|cudnn' | head
nvcc --version
```

## 2. System Packages

Install general build dependencies:

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  cmake \
  git \
  pkg-config \
  libopencv-dev \
  libfmt-dev \
  libeigen3-dev \
  libspdlog-dev \
  libyaml-cpp-dev \
  libusb-1.0-0-dev \
  nlohmann-json3-dev
```

Install TensorRT/CUDA development packages. On Jetson, prefer JetPack packages instead of manually unpacking x86 debs:

```bash
sudo apt install -y \
  nvidia-cuda-dev \
  tensorrt \
  libnvinfer-dev \
  libnvinfer-plugin-dev \
  libnvonnxparsers-dev
```

If the package names differ on your JetPack release, inspect available packages:

```bash
apt-cache search nvinfer
apt-cache search nvonnxparser
```

Install Ceres:

```bash
sudo apt install -y libceres-dev
```

If `libceres-dev` is too old or unavailable, build Ceres from source. Keep Eigen, glog, gflags, SuiteSparse installed:

```bash
sudo apt install -y libgoogle-glog-dev libgflags-dev libsuitesparse-dev
git clone --depth 1 --branch 2.2.0 https://github.com/ceres-solver/ceres-solver.git
cmake -S ceres-solver -B ceres-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF \
  -DBUILD_EXAMPLES=OFF
cmake --build ceres-build -j$(nproc)
sudo cmake --install ceres-build
sudo ldconfig
```

## 3. Optional Camera SDKs

The current CMake only builds HikRobot support when `MVCAM_SDK_PATH` is set before configuration. This avoids breaking builds on machines without the SDK.

For HikRobot:

```bash
export MVCAM_SDK_PATH=/opt/MVS
```

Then configure from a clean build directory. If `camera_name: "hikrobot"` is used but the SDK was not enabled at build time, runtime will throw a clear error.

MindVision still needs its SDK/libs installed if `camera_name: "mindvision"` is used.

Serial permissions:

```bash
sudo usermod -a -G dialout $USER
sudo usermod -a -G video $USER
```

Log out and log back in after changing groups.

## 4. Environment Variables

For normal JetPack installations, these are usually enough:

```bash
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}
```

If TensorRT is installed in a custom path:

```bash
export TENSORRT_ROOT=/path/to/TensorRT
export LD_LIBRARY_PATH=$TENSORRT_ROOT/lib:$TENSORRT_ROOT/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
```

If Ceres is installed in a custom path:

```bash
export CERES_ROOT=/path/to/ceres/install
export CMAKE_PREFIX_PATH=$CERES_ROOT:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=$CERES_ROOT/lib:$CERES_ROOT/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
```

## 5. Configure and Build

From the repository root:

```bash
cmake -S . -B build-orin \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DBUILD_AUTO_BUFF=ON \
  -DBUILD_OMNIPERCEPTION=OFF

cmake --build build-orin -j$(nproc)
```

If Ceres was installed to a custom prefix:

```bash
cmake -S . -B build-orin \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DBUILD_AUTO_BUFF=ON \
  -DBUILD_OMNIPERCEPTION=OFF \
  -DCMAKE_PREFIX_PATH="$CERES_ROOT" \
  -DCeres_DIR="$CERES_ROOT/lib/cmake/Ceres"
```

## 6. Model and Engine Notes

TensorRT engine files are platform-specific. Do not reuse an x86 engine on Orin.

Recommended first-run behavior:

- Keep ONNX files in `assets/`.
- Delete stale engine files copied from other machines.
- Let the program build engines on Orin.

Useful cleanup:

```bash
rm -f assets/0526.engine
rm -f assets/2025-07-28-rune.engine
rm -f assets/yolo11_buff_int8.engine
```

Relevant config keys:

```yaml
yolov5_model_path: assets/0526.onnx
yolov5_engine_path: assets/0526.engine

rune_model_path: assets/2025-07-28-rune.onnx
rune_engine_path: assets/2025-07-28-rune.engine
rune_input_width: 640
rune_input_height: 384
rune_input_layout: nhwc
rune_normalize_input: false
rune_confidence_threshold: 0.7
rune_delay_time: 0.3
rune_target_radius: 0.145
rune_fan_len: 0.7
```

The new rune predictor also needs:

```yaml
rune_auto_fire: true
rune_lost_count: 20
rune_direction_window: 10
rune_min_fit_data_size: 20
rune_min_shooting_size: 30
rune_max_fit_data_size: 1200
rune_fit_interval_frames: 3
rune_stale_reset_time: 2.0
rune_center_fallback_time: 0.1
rune_Qw: 0.2
rune_Qtheta: 0.08
rune_Rtheta: 0.4
rune_reject_pnp_outliers: true
rune_log_rejected_pnp: false
rune_pnp_min_distance: 1.0
rune_pnp_max_distance: 15.0
rune_pnp_min_depth: 0.5
rune_pnp_max_reprojection_error: 8.0
rune_pnp_max_distance_jump: 2.5
rune_pnp_max_angle_rate: 0.0
rune_pnp_angle_gate_max_gap: 0.25
rune_fit_use_raw_angle: false
rune_angle_filter: simple  # simple or legacy_ekf
```

`rune_pnp_max_reprojection_error` is the RMS reprojection gate for the four PnP model points, in pixels. `rune_pnp_max_distance_jump` only rejects sudden jumps in solved PnP distance between valid observations; it does not gate blade angle switching. Keep `rune_pnp_max_angle_rate: 0.0` unless you explicitly want an angular-rate gate.

Use `rune_angle_filter: simple` for the current recommended path. Set `rune_angle_filter: legacy_ekf` only when doing A/B tests against the old three-state angle EKF.

## 7. Verification Commands

Basic auto-buff unit test:

```bash
./build-orin/auto_buff_test --config-path=configs/standard3.yaml
```

Offline rune video test:

```bash
./build-orin/rune_video_test \
  --video /path/to/rune_video.mkv \
  --config configs/standard3.yaml \
  --out-dir outputs/rune_orin_test \
  --pose-mode identity \
  --mode big \
  --save-every 60
```

Armor auto-aim demo benchmark:

```bash
./build-orin/auto_aim_test \
  --config-path=configs/demo.yaml \
  --benchmark \
  --start-index=0 \
  --end-index=30 \
  assets/demo/demo
```

Runtime with camera and serial:

```bash
./build-orin/standard configs/standard3.yaml
```

Auto-buff runtime:

```bash
./build-orin/auto_buff_debug configs/standard3.yaml
```

Combined standard runtime with auto-buff support:

```bash
./build-orin/mt_standard configs/standard3.yaml
```

Serial yaw/pitch convention is shared by armor auto-aim and rune. Both modules return `io::Command` with radian yaw/pitch in the upper-level FLU convention: X forward, Y left, Z up; yaw rotates around +Z, and pitch keeps the same sign as armor auto-aim. The `neo` protocol transmits those values directly. The `legacy` protocol converts between FLU and the old MCU Y-up convention inside `io/gimbal/protocols/legacy.hpp`, so the predictor code should not add protocol-specific sign flips.

## 8. Common Problems

TensorRT headers not found:

```bash
sudo apt install -y libnvinfer-dev libnvonnxparsers-dev libnvinfer-plugin-dev
```

CMake cannot find Ceres:

```bash
sudo apt install -y libceres-dev
```

or pass:

```bash
-DCMAKE_PREFIX_PATH="$CERES_ROOT" -DCeres_DIR="$CERES_ROOT/lib/cmake/Ceres"
```

Engine loads on x86 but fails on Orin:

- Rebuild the engine on Orin.
- Engine files are not portable across GPU architecture, TensorRT version, or precision settings.

HikRobot camera build failure:

- Set `MVCAM_SDK_PATH` before configuring.
- Delete and recreate the build directory after changing the SDK path.

Runtime cannot open serial:

```bash
groups
ls -l /dev/ttyACM* /dev/ttyUSB*
```

Ensure the user is in `dialout`.

## 9. What the New Code Adds

New auto-buff code requires:

- TensorRT/CUDA runtime and development files.
- Ceres.
- Existing project dependencies: OpenCV, fmt, Eigen3, spdlog, yaml-cpp, nlohmann-json.

The armor auto-aim module does not link Ceres. It still links `trt_infer` for YOLO inference, as before. The auto-buff module links Ceres through `tasks/auto_buff/CMakeLists.txt`.
