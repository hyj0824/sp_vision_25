#include "tools/infer/trt_engine.hpp"
#include <utility>

#if __has_include(<NvInfer.h>) && __has_include(<cuda_runtime_api.h>)
#define TOOLS_TRT_AVAILABLE 1
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#else
#define TOOLS_TRT_AVAILABLE 0
#endif

#include <spdlog/spdlog.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tools {
namespace infer {

#if TOOLS_TRT_AVAILABLE

namespace {

template <typename T> struct TrtDeleter {
  void operator()(T *ptr) const { delete ptr; }
};

class TrtLogger : public nvinfer1::ILogger {
public:
  void log(Severity severity, const char *msg) noexcept override {
    if (severity == Severity::kINTERNAL_ERROR || severity == Severity::kERROR) {
      spdlog::error("[TensorRT] {}", msg);
      return;
    }

    if (severity == Severity::kWARNING) {
      spdlog::warn("[TensorRT] {}", msg);
      return;
    }

    spdlog::debug("[TensorRT] {}", msg);
  }
};

TrtLogger g_trt_logger;

const char *data_type_name(nvinfer1::DataType dtype) {
  switch (dtype) {
  case nvinfer1::DataType::kFLOAT:
    return "float32";
  case nvinfer1::DataType::kHALF:
    return "float16";
  case nvinfer1::DataType::kINT8:
    return "int8";
  case nvinfer1::DataType::kINT32:
    return "int32";
  case nvinfer1::DataType::kBOOL:
    return "bool";
#if defined(NV_TENSORRT_MAJOR) && NV_TENSORRT_MAJOR >= 10
  case nvinfer1::DataType::kUINT8:
    return "uint8";
  case nvinfer1::DataType::kFP8:
    return "fp8";
  case nvinfer1::DataType::kINT64:
    return "int64";
#endif
  default:
    return "unknown";
  }
}

std::size_t data_type_size(nvinfer1::DataType dtype) {
  switch (dtype) {
  case nvinfer1::DataType::kFLOAT:
    return sizeof(float);
  case nvinfer1::DataType::kHALF:
    return sizeof(std::uint16_t);
  case nvinfer1::DataType::kINT8:
    return sizeof(std::int8_t);
  case nvinfer1::DataType::kINT32:
    return sizeof(std::int32_t);
  case nvinfer1::DataType::kBOOL:
    return sizeof(bool);
#if defined(NV_TENSORRT_MAJOR) && NV_TENSORRT_MAJOR >= 10
  case nvinfer1::DataType::kUINT8:
    return sizeof(std::uint8_t);
  case nvinfer1::DataType::kFP8:
    return sizeof(std::uint8_t);
  case nvinfer1::DataType::kINT64:
    return sizeof(std::int64_t);
#endif
  default:
    throw std::runtime_error("Unsupported TensorRT tensor data type.");
  }
}

std::string shape_to_string(const std::vector<int64_t> &shape) {
  std::string result = "[";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    result += std::to_string(shape[i]);
    if (i + 1 < shape.size()) {
      result += ",";
    }
  }
  result += "]";
  return result;
}

bool has_dynamic_dim(const nvinfer1::Dims &dims) {
  for (int i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] < 0) {
      return true;
    }
  }
  return false;
}

std::vector<int64_t> dims_to_vector(const nvinfer1::Dims &dims) {
  std::vector<int64_t> result;
  result.reserve(dims.nbDims);
  for (int i = 0; i < dims.nbDims; ++i) {
    result.push_back(static_cast<int64_t>(dims.d[i]));
  }
  return result;
}

std::size_t element_count(const nvinfer1::Dims &dims) {
  if (dims.nbDims <= 0) {
    return 0;
  }

  std::size_t count = 1;
  for (int i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] <= 0) {
      throw std::runtime_error(
          "TensorRT tensor shape contains non-positive dim.");
    }
    count *= static_cast<std::size_t>(dims.d[i]);
  }
  return count;
}

} // namespace

class TrtEngine::Impl {
public:
  using RuntimePtr =
      std::unique_ptr<nvinfer1::IRuntime, TrtDeleter<nvinfer1::IRuntime>>;
  using EnginePtr =
      std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter<nvinfer1::ICudaEngine>>;
  using ContextPtr = std::unique_ptr<nvinfer1::IExecutionContext,
                                     TrtDeleter<nvinfer1::IExecutionContext>>;

  RuntimePtr runtime;
  EnginePtr engine;
  ContextPtr context;

  std::string input_name;
  std::string output_name;
  nvinfer1::DataType input_dtype = nvinfer1::DataType::kFLOAT;
  nvinfer1::DataType output_dtype = nvinfer1::DataType::kFLOAT;

  void *device_input = nullptr;
  void *device_output = nullptr;
  cudaStream_t stream = nullptr;

  ~Impl() {
    if (device_input != nullptr) {
      cudaFree(device_input);
      device_input = nullptr;
    }

    if (device_output != nullptr) {
      cudaFree(device_output);
      device_output = nullptr;
    }

    if (stream != nullptr) {
      cudaStreamDestroy(stream);
      stream = nullptr;
    }
  }
};

TrtEngine::TrtEngine(const std::string &engine_path)
    : impl_(std::make_unique<Impl>()) {
  if (engine_path.empty()) {
    throw std::runtime_error("TensorRT engine path must not be empty.");
  }

  if (!std::filesystem::exists(engine_path)) {
    spdlog::error("[TensorRT] engine file not found: {}", engine_path);
    throw std::runtime_error("TensorRT engine file not found: " + engine_path);
  }

  if (!load_engine(engine_path)) {
    throw std::runtime_error("Failed to load TensorRT engine: " + engine_path);
  }

  spdlog::info("[TensorRT] loaded engine: {}", engine_path);
  init_io();

  if (cudaStreamCreate(&impl_->stream) != cudaSuccess) {
    throw std::runtime_error("Failed to create CUDA stream.");
  }

  const auto input_bytes = input_elements_ * data_type_size(impl_->input_dtype);
  const auto output_bytes =
      output_elements_ * data_type_size(impl_->output_dtype);

  if (cudaMalloc(&impl_->device_input, input_bytes) != cudaSuccess) {
    throw std::runtime_error("Failed to allocate TensorRT input buffer.");
  }

  if (cudaMalloc(&impl_->device_output, output_bytes) != cudaSuccess) {
    throw std::runtime_error("Failed to allocate TensorRT output buffer.");
  }

  if (!impl_->context->setTensorAddress(impl_->input_name.c_str(),
                                        impl_->device_input)) {
    throw std::runtime_error("Failed to bind TensorRT input buffer.");
  }

  if (!impl_->context->setTensorAddress(impl_->output_name.c_str(),
                                        impl_->device_output)) {
    throw std::runtime_error("Failed to bind TensorRT output buffer.");
  }

  spdlog::info("[TensorRT] ready. input={} dtype={} shape={}, output={} "
               "dtype={} shape={}",
               input_elements_, data_type_name(impl_->input_dtype),
               shape_to_string(input_shape_), output_elements_,
               data_type_name(impl_->output_dtype),
               shape_to_string(output_shape_));
}

TrtEngine::~TrtEngine() = default;

bool TrtEngine::infer(const void *input_data, std::size_t input_bytes,
                      void *output_data, std::size_t output_bytes) {
  if (input_data == nullptr) {
    spdlog::error("[TensorRT] null input pointer.");
    return false;
  }

  if (output_data == nullptr) {
    spdlog::error("[TensorRT] null output pointer.");
    return false;
  }

  const auto expected_input_bytes =
      input_elements_ * data_type_size(impl_->input_dtype);
  if (input_bytes != expected_input_bytes) {
    spdlog::error("[TensorRT] input byte mismatch, expected {}, got {}",
                  expected_input_bytes, input_bytes);
    return false;
  }

  const auto expected_output_bytes =
      output_elements_ * data_type_size(impl_->output_dtype);
  if (output_bytes != expected_output_bytes) {
    spdlog::error("[TensorRT] output byte mismatch, expected {}, got {}",
                  expected_output_bytes, output_bytes);
    return false;
  }

  if (cudaMemcpyAsync(impl_->device_input, input_data, input_bytes,
                      cudaMemcpyHostToDevice, impl_->stream) != cudaSuccess) {
    spdlog::error("[TensorRT] cudaMemcpyAsync H2D failed.");
    return false;
  }

  if (!impl_->context->enqueueV3(impl_->stream)) {
    spdlog::error("[TensorRT] enqueueV3 failed.");
    return false;
  }

  if (cudaMemcpyAsync(output_data, impl_->device_output, output_bytes,
                      cudaMemcpyDeviceToHost, impl_->stream) != cudaSuccess) {
    spdlog::error("[TensorRT] cudaMemcpyAsync D2H failed.");
    return false;
  }

  if (cudaStreamSynchronize(impl_->stream) != cudaSuccess) {
    spdlog::error("[TensorRT] stream synchronize failed.");
    return false;
  }

  return true;
}

bool TrtEngine::input_is_fp16() const {
  return impl_->input_dtype == nvinfer1::DataType::kHALF;
}

bool TrtEngine::load_engine(const std::string &engine_path) {
  std::vector<char> bytes;
  try {
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open binary file: " + engine_path);
    }

    const auto size = static_cast<std::size_t>(file.tellg());
    std::vector<char> data(size);
    file.seekg(0, std::ios::beg);
    file.read(data.data(), static_cast<std::streamsize>(size));
    if (!file) {
      throw std::runtime_error("Failed to read binary file: " + engine_path);
    }
    bytes = std::move(data);
  } catch (const std::exception &e) {
    spdlog::error("[TensorRT] failed to read engine {}: {}", engine_path,
                  e.what());
    return false;
  }

  if (bytes.empty()) {
    spdlog::error("[TensorRT] engine file is empty: {}", engine_path);
    return false;
  }

  impl_->runtime.reset(nvinfer1::createInferRuntime(g_trt_logger));
  if (!impl_->runtime) {
    spdlog::error("[TensorRT] createInferRuntime failed.");
    return false;
  }

  impl_->engine.reset(
      impl_->runtime->deserializeCudaEngine(bytes.data(), bytes.size()));
  if (!impl_->engine) {
    spdlog::error("[TensorRT] deserializeCudaEngine failed: {}", engine_path);
    return false;
  }

  impl_->context.reset(impl_->engine->createExecutionContext());
  if (!impl_->context) {
    spdlog::error("[TensorRT] createExecutionContext failed.");
    return false;
  }

  return true;
}

void TrtEngine::init_io() {
  if (!impl_->engine || !impl_->context) {
    throw std::runtime_error("TensorRT engine/context not initialized.");
  }

  const auto io_count = impl_->engine->getNbIOTensors();
  for (int i = 0; i < io_count; ++i) {
    const char *name = impl_->engine->getIOTensorName(i);
    const auto mode = impl_->engine->getTensorIOMode(name);
    if (mode == nvinfer1::TensorIOMode::kINPUT && impl_->input_name.empty()) {
      impl_->input_name = name;
    }
    if (mode == nvinfer1::TensorIOMode::kOUTPUT && impl_->output_name.empty()) {
      impl_->output_name = name;
    }
  }

  if (impl_->input_name.empty() || impl_->output_name.empty()) {
    throw std::runtime_error(
        "TensorRT engine must contain at least one input and one output.");
  }

  const auto input_dims = impl_->engine->getTensorShape(impl_->input_name.c_str());
  const auto output_dims = impl_->engine->getTensorShape(impl_->output_name.c_str());

  if (has_dynamic_dim(input_dims) || has_dynamic_dim(output_dims)) {
    throw std::runtime_error(
        "TensorRT dynamic shapes are not supported. Please build a fixed-shape engine.");
  }

  impl_->input_dtype =
      impl_->engine->getTensorDataType(impl_->input_name.c_str());
  impl_->output_dtype =
      impl_->engine->getTensorDataType(impl_->output_name.c_str());

  input_shape_ = dims_to_vector(input_dims);
  output_shape_ = dims_to_vector(output_dims);
  input_elements_ = element_count(input_dims);
  output_elements_ = element_count(output_dims);
}

#else

class TrtEngine::Impl {};

TrtEngine::TrtEngine(const std::string &)
    : impl_(std::make_unique<Impl>()) {
  throw std::runtime_error(
      "TensorRT/CUDA headers are unavailable in current environment. "
      "Please build on Jetson with TensorRT dev packages installed.");
}

TrtEngine::~TrtEngine() = default;

bool TrtEngine::infer(const void *, std::size_t, void *, std::size_t) {
  return false;
}

bool TrtEngine::input_is_fp16() const { return false; }

bool TrtEngine::load_engine(const std::string &) { return false; }

void TrtEngine::init_io() {}

#endif

} // namespace infer
} // namespace tools
