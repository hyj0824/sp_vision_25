#include "tools/infer/trt_engine.hpp"

#if __has_include(<NvInfer.h>) && __has_include(<NvOnnxParser.h>) && __has_include(<cuda_runtime_api.h>)
#define TOOLS_TRT_AVAILABLE 1
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#else
#define TOOLS_TRT_AVAILABLE 0
#endif

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tools
{
namespace infer
{

std::vector<char> TrtEngine::read_binary(const std::string & path)
{
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open binary file: " + path);
  }

  const auto size = static_cast<std::size_t>(file.tellg());
  std::vector<char> data(size);
  file.seekg(0, std::ios::beg);
  file.read(data.data(), static_cast<std::streamsize>(size));
  if (!file) {
    throw std::runtime_error("Failed to read binary file: " + path);
  }
  return data;
}

void TrtEngine::write_binary(const std::string & path, const void * data, std::size_t size)
{
  std::ofstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to write binary file: " + path);
  }

  file.write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
  if (!file) {
    throw std::runtime_error("Failed to write binary file: " + path);
  }
}

#if TOOLS_TRT_AVAILABLE

namespace
{

using HalfType = __half;

template<typename T>
struct TrtDeleter
{
  void operator()(T * ptr) const
  {
    delete ptr;
  }
};

class TrtLogger : public nvinfer1::ILogger
{
public:
  void log(Severity severity, const char * msg) noexcept override
  {
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

const char * data_type_name(nvinfer1::DataType dtype)
{
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

std::size_t data_type_size(nvinfer1::DataType dtype)
{
  switch (dtype) {
    case nvinfer1::DataType::kFLOAT:
      return sizeof(float);
    case nvinfer1::DataType::kHALF:
      return sizeof(HalfType);
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

std::string shape_to_string(const std::vector<int64_t> & shape)
{
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

std::uint8_t float_to_u8(float v)
{
  // The current preprocess outputs [0,1]. Some graphs may expect [0,255] uint8.
  const float scaled = (v <= 1.0f) ? (v * 255.0f) : v;
  const float clamped = std::max(0.0f, std::min(255.0f, scaled));
  return static_cast<std::uint8_t>(clamped + 0.5f);
}

bool has_dynamic_dim(const nvinfer1::Dims & dims)
{
  for (int i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] < 0) {
      return true;
    }
  }
  return false;
}

nvinfer1::Dims vector_to_dims(const std::vector<int64_t> & shape)
{
  nvinfer1::Dims dims{};
  dims.nbDims = static_cast<int>(shape.size());
  for (int i = 0; i < dims.nbDims; ++i) {
    dims.d[i] = static_cast<int>(shape[static_cast<std::size_t>(i)]);
  }
  return dims;
}

std::vector<int64_t> dims_to_vector(const nvinfer1::Dims & dims)
{
  std::vector<int64_t> result;
  result.reserve(dims.nbDims);
  for (int i = 0; i < dims.nbDims; ++i) {
    result.push_back(static_cast<int64_t>(dims.d[i]));
  }
  return result;
}

std::size_t element_count(const nvinfer1::Dims & dims)
{
  if (dims.nbDims <= 0) {
    return 0;
  }

  std::size_t count = 1;
  for (int i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] <= 0) {
      throw std::runtime_error("TensorRT tensor shape contains non-positive dim.");
    }
    count *= static_cast<std::size_t>(dims.d[i]);
  }
  return count;
}

}  // namespace

class TrtEngine::Impl
{
public:
  using RuntimePtr = std::unique_ptr<nvinfer1::IRuntime, TrtDeleter<nvinfer1::IRuntime>>;
  using EnginePtr = std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter<nvinfer1::ICudaEngine>>;
  using ContextPtr =
    std::unique_ptr<nvinfer1::IExecutionContext, TrtDeleter<nvinfer1::IExecutionContext>>;

  RuntimePtr runtime;
  EnginePtr engine;
  ContextPtr context;

  std::string input_name;
  std::string output_name;
  nvinfer1::DataType input_dtype = nvinfer1::DataType::kFLOAT;
  nvinfer1::DataType output_dtype = nvinfer1::DataType::kFLOAT;

  std::vector<HalfType> input_half_host;
  std::vector<std::uint8_t> input_u8_host;

  std::vector<HalfType> output_half_host;
  std::vector<std::uint8_t> output_u8_host;

  void * device_input = nullptr;
  void * device_output = nullptr;
  cudaStream_t stream = nullptr;

  ~Impl()
  {
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

TrtEngine::TrtEngine(
  const std::string & onnx_path,
  const std::string & engine_path,
  const std::vector<int64_t> & input_shape,
  const TrtOptions & options)
: impl_(std::make_unique<Impl>())
{
  if (input_shape.empty()) {
    throw std::runtime_error("TensorRT input shape must not be empty.");
  }

  if (engine_path.empty()) {
    throw std::runtime_error("TensorRT engine path must not be empty.");
  }

  if (!build_or_load_engine(onnx_path, engine_path, input_shape, options)) {
    throw std::runtime_error("Failed to initialize TensorRT engine.");
  }

  init_io(input_shape);

  if (cudaStreamCreate(&impl_->stream) != cudaSuccess) {
    throw std::runtime_error("Failed to create CUDA stream.");
  }

  const auto input_bytes = input_elements_ * data_type_size(impl_->input_dtype);
  const auto output_bytes = output_elements_ * data_type_size(impl_->output_dtype);

  if (cudaMalloc(&impl_->device_input, input_bytes) != cudaSuccess) {
    throw std::runtime_error("Failed to allocate TensorRT input buffer.");
  }

  if (cudaMalloc(&impl_->device_output, output_bytes) != cudaSuccess) {
    throw std::runtime_error("Failed to allocate TensorRT output buffer.");
  }

  spdlog::info(
    "[TensorRT] ready. input={} dtype={} shape={}, output={} dtype={} shape={}",
    input_elements_, data_type_name(impl_->input_dtype), shape_to_string(input_shape_),
    output_elements_, data_type_name(impl_->output_dtype), shape_to_string(output_shape_));
}

TrtEngine::~TrtEngine() = default;

bool TrtEngine::infer(
  const float * input_data,
  std::size_t input_elements,
  std::vector<float> & output)
{
  if (input_data == nullptr) {
    spdlog::error("[TensorRT] null input pointer.");
    return false;
  }

  if (input_elements != input_elements_) {
    spdlog::error(
      "[TensorRT] input element mismatch, expected {}, got {}",
      input_elements_, input_elements);
    return false;
  }

  const auto input_bytes = input_elements_ * data_type_size(impl_->input_dtype);
  const auto output_bytes = output_elements_ * data_type_size(impl_->output_dtype);

  const void * host_input_ptr = nullptr;
  if (impl_->input_dtype == nvinfer1::DataType::kFLOAT) {
    host_input_ptr = input_data;
  } else if (impl_->input_dtype == nvinfer1::DataType::kHALF) {
    impl_->input_half_host.resize(input_elements_);
    for (std::size_t i = 0; i < input_elements_; ++i) {
      impl_->input_half_host[i] = __float2half_rn(input_data[i]);
    }
    host_input_ptr = impl_->input_half_host.data();
#if defined(NV_TENSORRT_MAJOR) && NV_TENSORRT_MAJOR >= 10
  } else if (impl_->input_dtype == nvinfer1::DataType::kUINT8) {
    impl_->input_u8_host.resize(input_elements_);
    for (std::size_t i = 0; i < input_elements_; ++i) {
      impl_->input_u8_host[i] = float_to_u8(input_data[i]);
    }
    host_input_ptr = impl_->input_u8_host.data();
#endif
  } else {
    spdlog::error(
      "[TensorRT] unsupported input dtype: {}",
      data_type_name(impl_->input_dtype));
    return false;
  }

  if (
    cudaMemcpyAsync(
      impl_->device_input,
      host_input_ptr,
      input_bytes,
      cudaMemcpyHostToDevice,
      impl_->stream) != cudaSuccess)
  {
    spdlog::error("[TensorRT] cudaMemcpyAsync H2D failed.");
    return false;
  }

  if (!impl_->context->setTensorAddress(impl_->input_name.c_str(), impl_->device_input)) {
    spdlog::error("[TensorRT] set input tensor address failed.");
    return false;
  }

  if (!impl_->context->setTensorAddress(impl_->output_name.c_str(), impl_->device_output)) {
    spdlog::error("[TensorRT] set output tensor address failed.");
    return false;
  }

  if (!impl_->context->enqueueV3(impl_->stream)) {
    spdlog::error("[TensorRT] enqueueV3 failed.");
    return false;
  }

  output.resize(output_elements_);

  if (impl_->output_dtype == nvinfer1::DataType::kFLOAT) {
    if (
      cudaMemcpyAsync(
        output.data(),
        impl_->device_output,
        output_bytes,
        cudaMemcpyDeviceToHost,
        impl_->stream) != cudaSuccess)
    {
      spdlog::error("[TensorRT] cudaMemcpyAsync D2H failed.");
      return false;
    }
  } else if (impl_->output_dtype == nvinfer1::DataType::kHALF) {
    impl_->output_half_host.resize(output_elements_);
    if (
      cudaMemcpyAsync(
        impl_->output_half_host.data(),
        impl_->device_output,
        output_bytes,
        cudaMemcpyDeviceToHost,
        impl_->stream) != cudaSuccess)
    {
      spdlog::error("[TensorRT] cudaMemcpyAsync D2H failed for fp16 output.");
      return false;
    }
#if defined(NV_TENSORRT_MAJOR) && NV_TENSORRT_MAJOR >= 10
  } else if (impl_->output_dtype == nvinfer1::DataType::kUINT8) {
    impl_->output_u8_host.resize(output_elements_);
    if (
      cudaMemcpyAsync(
        impl_->output_u8_host.data(),
        impl_->device_output,
        output_bytes,
        cudaMemcpyDeviceToHost,
        impl_->stream) != cudaSuccess)
    {
      spdlog::error("[TensorRT] cudaMemcpyAsync D2H failed for uint8 output.");
      return false;
    }
#endif
  } else {
    spdlog::error(
      "[TensorRT] unsupported output dtype: {}",
      data_type_name(impl_->output_dtype));
    return false;
  }

  if (cudaStreamSynchronize(impl_->stream) != cudaSuccess) {
    spdlog::error("[TensorRT] stream synchronize failed.");
    return false;
  }

  if (impl_->output_dtype == nvinfer1::DataType::kHALF) {
    for (std::size_t i = 0; i < output_elements_; ++i) {
      output[i] = __half2float(impl_->output_half_host[i]);
    }
#if defined(NV_TENSORRT_MAJOR) && NV_TENSORRT_MAJOR >= 10
  } else if (impl_->output_dtype == nvinfer1::DataType::kUINT8) {
    for (std::size_t i = 0; i < output_elements_; ++i) {
      output[i] = static_cast<float>(impl_->output_u8_host[i]);
    }
#endif
  }

  return true;
}

bool TrtEngine::build_or_load_engine(
  const std::string & onnx_path,
  const std::string & engine_path,
  const std::vector<int64_t> & requested_input_shape,
  const TrtOptions & options)
{
  if (!options.force_rebuild && std::filesystem::exists(engine_path)) {
    if (load_engine(engine_path)) {
      spdlog::info("[TensorRT] loaded engine: {}", engine_path);
      return true;
    }
    spdlog::warn("[TensorRT] failed to load existing engine, rebuilding: {}", engine_path);
  }

  if (onnx_path.empty()) {
    spdlog::error(
      "[TensorRT] no ONNX path available and engine file is missing/invalid: {}", engine_path);
    return false;
  }

  if (!build_engine(onnx_path, engine_path, requested_input_shape, options)) {
    return false;
  }

  return load_engine(engine_path);
}

bool TrtEngine::load_engine(const std::string & engine_path)
{
  std::vector<char> bytes;
  try {
    bytes = read_binary(engine_path);
  } catch (const std::exception & e) {
    spdlog::error("[TensorRT] failed to read engine {}: {}", engine_path, e.what());
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

  impl_->engine.reset(impl_->runtime->deserializeCudaEngine(bytes.data(), bytes.size()));
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

bool TrtEngine::build_engine(
  const std::string & onnx_path,
  const std::string & engine_path,
  const std::vector<int64_t> & requested_input_shape,
  const TrtOptions & options)
{
  if (!std::filesystem::exists(onnx_path)) {
    spdlog::error("[TensorRT] ONNX file not found: {}", onnx_path);
    return false;
  }

  auto builder = std::unique_ptr<nvinfer1::IBuilder, TrtDeleter<nvinfer1::IBuilder>>(
    nvinfer1::createInferBuilder(g_trt_logger));
  if (!builder) {
    spdlog::error("[TensorRT] createInferBuilder failed.");
    return false;
  }

  constexpr auto network_flags = 0U;

  auto network =
    std::unique_ptr<nvinfer1::INetworkDefinition, TrtDeleter<nvinfer1::INetworkDefinition>>(
      builder->createNetworkV2(network_flags));
  if (!network) {
    spdlog::error("[TensorRT] createNetworkV2 failed.");
    return false;
  }

  auto parser = std::unique_ptr<nvonnxparser::IParser, TrtDeleter<nvonnxparser::IParser>>(
    nvonnxparser::createParser(*network, g_trt_logger));
  if (!parser) {
    spdlog::error("[TensorRT] create ONNX parser failed.");
    return false;
  }

  if (!parser->parseFromFile(
      onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
  {
    spdlog::error("[TensorRT] parse ONNX failed: {}", onnx_path);
    for (int i = 0; i < parser->getNbErrors(); ++i) {
      spdlog::error("[TensorRT][ONNX] {}", parser->getError(i)->desc());
    }
    return false;
  }

  auto config = std::unique_ptr<nvinfer1::IBuilderConfig, TrtDeleter<nvinfer1::IBuilderConfig>>(
    builder->createBuilderConfig());
  if (!config) {
    spdlog::error("[TensorRT] createBuilderConfig failed.");
    return false;
  }

  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, options.workspace_size_bytes);

  if (options.enable_fp16 && builder->platformHasFastFp16()) {
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
  }

  auto * input_tensor = network->getInput(0);
  if (input_tensor == nullptr) {
    spdlog::error("[TensorRT] network has no input tensor.");
    return false;
  }

  auto input_dims = input_tensor->getDimensions();
  if (has_dynamic_dim(input_dims)) {
    auto * profile = builder->createOptimizationProfile();
    if (profile == nullptr) {
      spdlog::error("[TensorRT] createOptimizationProfile failed.");
      return false;
    }

    if (static_cast<int>(requested_input_shape.size()) != input_dims.nbDims) {
      spdlog::error(
        "[TensorRT] requested input shape rank {} does not match model input rank {}.",
        requested_input_shape.size(), input_dims.nbDims);
      return false;
    }

    auto fixed_dims = input_dims;
    for (int i = 0; i < fixed_dims.nbDims; ++i) {
      if (fixed_dims.d[i] < 0) {
        fixed_dims.d[i] = static_cast<int>(requested_input_shape[static_cast<std::size_t>(i)]);
      }
    }

    if (!profile->setDimensions(
        input_tensor->getName(), nvinfer1::OptProfileSelector::kMIN, fixed_dims))
    {
      spdlog::error("[TensorRT] set MIN profile dims failed.");
      return false;
    }

    if (!profile->setDimensions(
        input_tensor->getName(), nvinfer1::OptProfileSelector::kOPT, fixed_dims))
    {
      spdlog::error("[TensorRT] set OPT profile dims failed.");
      return false;
    }

    if (!profile->setDimensions(
        input_tensor->getName(), nvinfer1::OptProfileSelector::kMAX, fixed_dims))
    {
      spdlog::error("[TensorRT] set MAX profile dims failed.");
      return false;
    }

    const auto profile_index = config->addOptimizationProfile(profile);
    if (profile_index < 0) {
      spdlog::error("[TensorRT] add optimization profile failed.");
      return false;
    }
  }

  auto serialized = std::unique_ptr<nvinfer1::IHostMemory, TrtDeleter<nvinfer1::IHostMemory>>(
    builder->buildSerializedNetwork(*network, *config));

  if (!serialized) {
    spdlog::error("[TensorRT] buildSerializedNetwork failed.");
    return false;
  }

  const auto parent = std::filesystem::path(engine_path).parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent);
  }

  write_binary(engine_path, serialized->data(), serialized->size());
  spdlog::info("[TensorRT] engine generated: {}", engine_path);

  return true;
}

void TrtEngine::init_io(const std::vector<int64_t> & requested_input_shape)
{
  if (!impl_->engine || !impl_->context) {
    throw std::runtime_error("TensorRT engine/context not initialized.");
  }

  const auto io_count = impl_->engine->getNbIOTensors();
  for (int i = 0; i < io_count; ++i) {
    const char * name = impl_->engine->getIOTensorName(i);
    const auto mode = impl_->engine->getTensorIOMode(name);
    if (mode == nvinfer1::TensorIOMode::kINPUT && impl_->input_name.empty()) {
      impl_->input_name = name;
    }
    if (mode == nvinfer1::TensorIOMode::kOUTPUT && impl_->output_name.empty()) {
      impl_->output_name = name;
    }
  }

  if (impl_->input_name.empty() || impl_->output_name.empty()) {
    throw std::runtime_error("TensorRT engine must contain at least one input and one output.");
  }

  auto input_dims = impl_->engine->getTensorShape(impl_->input_name.c_str());
  if (has_dynamic_dim(input_dims)) {
    if (static_cast<int>(requested_input_shape.size()) != input_dims.nbDims) {
      throw std::runtime_error("Requested input shape rank does not match TensorRT input rank.");
    }

    const auto set_dims = vector_to_dims(requested_input_shape);
    if (!impl_->context->setInputShape(impl_->input_name.c_str(), set_dims)) {
      throw std::runtime_error("TensorRT setInputShape failed.");
    }

    input_dims = impl_->context->getTensorShape(impl_->input_name.c_str());
  }

  auto output_dims = impl_->context->getTensorShape(impl_->output_name.c_str());

  if (has_dynamic_dim(input_dims) || has_dynamic_dim(output_dims)) {
    throw std::runtime_error("TensorRT unresolved dynamic shape after setInputShape.");
  }

  impl_->input_dtype = impl_->engine->getTensorDataType(impl_->input_name.c_str());
  impl_->output_dtype = impl_->engine->getTensorDataType(impl_->output_name.c_str());

  input_shape_ = dims_to_vector(input_dims);
  output_shape_ = dims_to_vector(output_dims);
  input_elements_ = element_count(input_dims);
  output_elements_ = element_count(output_dims);
}

#else

class TrtEngine::Impl
{
};

TrtEngine::TrtEngine(
  const std::string &,
  const std::string &,
  const std::vector<int64_t> &,
  const TrtOptions &)
: impl_(std::make_unique<Impl>())
{
  throw std::runtime_error(
    "TensorRT/CUDA headers are unavailable in current environment. "
    "Please build on Jetson with TensorRT dev packages installed.");
}

TrtEngine::~TrtEngine() = default;

bool TrtEngine::infer(const float *, std::size_t, std::vector<float> &)
{
  return false;
}

bool TrtEngine::build_or_load_engine(
  const std::string &,
  const std::string &,
  const std::vector<int64_t> &,
  const TrtOptions &)
{
  return false;
}

bool TrtEngine::load_engine(const std::string &)
{
  return false;
}

bool TrtEngine::build_engine(
  const std::string &,
  const std::string &,
  const std::vector<int64_t> &,
  const TrtOptions &)
{
  return false;
}

void TrtEngine::init_io(const std::vector<int64_t> &)
{
}

#endif

}  // namespace infer
}  // namespace tools
