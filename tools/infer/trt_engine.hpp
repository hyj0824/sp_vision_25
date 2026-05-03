#ifndef TOOLS__INFER__TRT_ENGINE_HPP
#define TOOLS__INFER__TRT_ENGINE_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tools
{
namespace infer
{

struct TrtOptions
{
  bool enable_fp16 = true;
  bool force_rebuild = false;
  std::size_t workspace_size_bytes = 1ULL << 30;
};

class TrtEngine
{
public:
  TrtEngine(
    const std::string & onnx_path,
    const std::string & engine_path,
    const std::vector<int64_t> & input_shape,
    const TrtOptions & options = {});

  ~TrtEngine();

  bool infer(const float * input_data, std::size_t input_elements, std::vector<float> & output);

  bool infer_fp16(const void * input_data, std::size_t input_elements, std::vector<float> & output);

  bool input_is_fp16() const;

  std::size_t input_elements() const { return input_elements_; }

  std::size_t output_elements() const { return output_elements_; }

  const std::vector<int64_t> & input_shape() const { return input_shape_; }

  const std::vector<int64_t> & output_shape() const { return output_shape_; }

private:
  bool build_or_load_engine(
    const std::string & onnx_path,
    const std::string & engine_path,
    const std::vector<int64_t> & requested_input_shape,
    const TrtOptions & options);

  bool load_engine(const std::string & engine_path);

  bool build_engine(
    const std::string & onnx_path,
    const std::string & engine_path,
    const std::vector<int64_t> & requested_input_shape,
    const TrtOptions & options);

  void init_io(const std::vector<int64_t> & requested_input_shape);

  bool infer_prepared(
    const void * host_input,
    std::size_t input_elements,
    std::vector<float> & output);

  static std::vector<char> read_binary(const std::string & path);

  static void write_binary(const std::string & path, const void * data, std::size_t size);

  class Impl;
  std::unique_ptr<Impl> impl_;

  std::vector<int64_t> input_shape_;
  std::vector<int64_t> output_shape_;

  std::size_t input_elements_ = 0;
  std::size_t output_elements_ = 0;
};

}  // namespace infer
}  // namespace tools

#endif  // TOOLS__INFER__TRT_ENGINE_HPP
