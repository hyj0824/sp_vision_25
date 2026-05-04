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

class TrtEngine
{
public:
  explicit TrtEngine(const std::string & engine_path);

  ~TrtEngine();

  bool infer(
    const void * input_data,
    std::size_t input_bytes,
    void * output_data,
    std::size_t output_bytes);

  bool input_is_fp16() const;

  std::size_t input_elements() const { return input_elements_; }

  std::size_t output_elements() const { return output_elements_; }

  const std::vector<int64_t> & input_shape() const { return input_shape_; }

  const std::vector<int64_t> & output_shape() const { return output_shape_; }

private:
  bool load_engine(const std::string & engine_path);

  void init_io();

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
