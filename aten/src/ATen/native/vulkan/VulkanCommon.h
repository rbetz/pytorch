#pragma once
#include <ATen/Tensor.h>
#include <array>

#define COUT_FLF std::cout << __FILE__ << __LINE__ << __FUNCTION__
#define COUT_FLFE std::cout << __FILE__ << __LINE__ << __FUNCTION__ << std::endl
#define COUT_FLF0 \
  std::cout << __FILE__ << __LINE__ << __FUNCTION__ << " 000" << std::endl

namespace at {
namespace native {
namespace vulkan {

struct ContextConv2D final {
  at::Tensor weight_prepacked_vulkan_;
  c10::optional<at::Tensor> bias_vulkan_;
  std::array<int64_t, 4> weight_size_;
  std::array<int64_t, 2> padding_;
  std::array<int64_t, 2> stride_;
  std::array<int64_t, 2> dilation_;
  int64_t groups_;

  ContextConv2D() = delete;

  ContextConv2D(
      at::Tensor&& weight_prepacked_vulkan,
      c10::optional<at::Tensor>&& bias_vulkan,
      std::array<int64_t, 4> weight_size,
      std::array<int64_t, 2> padding,
      std::array<int64_t, 2> stride,
      std::array<int64_t, 2> dilation,
      int64_t groups)
      : weight_prepacked_vulkan_(std::move(weight_prepacked_vulkan)),
        bias_vulkan_(std::move(bias_vulkan)),
        weight_size_(weight_size),
        padding_(padding),
        stride_(stride),
        dilation_(dilation),
        groups_(groups) {
    COUT_FLFE;
  }

  ContextConv2D(ContextConv2D&&) = default;
  ContextConv2D& operator=(ContextConv2D&&) = default;

  ~ContextConv2D() {
    COUT_FLFE;
  }

  static constexpr float kMin = -std::numeric_limits<float>::infinity();
  static constexpr float kMax = std::numeric_limits<float>::infinity();
};

} // namespace vulkan
} // namespace native
} // namespace at
