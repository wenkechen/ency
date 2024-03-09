#pragma once
#include <ATen/Tensor.h>

#include <cstdint>

namespace linalg {
namespace activation {

enum class ActivationType : uint32_t {
    No,
    Relu,
    FastGelu,
    Silu
};

void run_relu_fwd(const at::Tensor& input, at::Tensor& output);
void run_relu_bwd(const at::Tensor& dout, const at::Tensor& input, at::Tensor& din);
void run_fast_gelu_fwd(const at::Tensor& input, at::Tensor& output);
void run_fast_gelu_bwd(const at::Tensor& dout, const at::Tensor& input, at::Tensor& din);
void run_silu_fwd(const at::Tensor& input, at::Tensor& output);
void run_silu_bwd(const at::Tensor& dout, const at::Tensor& input, at::Tensor& din);
}  // namespace activation
}  // namespace linalg
