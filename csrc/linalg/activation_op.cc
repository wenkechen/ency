#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <torch/python.h>

#include "activation.h"

namespace linalg {
namespace activation_op {
at::Tensor activation_fwd(const at::Tensor& input, const int64_t activation_type) {
    at::cuda::CUDAGuard device_guard{(char)input.get_device()};
    at::Tensor output;
    switch (activation_type) {
    case 0:  // No
        output = input;
        break;
    case 1:  // Relu
        // TODO: check at::empty_like's signature
        output = at::empty_like(input, input.suggest_memory_format());
        linalg::activation::run_relu_fwd(input, output);
        break;
    case 2:  // FastGelu
        output = at::empty_like(input, input.suggest_memory_format());
        linalg::activation::run_fast_gelu_fwd(input, output);
        break;
    case 3:  // Silu
        output = at::empty_like(input, input.suggest_memory_format());
        linalg::activation::run_silu_fwd(input, output);
        break;
    default:
        TORCH_CHECK(false, "Unsupported activation type: ", activation_type);
        break;
    }
    return output;
}

at::Tensor activation_bwd(const at::Tensor& dout, const at::Tensor& input, const int64_t activation_type) {
    at::cuda::CUDAGuard device_guard{(char)dout.get_device()};
    at::Tensor din;
    switch (activation_type) {
    case 0:  // No
        din = dout;
        break;
    case 1:  // Relu
        // TODO: check at::empty_like's signature
        din = at::empty_like(input, input.suggest_memory_format());
        linalg::activation::run_relu_bwd(dout, input, din);
        break;
    case 2:  // FastGelu
        din = at::empty_like(input, input.suggest_memory_format());
        linalg::activation::run_fast_gelu_bwd(dout, input, din);
        break;
    case 3:  // Silu
        din = at::empty_like(input, input.suggest_memory_format());
        linalg::activation::run_silu_bwd(dout, input, din);
        break;
    default:
        TORCH_CHECK(false, "Unsupported activation type: ", activation_type);
        break;
    }
    return din;
}
}  // namespace activation_op
}  // namespace linalg