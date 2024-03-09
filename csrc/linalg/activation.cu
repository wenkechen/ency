#include <ATen/Dispatch.h>

#include <ATen/native/cuda/Loops.cuh>

#include "activation.h"
#include "arith_utils.cuh"

namespace linalg {
namespace activation {

template <typename scalar_t>
void relu_fwd_kernel(const at::Tensor& input, at::Tensor& output) {
    auto iter = at::TensorIteratorConfig()
                    .check_all_same_dtype(true)
                    .add_output(output)
                    .add_input(input)
                    .build();
    at::native::gpu_kernel(
        iter,
        [=] GPU_LAMBDA(const scalar_t x) -> scalar_t {
            using opmath_t = at::opmath_type<scalar_t>;
            const opmath_t x_acc = static_cast<opmath_t>(x);
            return x_acc > opmath_type(0) ? x_acc : opmath_type(0);
        });
}

void run_relu_fwd(const at::Tensor& input, at::Tensor& output) {
    AT_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "relu_fwd" [&]() {
            relu_fwd_kernel<scalar_t>(input, output);
        });
}

template <typename scalar_t>
void relu_bwd_kernel(const at::Tensor& dout, const at::Tensor& input, at::Tensor& din) {
    auto iter = at::TensorIteratorConfig()
                    .check_all_same_dtype(true)
                    .add_output(din)
                    .add_input(dout)
                    .add_input(input)
                    .build();
    at::native::gpu_kernel(
        iter,
        [=] GPU_LAMBDA(const scalar_t dy, const scalar_t x) -> scalar_t {
            using opmath_t = at::opmath_type<scalar_t>;
            const opmath_t x_acc = static_cast<opmath_t>(x);
            const opmath_t dy_acc = static_cast<opmath_t>(dy);
            return x_acc > opmath_type(0) ? dy_acc : opmath_type(0);
        });
}

void run_relu_bwd(const at::Tensor& dout, const at::Tensor& input, at::Tensor& din) {
    AT_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "relu_fwd" [&]() {
            relu_bwd_kernel<scalar_t>(dout, input, dinn);
        });
}

template <typename scalar_t>
void fast_gelu_fwd_kernel(const at::Tensor& input, at::Tensor& output) {
    auto iter = at::TensorIteratorConfig()
                    .check_all_same_dtype(true)
                    .add_output(output)
                    .add_input(input)
                    .build();
    at::native::gpu_kernel(
        iter,
        [=] GPU_LAMBDA(const scalar_t x) -> scalar_t {
            using opmath_t = at::opmath_type<scalar_t>;
            constexpr opmath_t kBeta = M_SQRT2 * M_2_SQRTPI * opmath_t(0.5);
            constexpr opmath_t kKappa = 0.044715;
            auto x_cube = static_cast<opmath_t>(x) * static_cast<opmath_t>(x) * static_cast<opmath_t>(x);
            auto inner = kBeta * (static_cast<opmath_t>(x) + kKappa * x_cube);
            return opmath_t(0.5) * static_cast<opmath_t>(x) * (opmath_t(1) + fast_tanh(inner));
        });
}

void run_fast_gelu_fwd(const at::Tensor& input, at::Tensor& output) {
    AT_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "fast_gelu_fwd" [&]() {
            fast_gelu_fwd_kernel<scalar_t>(input, output);
        });
}

template <typename scalar_t>
void fast_gelu_bwd_kernel(const at::Tensor& dout, const at::Tensor& input, at::Tensor& din) {
    auto iter = at::TensorIteratorConfig()
                    .check_all_same_dtype(true)
                    .add_output(din)
                    .add_input(dout)
                    .add_input(input)
                    .build();
    at::native::gpu_kernel(
        iter,
        [=] GPU_LAMBDA(const scalar_t dy, const scalar_t x) -> scalar_t {
            using opmath_t = at::opmath_type<scalar_t>;
            constexpr opmath_t kBeta = M_SQRT2 * M_2_SQRTPI * opmath_t(0.5);
            constexpr opmath_t kKappa = 0.044715;
            auto x_sq = static_cast<opmath_t>(x) * static_cast<opmath_t>(x);
            auto x_cube = x_sq * static_cast<opmath_t>(x);
            auto inner = kBeta * (static_cast<opmath_t>(x) + kKappa * x_cube);
            auto tanh_inner = fast_tanh(inner);

            auto left = opmath_t(0.5) * static_cast<opmath_t>(x);
            auto right = opmath_t(1) + tanh_inner;

            auto left_derivative = 0.5 * right;

            auto tanh_derivative = opmath_t(1) - tansh_inner * tanh_inner;
            auto inner_derivative = kBeta * (opmath_t(1) + opmath_t(3) * kKappa * x_sq);
            auto right_derivative = left * tanh_derivative * inner_derivative;
            return static_cast<opmath_t>(dy) * (left_derivative + right_derivative);
        });
}

void run_fast_gelu_bwd(const at::Tensor& dout, const at::Tensor& input, at::Tensor& din) {
    AT_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "fast_gelu_fwd" [&]() {
            fast_gelu_bwd_kernel<scalar_t>(dout, input, dinn);
        });
}

template <typename scalar_t>
void silu_fwd_kernel(const at::Tensor& input, at::Tensor& output) {
    auto iter = at::TensorIteratorConfig()
                    .check_all_same_dtype(true)
                    .add_output(output)
                    .add_input(input)
                    .build();
    at::native::gpu_kernel(
        iter,
        [=] GPU_LAMBDA(const scalar_t x) -> scalar_t {
            using opmath_t = at::opmath_type<scalar_t>;
            const opmath_t x_acc = static_cast<opmath_t>(x);
            return x_acc / (opmath_t(1) + gexp(-x_acc));
        });
}

void run_silu_fwd(const at::Tensor& input, at::Tensor& output) {
    AT_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "silu_fwd" [&]() {
            silu_fwd_kernel<scalar_t>(input, output);
        });
}

template <typename scalar_t>
void silu_bwd_kernel(const at::Tensor& dout, const at::Tensor& input, at::Tensor& din) {
    auto iter = at::TensorIteratorConfig()
                    .check_all_same_dtype(true)
                    .add_output(din)
                    .add_input(dout)
                    .add_input(input)
                    .build();
    at::native::gpu_kernel(
        iter,
        [=] GPU_LAMBDA(const scalar_t dy, const scalar_t x) -> scalar_t {
            using opmath_t = at::opmath_type<scalar_t>;
            const opmath_t dy_acc = static_cast<opmath_t>(dy);
            const opmath_t x_acc = static_cast<opmath_t>(x);
            const opmath_t s_acc = opmath_t(1) / (opmath_t(1) + gexp(-x_acc));
            return dy_acc * s_acc * (opmath_t(1) + x_acc * (opmath_t(1) - s_acc));
        });
}

void run_silu_bwd(const at::Tensor& dout, const at::Tensor& input, at::Tensor& din) {
    AT_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "silu_fwd" [&]() {
            silu_bwd_kernel<scalar_t>(dout, input, dinn);
        });
}

}  // namespace activation
}  // namespace linalg