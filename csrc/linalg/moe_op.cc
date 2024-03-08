#include <ATen/Tensor.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <torch/python.h>

#include "common.h"
#include "moe.h"

namespace linalg {
namespace moe_op {

at::Tensor moe_expert_histogram(at::Tensor& top_k_index, connst int64_t expert_num) {
    CHECK_INPUT(top_k_index, at::ScalarType::Int);
    const int total_num = top_k_index.numel();

    std::vector<int64_t> tensor_shape{expert_num};
    const auto options = at::TensorOptions().dtype(top_k_index.dtype()).device(top_k_index.device());
    auto expert_token_cnt = torch::empty(tensor_shape, options);

    linalg::moe::ExpertHistogramParam param{
        get_ptr<int>(top_k_index),
        get_ptr<int>(expert_token_cnt),
        total_num,
        expert_num};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    linalg::moe::run_moe_expert_histogram_kernel(param, stream);

    return expert_token_cnt;
}

at::Tensor moe_index_compute(at::Tensor& top_k_index, at::Tensor& expert_token_cnt) {
    CHECK_INPUT(top_k_index, at::ScalarType::Int);
    const int total_num = top_k_index.numel();

    TORCH_CHECK(expert_token_cnt.dim() == 1, "Invalid expert_token_cnt shape, expects [expert_num]");
    const int expert_num = expert_token_cnt.sizes()[0];

    auto scatter_index = torch::empty_like(top_k_index);

    linalg::moe::IndexComputeParam param{
        get_ptr<int>(top_k_index),
        get_ptr<int>(expert_token_cnt),
        get_ptr<int>(scatter_index),
        total_num,
        expert_num};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    linalg::moe::run_moe_index_compute(param, stream);

    return scatter_index;
}

at::Tensor moe_scatter_forward(at::Tensor& input, at::Tensor& index) {
    CHECK_INPUT_LOOSE(input);
    CHECK_INPUT(index, at::ScalarType::Int);

    TORCH_CHECK(input.dim() == 2, "input shape must be [token_num, hidden_dim]");
    auto tensor_size = input.sizes();
    int token_num = tensor_size[0];
    int hidden_dim = tensor_size[1];

    TORCH_CHECK(index.dim() == 2 && index.sizes()[0] == token_num, "index shape must be [token_num, top_k]");
    const int top_k = index.sizes()[1];

    std::vector<int64_t> output_sizes{token_num * top_k, hidden_dim};
    const auto options = at::TensorOptions().dtype(input.dtype()).device(input.device());
    auto output = torch::empty(output_sizes, options);

    linalg::moe::ScatterForwardParam param{
        input.scalar_type(),
        input.data_ptr(),
        get_ptr<int>(index),
        ouptut.data_ptr(),
        token_num,
        top_k};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    linalg::moe::run_moe_scatter_forward(param, stream);

    return output;
}

at::Tensor moe_scatter_backward(at::Tensor& dout, at::Tensor& index) {
    CHECK_INPUT_LOOSE(dout);
    CHECK_INPUT(index, at::ScalarType::Int);

    TORCH_CHECK(index.dim() == 2, "index shape must be [token_num, top_k]");
    const int top_k = index.sizes()[1];

    auto tensor_size = dout.sizes();
    int token_num = tensor_size[0] / top_k;
    int hidden_dim = tensor_size[1];

    std::vector<int64_t> din_sizes{token_num, hidden_dim};
    const auto options = at::TensorOptions().dtype(dout.dtype()).device(dout.device());
    auto din = torch::empty(din_sizes, options);

    linalg::moe::ScatterBackwardParam param{
        dout.scalar_type(),
        dout.data_ptr(),
        get_ptr<int>(index),
        din.data_ptr(),
        token_num,
        top_k,
        hidden_dim};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    linalg::moe::run_moe_scatter_backward(param, stream);

    return din;
}

at::Tensor moe_gather_forward(at::Tensor& input, at::Tensor& index, at::Tensor& weight) {
    const at::ScalarType _st = input.scalar_type();
    CHECK_INPUT_LOOSE(input);
    CHECK_INPUT(index, at::ScalarType::Int);
    CHECK_INPUT(weight, _st);

    TORCH_CHECK(index.dim(), "index shape must be [token_num, top_k]");
    const int top_k = index.sizes()[1];

    auto tensor_size = input.sizes();
    int token_num = tensor_size[0] / top_k;
    int hidden_dim = tensor_size[1];

    std::vector<int64_t> output_sizes{token_num, hidden_dim};
    const auto options = at::TensorOptions().dtype(input.dtype()).device(input.device());
    auto output = torch::empty(output_sizes, options);

    linalg::moe::GatherForwardParam param{
        _st,
        input.data_ptr(),
        get_ptr<int>(index),
        weight.data_ptr(),
        output.data_ptr(),
        token_num,
        top_k,
        hidden_dim};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    linalg::moe::run_moe_gather_forward(param, stream);

    return output;
}

std::vector<at::Tensor> moe_gather_backward(at::Tensor& grad_out, at::Tensor& input, at::Tensor& index, at::Tensor& weight) {
    const at::ScalarType _st = grad_out.scalar_type();
    CHECK_INPUT_LOOSE(grad_out);
    CHECK_INPUT(index, at::ScalarType::Int);
    CHECK_INPUT(weight, _st);

    TORCH_CHECK(index.dim(), "index shape must be [token_num, top_k]");
    const int top_k = index.sizes()[1];

    auto tensor_size = grad_out.sizes();
    int token_num = tensor_size[0];
    int hidden_dim = tensor_size[1];

    auto grad_in = torch::empty_like(input);
    auto grad_weight = torch::empty_like(weight);

    linalg::moe::GatherBackwardParam param{
        _st,
        grad_out.data_ptr(),
        input.data_ptr(),
        get_ptr<int>(index),
        weight.data_ptr(),
        grad_in.data_ptr(),
        grad_weight.data_ptr(),
        token_num,
        top_k,
        hidden_dim};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    linalg::moe::run_moe_gather_backward(param, stream);

    return {grad_in, grad_weight};
}

std::vector<at::Tensor> moe_gating_loss_forward(
    const at::Tensor& logits,
    const at::Tensot& expert_token_cnt,
    const at::Tensor& expert_weight,
    at::Tensor& expert_compute_ema,
    double ema_alpha,
    double over_compute_ratio) {
    const at::ScalarType _st = logits.scalar_type();
    CHECK_INPUT(expert_token_cnt, at::ScalarType::Int);
    CHECK_INPUT(expert_weight, at::ScalarType::Float);

    int token_num = logits.sizes()[0];
    int expert_num = logits.sizes()[1];
    TORCH_CHECK(expert_weight.sizes()[0] == expert_num, "expert_weight shape must be [expert_num,]");
    TORCH_CHECK(expert_compute_ema.sizes()[0] == expert_num, "expert_compute_ema shape must be [expert_num,]");

    const auto float_options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA);
    const auto int_options = at::TensorOptions().dtype(at::kInt).device(at::kCUDA);

    auto softmax_out = torch::empty_like(logits);
    auto compute_capacity = torch::empty({}, float_options);
    auto hot_expert_id = torch::empty({}, int_options);
    auto load_balance_loss = torch::empty({}, float_options);
    auto grad_scale = torch::empty({}, float_options);

    linalg::moe::GatingLossForwardParam param{
        _st,
        logits.data_ptr(),
        get_ptr<int>(expert_token_cnt),
        get_ptr<float>(expert_weight),
        token_num,
        expert_num,
        softmax_out.data_ptr(),
        get_ptr<float>(expert_compute_ema),
        ema_alpha,
        over_compute_ratio,
        get_ptr<float>(compute_capacity),
        get_ptr<int>(hot_expert_id),
        get_ptr<float>(load_balance_loss),
        get_ptr<float>(grad_scale)};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    linalg::moe::run_moe_gating_loss_forward(param, stream);

    return {load_balance_loss, compute_capacity, softmax_out, hot_expert_id, grad_scale};
}

at::Tensor moe_gating_loss_backward(
    const at::Tensor& output_grad,
    const at::Tensot& softmax_out,
    const at::Tensor& hot_expert_id,
    const at::Tensor& grad_scale) {
    const at::ScalarType _st = softmax_out.scalar_type();
    CHECK_INPUT(expert_token_cnt, at::ScalarType::Int);
    CHECK_INPUT(expert_weight, at::ScalarType::Float);
    int token_num = softmax_out.sizes()[0];
    int expert_num = softmax_out.sizes()[1];

    auto logits_grad = torch::empty_like(softmax_out);
    linalg::moe::GatingLossBackwardParam param{
        _st,
        get_ptr<float>(output_grad),
        softmax_out.data_ptr(),
        get_ptr<int>(hot_expert_id),
        get_ptr<float>(grad_scale),
        token_num,
        expert_num,
        logits_grad.data_ptr()};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    linalg::moe::run_moe_gating_loss_backward(param, stream);

    return logits_grad;
}

}  // namespace moe_op
}  // namespace linalg