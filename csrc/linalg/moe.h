#pragma once
#include <ATen/cuda/CUDAContext.h>

namespace linalg {
namespace moe {
struct ExpertHistogramParam {
    const int* __restrict__ top_k_index;
    int* __restrict__ expert_token_cnt;
    const int total_num;
    const int expert_num;
};

struct IndexComputeParam {
    const int* __restrict__ top_k_index;
    const int* __restrict__ expert_token_cnt;
    int* __restrict__ scatter_index;
    const int total_num;
    const int expert_num;
};

struct ScatterForwardParam {
    c10::ScalarType st;
    const void* __restrict__ input;
    const int* __restrict__ index;
    void* __restrict__ output;
    const int token_num;
    const int top_k;
    const int hidden_dim;
};

struct ScatterBackwardParam {
    c10::ScalarType st;
    const void* __restrict__ dout;
    const int* __restrict__ index;
    void* __restrict__ din;
    const int token_num;
    const int top_k;
    const int hidden_dim;
};

struct GatherForwardParam {
    c10::ScalarType st;
    const void* __restrict__ input;
    const int* __restrict__ index;
    const void* __restrict__ weight;
    void* __restrict__ output;
    const int token_num;
    const int top_k;
    const int hidden_dim;
};

struct GatherBackwardParam {
    c10::ScalarType st;
    const void* __restrict__ grad_out;
    const void* __restrict__ input;
    const int* __restrict__ index;
    const void* __restrict__ weight;
    void* __restrict__ grad_in;
    void* __restrict__ grad_weight;
    const int token_num;
    const int top_k;
    const int hidden_dim;
};

struct GatingLossForwardParam {
    c10::ScalarType st;
    const void* __restrict__ logits;
    const int* __restrict__ expert_token_cnt;
    const float* __restrict__ expert_weight;
    const int token_num;
    const int expert_num;
    void* __restrict__ softmax_out;
    float* __restrict__ expert_compute_ema;
    const float ema_alpha;
    const float over_compute_ratio;
    float* __restrict__ compute_capacity;
    int* __restrict__ hot_expert_id;
    float* __restrict__ load_balance_loss;
    float* __restrict__ grad_scale;
};

struct GatingLossBackwardParam {
    c10::ScalarType st;
    const float* __restrict__ output_grad;
    const void* __restrict__ softmax_out;
    const int* __restrict__ hot_expert_id;
    const float* __restrict__ grad_scale;
    const int token_num;
    const int expert_num;
    void* __restrict__ logits_grad;
};

void run_moe_expert_histogram_kernel(ExpertHistogramParam& param, cudaStream_t stream);
void run_moe_index_compute(IndexComputeParam& param, cudaStream_t stream);
void run_moe_scatter_forward(ScatterForwardParam& param, cudaStream_t stream);
void run_moe_scatter_backward(ScatterBackwardParam& param, cudaStream_t stream);
void run_moe_gather_forward(GatherForwardParam& param, cudaStream_t stream);
void run_moe_gather_backward(GatherBackwardParam& param, cudaStream_t stream);
void run_moe_gating_loss_forward(GatingLossForwardParam& param, cudaStream_t stream);
void run_moe_gating_loss_backward(GatingLossBackwardParam& param, cudaStream_t stream);
}  // namespace moe
}  // namespace linalg