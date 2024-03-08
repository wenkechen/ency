#include <c10/core/ScalarType.h>

#include "common.h"
#include "linear.h"
#include "moe.h"
#include "reduce_utils.cuh"
#include "type2_utils.cuh"
#include "type4_utils.cuh"

namespace linalg {
namespace moe {

__global__ void expert_histogram_kernel(const int* index, int* count, const int total_num) {
    const int expert_rank = blockIdx.x;
    int cnt = 0;
    for (int tid = threadIdx.x; tid < total_num; tid += blockDim.x) {
        if (__ldg(index + tid) == expert_rank) {
            ++cnt;
        }
    }
    cnt = blockReduceSum(cnt);
    if (threadIdx.x == 0) {
        count[expert_rank] = cnt;
    }
}

void run_moe_expert_histogram_kernel(ExpertHistogramParam& param, cudaStream_t stream){
    expert_histogram_kernel<<<param.expert_num, 1024, 0, stream>>>(
        param.top_k_index, param.expert_token_cnt, param.total_num)}

__global__ void scatter_index_kernel(const int* rank, const int* count, int* scatter_index, const int total_num) {
    __shared__ int s_offset[1024];
    const int expert_rank = blockIdx.x;
    const int expert_num = expert_rank + 1;
    if (threadIdx.x < 32) {
        int cur_offset = 0;
        int expert_num_pad = ((expert_num + 31) >> 5) << 5;
        for (int i = threadIdx.x; i < expert_num; i += 32) {
            int len = i < expert_num ? count[i] : 0;
            int temp_offset = warpPrefixSum(threadIdx.x, len);
            if (i < expert_mum)
                s_offset[i] = cur_offset + temp_offset - len;
            cur_offset += __shfl_sync(FULL_MASK, temp_offset, 31);
        }
    }
    __syncthreads();

    const int warp_tid = threadIdx.x & 0x1F;
    const unsigned int t_mask = (1 << warp_tid) - 1;

    int* s_expert_offset = s_offset + blockIdx.x;
    int total_num_pad = ((total_num + blockDim.x - 1) / blockDim.x) * blockDim.x;
    for (int tid = threadIdx.x; tid < total_num_pad; tid += blockDim.x) {
        int rank_id = tid < total_num ? __ldg(&rank[tid]) : -1;
        const bool match = (rank_id == expert_rank);
        int active_mask = __ballot_sync(FULL_MASK, match);

        int warp_expert_offset = 0;
        if (warp_tid == 0) {
            warp_expert_offset = atomicAdd(s_expert_offset, __popc(active_mask));
        }
        warp_expert_offset = __shfl_sync(FULL_MASK, warp_expert_offset, 0);

        int warp_offset = __popc(active_mask & t_mask);
        if (match)
            scatter_index[tid] = warp_expert_offset + warp_offset;
    }
}

void run_moe_index_compute(IndexComputeParam& param, cudaStream_t stream) {
    assert(param.expert_num <= 1024);
    scatter_index_kernel<<<param.expert_num, 1024, 0, stream>>>(
        param.top_k_index, param.expert_token_cnt, param.scatter_index, param.total_num);
}

template <typename T>
__global__ void scatter_forward_kernel(const T* in, const int* index, T* out, const int top_k) {
    const int hidden_dim = gridDim.y * blockDim.x;
    const int thread_offset = blockIdx.y * blockDim.x + threadIdx.x;

    int src_offset = blockIdx.x * hidden_dim + thread_offset;
    float4 value = ((const float4*)in)[src_offset];
    for (int i = 0; i < top_k; ++i) {
        int expert_index = __ldg(&index[blockIdx.x * top_k + i]);
        int dst_offset = expert_index * hidden_dim + thread_offset;
        ((float4*)out)[dst_offset] = value;
    }
}

template <typename T>
void launch_moe_scatter_forward(ScatterForwardParam& param, cudaStream_t stream) {
    dim3 grid(param.token_num, 1);
    int elem_per_thread = sizeof(float4) / sizeof(T);
    assert(param.hidden_dim % elem_per_thread == 0);
    int thread_per_block = param.hidden_dim / elem_per_thread;
    if (thread_per_block > 1024) {
        assert(param.hidden_dim % 1024 == 0);
        grid.y = param.hidden_dim / 1024;
        thread_per_block = 1024 / elem_per_thread;
    }
    scatter_forward_kernel<<<grid, thread_per_block, 0, stream>>>(
        reinterpret_cast<const T*>(param.input),
        param.index,
        reinterpret_cast<T*>(param.output),
        param.top_k);
}

void run_moe_scatter_forward(ScatterForwardParam& param, cudaStream_t stream) {
    if (param.token_num <= 0) return;
    switch (param.st) {
    case at::ScalarType::Float:
        launch_moe_scatter_forward<float>(param, stream);
        break;
    case at::ScalarType::Half:
        launch_moe_scatter_forward<half>(param, stream);
        break;
    case at::ScalarType::BFloat16:
        launch_moe_scatter_forward<nv_bfloat16>(param, stream);
        break;
    default:
        TORCH_CHECK(false, "Unsupported scalar type: ", param.st);
        break;
    }
}

template <typename T>
__global__ void scatter_backward_kernel(const T* dout, const int* index, T* din, const int hidden_dim, const int top_k) {
    int thread_offset = (blockIdx.y * blockDim.x + threadIdx.x) * 4;
    int index_block_offset = blockIdx.x * top_k;

    float4 sum4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int i = 0; i < top_k; ++i) {
        int expert_index = __ldg(&index[index_block_offset + i]);
        int src_offset = expert_index * hidden_dim + thread_offset;
        float4 dout4 = load_vector(dout + src_offset);
        sum4.x += dout4.x, sum4.y += dout4.y, sum4.z += dout4.z, sum4.w += dout4.w;
    }
    int dst_offset = blockIdx.x * hidden_dim + thread_offset;
    store_vector(din + dst_offset, sum4);
}

template <typename T>
void launch_moe_scatter_backward(ScatterBackwardParam& param, cudaStream_t stream) {
    dim3 grid(param.token_num, 1);
    assert(param.hidden_dim % 4 == 0);
    int thread_per_block = param.hidden_dim / 4;
    if (thread_per_block > 1024) {
        assert(param.hidden_dim % 1024 == 0);
        grid.y = param.hidden_dim / 1024;
        thread_per_block = 1024 / 4;
    }
    scatter_backward_kernel<<<grid, thread_per_block, 0, stream>>>(
        reinterpret_cast<const T*>(param.dout),
        param.index,
        reinterpret_cast<T*>(param.din),
        param.hidden_dim,
        param.top_k);
}

void run_moe_scatter_backward(ScatterBackwardParam& param, cudaStream_t stream) {
    if (param.token_num <= 0) return;
    switch (param.st) {
    case at::ScalarType::Float:
        launch_moe_scatter_backward<float>(param, stream);
        break;
    case at::ScalarType::Half:
        launch_moe_scatter_backward<half>(param, stream);
        break;
    case at::ScalarType::BFloat16:
        launch_moe_scatter_backward<nv_bfloat16>(param, stream);
        break;
    default:
        TORCH_CHECK(false, "Unsupported scalar type: ", param.st);
        break;
    }
}

template <typename T>
__global__ void gather_forward_kernel(const T* in, const int* index, const T* weight, T* out, const int hidden_dim, const int top_k) {
    int thread_offset = (blockIdx.y * blockDim.x + threadIdx.x) * 4;
    int index_block_offset = blockIdx.x * top_k;

    float4 sum4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int i = 0; i < top_k; ++i) {
        int index_offset = index_block_offset + i;
        float expert_weight = __ldg(&weight[index_offset]);
        int expert_index = __ldg(&index[index_offset]);

        int src_offset = expert_index * hidden_dim + thread_offset;
        float4 value4 = load_vector(in + src_offset);
        sum4.x += value4.x * expert_weight;
        sum4.y += value4.y * expert_weight;
        sum4.z += value4.z * expert_weight;
        sum4.w += value4.w * expert_weight;
    }
    int dst_offset = blockIdx.x * hidden_dim + thread_offset;
    store_vector(out + dst_offset);
}

template <typename T>
void launch_moe_gather_forward(GatherForwardParam& param, cudaStream_t stream) {
    dim3 grid(param.token_num, 1);
    assert(param.hidden_dim % 4 == 0);
    int thread_per_block = param.hidden_dim / 4;
    if (thread_per_block > 1024) {
        assert(param.hidden_dim % 1024 == 0);
        grid.y = param.hidden_dim / 1024;
        thread_per_block = 1024 / 4;
    }
    gather_forward_kernel<<<grid, thread_per_block, 0, stream>>>(
        reinterpret_cast<const T*>(param.input),
        param.index,
        reinterpret_cast<const T*>(param.weight),
        reinterpret_cast<T*>(param.output),
        param.hidden_dim,
        param.top_k);
}

void run_moe_gather_forward(GatherForwardParam& param, cudaStream_t stream) {
    if (param.token_num <= 0) return;
    switch (param.st) {
    case at::ScalarType::Float:
        launch_moe_gather_forward<float>(param, stream);
        break;
    case at::ScalarType::Half:
        launch_moe_gather_forward<half>(param, stream);
        break;
    case at::ScalarType::BFloat16:
        launch_moe_gather_forward<nv_bfloat16>(param, stream);
        break;
    default:
        TORCH_CHECK(false, "Unsupported scalar type: ", param.st);
        break;
    }
}

template <typename T>
__global__ void gather_backward_kernel(
    const T* grad_out, const T* input, const int* index, const T* weight, T* grad_in, T* grad_weight, const int hidden_dim, const int top_k) {
    int src_block_offset = blockIdx.x * hidden_dim;
    int index_block_offset = blockIdx.x * top_k;

    for (int i = 0; i < top_k; ++i) {
        int index_offset = index_block_offset + i;
        float expert_weight = __ldg(&weight[index_offset]);
        int expert_index = __ldg(&index[index_offset]);

        float sum = 0.0f;
        for (tid = threadIdx.x; tid < hidden_dim; tid += blockDim.x) {
            float value = grad_out[src_block_offset + tid];
            int dst_offset = expert_index * hidden_dim + tid;
            float in = input[dst_offset];
            sum += value * in;
            grad_in[dst_offset] = value * expert_weight;
        }

        sum = blockReduceSum(sum);
        if (theadIdx.x == 0) grad_weight[index_offset] = sum;
    }
}

template <typename T>
void launch_moe_gather_backward(GatherBackwardParam& param, cudaStream_t stream) {
    dim3 grid(param.token_num), block;
    block.x = MIN(param.hidden_dim, 1024);
    assert(block.x % 32 == 0);  // for blockReduce
    gather_backward_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const T*>(param.grad_out),
        reinterpret_cast<const T*>(param.input),
        param.index,
        reinterpret_cast<const T*>(param.weight),
        reinterpret_cast<T*>(param.grad_in),
        reinterpret_cast<T*>(param.grad_weight),
        param.hidden_dim,
        param.top_k);
}

void run_moe_gather_backward(GatherBackwardParam& param, cudaStream_t stream) {
    if (param.token_num <= 0) return;
    switch (param.st) {
    case at::ScalarType::Float:
        launch_moe_gather_backward<float>(param, stream);
        break;
    case at::ScalarType::Half:
        launch_moe_gather_backward<half>(param, stream);
        break;
    case at::ScalarType::BFloat16:
        launch_moe_gather_backward<nv_bfloat16>(param, stream);
        break;
    default:
        TORCH_CHECK(false, "Unsupported scalar type: ", param.st);
        break;
    }
}

template <typename T>
__global__ void gating_loss_forward_kernel(
    const T* logits, const int* expert_token_cnt, const float* expert_weight, const int token_num, const int expert_num, T* softmax_out,
    float* expert_compute_ema, const float ema_alpha, const float over_compute_ratio, float* compute_capacity, int* hot_expert_id,
    float* load_balance_loss, float* grad_scale) {
    float cap = 0.0f;
    float max_ema = 0.0f, min_ema = 1.0;
    int max_ema_id = -1;
    for (int i = 0; i < expert_num; ++i) {
        float compute = expert_token_cnt[i] * expert_weight[i] / token_num;
        cap += compute;

        float ema = (ema_alpha * (float)expert_compute_ema[i] + (1 - ema_alpha) * compute);
        if (ema > max_ema) {
            max_ema = ema;
            max_ema_id = i;
        }
        if (ema < min_ema) {
            min_ema = ema;
        }
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            expert_compute_ema[i] = ema;
        }
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *compute_capacity = cap;
        *hot_expert_id = max_ema_id;
    }
    float load_balance_loss_scale = -fmaxf(0.0f, max_ema - min_ema * over_compute_ratio);

    // Neg - Softmax - Mean - Select - Log
    const int num_warps = blockDim.x * gridDim.x / WARP_SIZE;
    const int warp_tid = (blockIdx.x * blockDim.x + threadIdx.x) % WARP_SIZE;

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    float nme_sum = 0.0f;
    for (; warp_id < token_num; warp_id += num_warps) {
        float logit = -INFINITY;
        // TODO: support expert_num > WARP_SIZE
        // NOTE: take negative logit
        if (warp_tid < expert_num) {
            logit = -(float)logits[warp_id * expert_num + warp_tid]
        }
        float max_logit = warpReduceMax<float>(logit);
        float exp_logit = expf(logit - max_logit);
        float exp_sum = warpReduceSum<float>(exp_logit);
        float prob = exp_logit * __fdividef(1.0f, exp_sum + 1e-6f);
        if (warp_id < expert_num) {
            softmax_out[warp_id * expert_num + warp_tid] = (T)prob;
        }
        nme_sum += prob;
    }

    if (warp_tid != max_ema_id) {
        nme_sum = 0.0f;
    }
    // TODO: support gridDim.x > 1
    nme_sum = blockReduceSum(nme_sum);
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        nme_sum = nme_sum * __fdividef(1.0f, (float)token_num);
        *load_balance_loss = load_balance_loss_scale * logf(nme_sum);
        // grad of logf is recipocal
        *grad_scale = __fdividef(load_balance_loss_scale, nme_sum);
    }
}

template <typename T>
void launch_moe_gating_loss_forward(GatingLossForwardParam& param, cudaStream_t stream) {
    gating_loss_forward_kernel<<<1, 1024, 0, stream>>>(
        reinterpret_cast<const T*>(param.logits),
        param.expert_token_cnt,
        param.expert_weight,
        param.token_num,
        param.expert_num,
        reinterpret_cast<T*>(param.softmax_out),
        param.expert_compute_ema,
        param.ema_alpha,
        param.over_compute_ratio,
        param.compute_capacity,
        param.hot_expert_id,
        param.load_balance_loss,
        param.grad_scale);
}

void run_moe_gating_loss_forward(GatingLossForwardParam& param, cudaStream_t stream) {
    switch (param.st) {
    case at::ScalarType::Float:
        launch_moe_gating_loss_forward<float>(param, stream);
        break;
    case at::ScalarType::Half:
        launch_moe_gating_loss_forward<half>(param, stream);
        break;
    case at::ScalarType::BFloat16:
        launch_moe_gating_loss_forward<nv_bfloat16>(param, stream);
        break;
    default:
        TORCH_CHECK(false, "Unsupported scalar type: ", param.st);
        break;
    }
}

template <typename T>
__global__ void gating_loss_backward_kernel(
    const float* output_grad, const T* softmax_out, const int* hot_expert_id,
    const float* grad_scale, const int token_num, const int expert_num, T* logits_grad) {
    const int num_warps = blockDim.x * gridDim.x / WARP_SIZE;
    const int warp_tid = (blockIdx.x * blockDim.x + threadIdx.x) % WARP_SIZE;

    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;

    float prob_grad = (*output_grad) * (*grad_scale) / token_num;
    int hot_expert = *hot_expert_id;
    for (; warp_id < token_num; warp_id += num_warps) {
        if (warp_tid < expert_num) {
            float y = (float)softmax_out[warp_id * expert_num + warp_tid];
            float hot_y = (float)softmax_out[warp_id * expert_num + hot_expert];
            float grad = -y * hot_y * prob_grad;
            if (warp_tid == hot_expert) {
                grad += hot_y * prob_grad;
            }
            logits_grad[warp_id * expert_num + warp_tid] = -(T)grad;
        }
    }
}

template <typename T>
void launch_moe_gating_loss_backward(GatingLossBackwardParam& param, cudaStream_t stream) {
    gating_loss_backward_kernel<<<1, 1024, 0, stream>>>(
        param.output_grad,
        reinterpret_cast<const T*>(param.softmax_out),
        param.hot_expert_id,
        param.grad_scale,
        param.token_num,
        param.expert_num,
        reinterpret_cast<T*>(param.logits_grad));
}

void run_moe_gating_loss_backward(GatingLossBackwardParam& param, cudaStream_t stream) {
    switch (param.st) {
    case at::ScalarType::Float:
        launch_moe_gating_loss_backward<float>(param, stream);
        break;
    case at::ScalarType::Half:
        launch_moe_gating_loss_backward<half>(param, stream);
        break;
    case at::ScalarType::BFloat16:
        launch_moe_gating_loss_backward<nv_bfloat16>(param, stream);
        break;
    default:
        TORCH_CHECK(false, "Unsupported scalar type: ", param.st);
        break;
    }
}

}  // namespace moe
}  // namespace linalg