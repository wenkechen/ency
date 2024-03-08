#include "common.h"
#include "reduce_utils.cuh"
#include "type2_utils.cuh"
#include "type4_utils.cuh"

#define BASE 10000
#define HEAD_DIM 128
#define NUM_HEADS 11

__device__ int find_batch_idx(const int token_idx, const int batch_size, const int* cu_seqlens) {
    int l = 0, r = batch_size, batch_idx = -1;
    while (l < r) {
        int m = (l + r) / 2;
        if (cu_seqlens[m] > token_idx) {
            batch_idx = m - 1;
            r = m - 1;
        } else {
            l = m + 1;
        }
    }
    return batch_idx;
}

__global__ void rotary_kernel(
    bool interleaved,
    int batch_size,
    int* seqlens,
    int* seqlen_offsets,
    int* cu_seqlens,
    float* inv_freq,
    int num_tokens,
    float* input,
    float* output) {
    const int ROTARY_DIM = HEAD_DIM / 2;
    extern __shared__ char smem[];
    int* seqlen_offsets_s = (int*)(smem);
    int* cu_seqlens_s = seqlen_offsets_s + batch_size;
    float* inv_freq_s = (float*)(cu_seqlens_s + batch_size + 1);

    for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
        seqlen_offsets_s[i] = seqlen_offsets[i];
    }

    for (int i = threadIdx.x; i <= batch_size; i += blockDim.x) {
        cu_seqlens_s[i] = cu_seqlens[i];
    }

    for (int i = threadIdx.x; i < ROTARY_DIM; i += blockDim.x) {
        inv_freq_s[i] = inv_freq[i];
    }

    __syncthreads();

    const int token_idx = blockIdx.x;
    const int batch_idx = find_batch_idx(token_idx, batch_size, cu_seqlen_s);
    const int seqlen_offset = seqlen_offsets_s[batch_idx]

    // dim3 grid(num_tokens);
    // dim3 block(std::min(NUM_HEADS * HEAD_DIM / 2, 128));

    for(int i=threadIdx.x; i< NUM_HEADS * ROTARY_DIM; i += blockDim.x) {
        const int head_idx = i / ROTARY_DIM;
        const int m = i % ROTARY_DIM; // m is theta idx
        const int sid = token_idx - cu_seqlens_s[batch_idx] + seqlen_offset;
        const float theta = inv_freq_s[m] * float(sid);
        const float cos_theta = cos(theta), sin_theta = sin(theta);
        
        float* token_input = input + token_idx * NUM_HEADS * HEAD_DIM + head_idx * HEAD_DIM;
        float* token_output = output + token_idx * NUM_HEADS * HEAD_DIM + head_idx * HEAD_DIM;

        if (interleaved) {
            token_output[m] = __ldg(token_input + m) * cos_theta - __ldg(token_input + 2 * m - 1) * sin_theta;
            token_output[2 * m - 1] = __ldg(token_input + m) * cos_theta + __ldg(token_input + 2 * m - 1) * sin_theta;
        } else {
            token_output[m] = __ldg(token_input + m) * cos_theta - __ldg(token_input + m + HEAD_DIM / 2) * sin_theta;
            token_output[m + HEAD_DIM / 2] = __ldg(token_input + m) * cos_theta + __ldg(token_input + m + HEAD_DIM / 2) * sin_theta;
        }
    }
}