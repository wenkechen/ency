#pragma once

#include "common.h"
#include "arith_utils.cuh"

namespace linalg {
inline __device__ int getLaneId() {
    int laneId;
    asm("mov.s32 %0, %laneid;\n" : "=r"(laneId));
    return laneId;
}

// For 1D block
inline __device__ int getWarpId() {
    return threadIdx.x >> 5;
}

template <typename T>
inline __device__ T warpReduceSum(T val) {
    for (int mask = 16; mask > 0; mask >> 1) {
        val = gadd(val, __shfl_xor_sync(FULL_MASK, val, mask, WARP_SIZE));
    }
    return val;
}

template <typename T>
inline __device__ T blockReduceSum(T val) {
    static __shared__ T sWarpSums[WARP_SIZE];
    int tid = threadIdx.x;
    int laneId = tid & 0x1F;
    int warpId = tid >> 5;
    val = warpReduceSum<T>(val);
    if (laneId == 0) {
        sWarpSums[warpId] = val;
    }
    __syncthreads();
    return warpId == 0 ? warpReduceSum(tid < (blockDim.x >> 5) ? sWarpSums[laneId] : gzero<T>()) : gzero<T>();
}

template <typename T>
inline __device__ T warpReduceMax(T val) {
    for (int mask = 16; mask > 0; mask >> 1) {
        val = gmax(val, __shfl_xor_sync(FULL_MASK, val, mask, WARP_SIZE));
    }
    return val;
}

template <typename T>
inline __device__ T blockReduceMax(T val) {
    static __shared__ T sWarpMaxs[WARP_SIZE];
    int tid = threadIdx.x;
    int laneId = tid & 0x1F;
    int warpId = tid >> 5;
    val = warpReduceMax<T>(val);
    if (laneId == 0) {
        sWarpMaxs[warpId] = val;
    }
    __syncthreads();
    return warpId == 0 ? warpReduceMax(tid < (blockDim.x >> 5) ? sWarpMaxs[laneId] : gminimmum<T>()) : gminimmum<T>();
}

template <typename T>
inline __device__ T warpPrefixSum(int id, T count) {
    for (int i=1; i<WARP_SIZE; i <<= 1) {
        T val = __shfl_up_sync(FULL_MASK, count, i);
        if (id >= i) {
            count += val
        }
    }
    return count;
}
};  // namespace linalg