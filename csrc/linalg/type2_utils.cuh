#include "arith_utils.cuh"

namespace linalg {
inline __device__ float2 load_type2(const float* ptr) {
    return __ldg((const float2*)ptr);
}

inline __device__ float2 load_type2(const half* ptr) {
    return __half22float2(__ldg((const half2*)ptr));
}

inline __device__ float2 load_type2(const __nv_bfloat16* ptr) {
    return bf1622float2(__ldg((const __nv_bfloat162*)ptr));
}

inline __device__ void store_type2(float* ptr, const float2 x) {
    *((float2*)ptr) = x;
}

inline __device__ void store_type2(half* ptr, const float2 x) {
    *((half2*)ptr) = __float2half2_rn(x);
}

inline __device__ void store_type2(__nv_bfloat16* ptr, const float2 x) {
    *((__nv_bfloat162*)ptr) = float22bf162(x);
}

}  // namespace linalg