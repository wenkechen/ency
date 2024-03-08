#include "arith_utils.cuh"

namespace linalg {
inline __device__ float2 load_type2(const float* ptr) {
    return __ldg((const float2*)ptr);
}

inline __device__ float2 load_type2(const half* ptr) {
    return type22float2(__ldg((const half2*)ptr));
}

inline __device__ float2 load_type2(const __nv_bfloat16* ptr) {
    return type22float2(__ldg((const __nv_bfloat162*)ptr));
}

inline __device__ void store_type2(float* ptr, const float2 a) {
    *((float2*)ptr) = a;
}

inline __device__ void store_type2(half* ptr, const float2 a) {
    *((half2*)ptr) = __float2half2_rn(a);
}

inline __device__ void store_type2(__nv_bfloat16* ptr, const float2 a) {
    *((__nv_bfloat162*)ptr) = __float22bfloat162_rn(a);
}

}  // namespace linalg