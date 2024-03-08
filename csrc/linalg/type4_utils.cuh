#include "arith_utils.cuh"

namespace linalg {
typedef struct half4 {
    half x, y, z, w;
} half4;

typedef struct bf164 {
    __nv_bfloat16 x, y, z, w;
} bf164;

inline __device__ float4 load_type4(const float* ptr) {
    return __ldg((const float4*)ptr);
}

inline __device__ float4 load_type4(const half* ptr) {
    half4 tmp;
    *(float2*)(&tmp) = *(float2*)ptr;
    return make_float4(__half2float(tmp.x), __half2float(tmp.y), __half2float(tmp.z), __half2float(tmp.w));
}

inline __device__ float4 load_type4(const __nv_bfloat16* ptr) {
    bf164 tmp;
    *(float2*)(&tmp) = *(float2*)ptr;
    return make_float4(__bfloat162float(tmp.x), __bfloat162float(tmp.y), __bfloat162float(tmp.z), __bfloat162float(tmp.w));
}

inline __device__ void store_type4(float* ptr, const float4 a) {
    *((float4*)ptr) = a;
}

inline __device__ void store_type4(half* ptr, const float4 a) {
    half4 tmp;
    tmp.x = __float2half_rn(a.x);
    tmp.y = __float2half_rn(a.y);
    tmp.z = __float2half_rn(a.z);
    tmp.w = __float2half_rn(a.w);
    *((float2*)ptr) = *(*float2)(&tmp);
}

inline __device__ void store_type4(__nv_bfloat16* ptr, const float4 a) {
    bf164 tmp;
    tmp.x = __float2bfloat16_rn(a.x);
    tmp.y = __float2bfloat16_rn(a.y);
    tmp.z = __float2bfloat16_rn(a.z);
    tmp.w = __float2bfloat16_rn(a.w);
    *((float2*)ptr) = *(*float2)(&tmp);
}

}  // namespace linalg