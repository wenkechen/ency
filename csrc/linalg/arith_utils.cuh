#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <climits>
#include <cfloat>

#include "common.h"

namespace linalg {
inline __device__ float2 bf1622float2(const __nv_bfloat162 val) {
    return __bfloat1622float2(val);
}

inline __device__ __nv_bfloat162 float22bf162(const float2 val) {
    return __float22bfloat162_rn(val);
}

inline __device__ __nv_bfloat162 bf162bf162(const __nv_bfloat16 val) {
    return __bfloat162bfloat162(val);
}

inline __device__ __nv_bfloat162 bf16hadd2(const __nv_bfloat162 x, const __nv_bfloat162 y) {
    return __hadd2(x, y);
}

inline __device__ __nv_bfloat16 bf16hadd(const __nv_bfloat16 x, const __nv_bfloat16 y) {
    return __hadd(x, y);
}

inline __device__ __nv_bfloat162 bf16hsub2(const __nv_bfloat162 x, const __nv_bfloat162 y) {
    return __hsub2(x, y);
}

inline __device__ __nv_bfloat16 bf16hsub(const __nv_bfloat16 x, const __nv_bfloat16 y) {
    return __hsub(x, y);
}

inline __device__ __nv_bfloat162 bf16hmul2(const __nv_bfloat162 x, const __nv_bfloat162 y) {
    return __hmul2(x, y);
}

inline __device__ __nv_bfloat16 bf16hmul(const __nv_bfloat16 x, const __nv_bfloat16 y) {
    return __hmul(x, y);
}

inline __device__ __nv_bfloat162 bf16hfma2(const __nv_bfloat162 x, const __nv_bfloat162 y, const __nv_bfloat162 z) {
    return __hfma2(x, y, z);
}

inline __device__ __nv_bfloat16 bf16hfma(const __nv_bfloat16 x, const __nv_bfloat16 y, const __nv_bfloat16 z) {
    return __hfma(x, y, z);
}

inline __device__ __nv_bfloat162 bf16exp2(const __nv_bfloat162 x) {
    return h2exp(x);
}

template <typename T>
inline __device__ T ldg(const T* ptr) {
    return __ldg(ptr);
}

// template <>
// inline __device__ __nv_bfloat162 ldg(const __nv_bfloat162* ptr) {
//     return __ldg(ptr);
// }

// template <>
// inline __device__ __nv_bfloat16 ldg(const __nv_bfloat16* ptr) {
//     return __ldg(ptr);
// }

// Get type2 from type or vice versa (applied to half and bfloat16)
template <typename T>
struct TypeConverter;

template <>
struct TypeConverter<half2> {
    using Type = half;
};

template <>
struct TypeConverter<half> {
    using Type = half2;
};

template <>
struct TypeConverter<__nv_bfloat162> {
    using Type = __nv_bfloat16;
};

template <>
struct TypeConverter<__nv_bfloat16> {
    using Type = __nv_bfloat162;
};

// Convert float to type2 (applied to half2 and blfoat162)
template <typename T>
inline __device__ T float2type2(float val);

template <>
inline __device__ half2 float2type2(float val) {
    return __float2half2_rn(val);
}

template <>
inline __device__ __nv_bfloat162 float2type2(float val) {
    return __float2bfloat162_rn(val);
}

// Convert float to type (applied to half and blfoat16)
template <typename T>
inline __device__ T float2type(float val);

template <>
inline __device__ half float2type(float val) {
    return __float2half_rn(val);
}

template <>
inline __device__ __nv_bfloat16 float2type(float val) {
    return __float2bfloat16_rn(val);
}

// Convert type to float (applied to half and blfoat16)
template <typename T>
inline __device__ float type2float(T val);

template <>
inline __device__ float type2float(half val) {
    return __half2float(val);
}

template <>
inline __device__ float type2float(__nv_bfloat16 val) {
    return __bfloat162float(val);
}

// Convert type2 to float2 (applied to half2 and blfoat162)
template <typename T>
inline __device__ float2 type22float2(T val);

template <>
inline __device__ float2 type22float2(half2 val) {
    return __half22float2(val);
}

template <>
inline __device__ float2 type22float2(__nv_bfloat162 val) {
    return bf1622float2(val);
}

// Convert float2 to type2 (applied to half and blfoat16)
template <typename T>
inline __device__ T float22type2(float2 val);

template <>
inline __device__ half2 float22type2(float2 val) {
    return __float2half2_rn(val);
}

template <>
inline __device__ __nv_bfloat162 float22type2(float2 val) {
    return float22bf162(val);
}

// Convert type to type2 (applied to half and blfoat16)
template <typename IN, typename OUT>
inline __device__ OUT type2type2(IN val);

template <>
inline __device__ half2 type2type2(half val) {
    return __half2half2(val);
}

template <>
inline __device__ __nv_bfloat162 type2type2(__nv_bfloat16 val) {
    return bf162bf162(val);
}

template <typename T>
inline __device__ T hadd2(T a, T b) {
    return __hadd2(a, b);
}

template <>
inline __device__ __nv_bfloat162 hadd2(__nv_bfloat162 a, __nv_bfloat162 b) {
    return bf16hadd2(a, b);
}

// gadd
template <typename T>
inline __device__ T gadd(T a, T b) {
    return a + b;
}

template <>
inline __device__ __nv_bfloat162 gadd(__nv_bfloat162 a, __nv_bfloat162 b) {
    return bf16hadd2(a, b);
}

template <>
inline __device__ __nv_bfloat16 gadd(__nv_bfloat16 a, __nv_bfloat16 b) {
    return bf16hadd(a, b);
}

inline __device__ __nv_bfloat16 gadd(__nv_bfloat16 a, float b) {
    return bf16hadd(a, __float2bfloat16_rn(b));
}

template <>
inline __device__ half2 gadd(half2 a, half2 b) {
    return __hadd2(a, b);
}

template <>
inline __device__ half gadd(half a, half b) {
    return __hadd(a, b);
}

template <>
inline __device__ float2 gadd(float2 a, float2 b) {
    float2 c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    return c;
}

template <>
inline __device__ float4 gadd(float4 a, float4 b) {
    float4 c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    c.z = a.z + b.z;
    c.w = a.w + b.w;
    return c;
}

template <typename T>
inline __device__ T gadd(T a, T b, T c) {
    return gadd(gadd(a, b), c);
}

template <typename T>
inline __device__ T gadd(T a, T b, T c, T d) {
    return gadd(gadd(a, b), gadd(c, d));
}

template <typename T>
inline __device__ T gadd(T a) {
    return a;
}

template <>
inline __device__ __nv_bfloat16 gadd(__nv_bfloat162 a) {
    return gadd(a.x, a.y);
}

template <>
inline __device__ half gadd(half2 a) {
    return gadd(a.x, a.y);
}

template <>
inline __device__ float gadd(float2 a) {
    return gadd(a.x, a.y);
}

template <>
inline __device__ float gadd(float4 a) {
    return gadd(a.x, a.y, a.z, a.w);
}

template <typename T>
inline __device__ T hsub2(T a, T b) {
    return __hsub2(a, b);
}

template <>
inline __device__ __nv_bfloat162 hsub2(__nv_bfloat162 a, __nv_bfloat162 b) {
    return bf16hsub2(a, b);
}

template <typename T>
inline __device__ T hmul2(T a, T b) {
    return __hmul2(a, b);
}

template <>
inline __device__ __nv_bfloat162 hmul2(__nv_bfloat162 a, __nv_bfloat162 b) {
    return bf16hmul2(a, b);
}

template <typename T>
inline __device__ T hmul2(T a, T b, T c) {
    return hmul2(hmul2(a, b), c);
}

template <typename T>
inline __device__ T hmul(T a, T b, T c) {
    return a * b * c;
}

inline __device__ __nv_bfloat162 hmul(__nv_bfloat162 a, __nv_bfloat162 b, __nv_bfloat162 c) {
    return hmul2(a, b, c);
}

inline __device__ __nv_bfloat16 hmul(__nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 c) {
    return bf16hmul(bf16hmul(a, b), c);
}

template <typename T>
inline __device__ T fma(T a, T b, T c) {
    return gadd(hmul(a, b), c);
}

template <typename T>
inline __device__ T fma(T a, T b, T c, T d) {
    return gadd(hmul(a, b, c), d);
}

template <typename T>
inline __device__ T hexp2(const T x) {
    return h2exp(x);
}

// gmax
template <typename T>
inline __device__ T gmax(T a, T b) {
    return MAX(a, b)
}

template <>
inline __device__ __nv_bfloat162 gmax(__nv_bfloat162 a, __nv_bfloat162 b) {
    __nv_bfloat162 c;
    c.x = MAX(a.x, b.x);
    c.y = MAX(a.y, b.y);
    return c;
}

template <>
inline __device__ half2 gmax(half2 a, half2 b) {
    half2 c;
    c.x = MAX(a.x, b.x);
    c.y = MAX(a.y, b.y);
    return c;
}

template <>
inline __device__ float2 gmax(float2 a, float2 b) {
    float2 c;
    c.x = MAX(a.x, b.x);
    c.y = MAX(a.y, b.y);
    return c;
}

inline __device__ float4 gmax(float4 a, float4 b) {
    float4 c;
    c.x = MAX(a.x, b.x);
    c.y = MAX(a.y, b.y);
    c.z = MAX(a.z, b.z);
    c.w = MAX(a.w, b.w);
    return c;
}

template <typename T>
inline __device__ T gmax(T a, T b, T c) {
    return gmax(gmax(a, b), c);
}

template <typename T>
inline __device__ T gmax(T a, T b, T c, T d) {
    return gmax(gmax(a, b), gmax(c, d));
}

template <typename T>
inline __device__ T gmax(T a) {
    return a;
}

template <>
inline __device__ __nv_bfloat16 gmax(__nv_bfloat162 a) {
    return gmax(a.x, a.y);
}

template <>
inline __device__ half gmax(half2 a) {
    return gmax(a.x, a.y);
}

template <>
inline __device__ float gmax(float2 a) {
    return gmax(a.x, a.y);
}

template <>
inline __device__ float gmax(float4 a) {
    return gmax(a.x, a.y, a.z, a.w);
}

// gzero
template <typename T>
inline __device__ T gzero() {
    return 0;
}

template <>
inline __device__ __nv_bfloat16 gzero() {
    return __float2bfloat16_rn(0.0f);
}

template <>
inline __device__ half gzero() {
    return __float2half_rn(0.0f);
}

template <>
inline __device__ float gzero() {
    return 0.0f;
}

template <>
inline __device__ float2 gzero() {
    return make_float2(0.0f, 0.0f);
}

template <>
inline __device__ float4 gzero() {
    return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
}

// gminimum
template <typename T>
inline __device__ T gminimmum() {
    return -INT_MAX;
}

template <>
inline __device__ __nv_bfloat16 gminimmum() {
    return __float2bfloat16_rn(-FLT_MAX);
}

template <>
inline __device__ half gminimmum() {
    return __float2half_rn(-FLT_MAX);
}

template <>
inline __device__ float gminimmum() {
    return -FLT_MAX;
}

template <>
inline __device__ float2 gminimmum() {
    return make_float2(-FLT_MAX, -FLT_MAX);
}

template <>
inline __device__ float4 gminimmum() {
    return make_float4(-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);
}

}  // namespace linalg