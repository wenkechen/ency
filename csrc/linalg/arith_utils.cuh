#pragma once
#include <c10/cuda/CUDAMathCompat.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cfloat>
#include <climits>
#include <cmath>

#include "common.h"

namespace linalg {
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
    return __bfloat1622float2(val);
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
    return __float22bfloat162_rn(val);
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
    return __bfloat162bfloat162(val);
}

// gadd
template <typename T>
inline __device__ T gadd(T a, T b) {
    return a + b;
}

template <>
inline __device__ __nv_bfloat162 gadd(__nv_bfloat162 a, __nv_bfloat162 b) {
    return __hadd2(a, b);
}

template <>
inline __device__ __nv_bfloat16 gadd(__nv_bfloat16 a, __nv_bfloat16 b) {
    return __hadd(a, b);
}

inline __device__ __nv_bfloat16 gadd(__nv_bfloat16 a, float b) {
    return __hadd(a, __float2bfloat16_rn(b));
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
    a.x += b.x;
    a.y += b.y;
    return a;
}

template <>
inline __device__ float4 gadd(float4 a, float4 b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
    return a;
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

// gsub
template <typename T>
inline __device__ T gsub(T a, T b) {
    return a - b;
}

template <>
inline __device__ __nv_bfloat162 gsub(__nv_bfloat162 a, __nv_bfloat162 b) {
    return __hsub2(a, b);
}

template <>
inline __device__ __nv_bfloat16 gsub(__nv_bfloat16 a, __nv_bfloat16 b) {
    return __hsub(a, b);
}

inline __device__ __nv_bfloat16 gsub(__nv_bfloat16 a, float b) {
    return __hsub(a, __float2bfloat16_rn(b));
}

template <>
inline __device__ half2 gsub(half2 a, half2 b) {
    return __hsub2(a, b);
}

template <>
inline __device__ half gsub(half a, half b) {
    return __hsub(a, b);
}

template <>
inline __device__ float2 gsub(float2 a, float2 b) {
    a.x -= b.x;
    a.y -= b.y;
    return a;
}

template <>
inline __device__ float4 gsub(float4 a, float4 b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
    return a;
}

// gmul
template <typename T>
inline __device__ T gmul(T a, T b) {
    return a * b;
}

template <>
inline __device__ __nv_bfloat162 gmul(__nv_bfloat162 a, __nv_bfloat162 b) {
    return __hmul2(a, b);
}

template <>
inline __device__ __nv_bfloat16 gmul(__nv_bfloat16 a, __nv_bfloat16 b) {
    return __hmul(a, b);
}

inline __device__ __nv_bfloat16 gmul(__nv_bfloat16 a, float b) {
    return __hmul(a, __float2bfloat16_rn(b));
}

template <>
inline __device__ half2 gmul(half2 a, half2 b) {
    return __hmul2(a, b);
}

template <>
inline __device__ half gmul(half a, half b) {
    return __hmul(a, b);
}

template <>
inline __device__ float2 gmul(float2 a, float2 b) {
    a.x *= b.x;
    a.y *= b.y;
    return a;
}

template <>
inline __device__ float4 gmul(float4 a, float4 b) {
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
    return a;
}

template <typename T>
inline __device__ T gmul(T a, T b, T c) {
    return gmul(gmul(a, b), c);
}

template <typename T>
inline __device__ T gmul(T a, T b, T c, T d) {
    return gmul(gmul(a, b), gmul(c, d));
}

// gfma
template <typename T>
inline __device__ T gfma(T a, T b, T c) {
    return gadd(gmul(a, b), c);
}

inline __device__ __nv_bfloat162 gfma(__nv_bfloat162 a, __nv_bfloat162 b, __nv_bfloat162 c) {
    return __hfma2(a, b, c);
}

inline __device__ __nv_bfloat16 gfma(__nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 c) {
    return __hfma(a, b, c);
}

inline __device__ half2 gfma(half2 a, half2 b, half2 c) {
    return __hfma2(a, b, c);
}

inline __device__ half gfma(half a, half b, half c) {
    return __hfma(a, b, c);
}

template <typename T>
inline __device__ T gfma(T a, T b, T c, T d) {
    return gadd(gmul(a, b, c), d);
}

// gexp
template <typename T>
inline __device__ T gexp(const T a) {
    return (T)std::exp(float(a));
}

inline __device__ __nv_bfloat162 gexp(__nv_bfloat162 a) {
    return h2exp(a);
}

inline __device__ __nv_bfloat16 gexp(__nv_bfloat16 a) {
    return hexp(a);
}

inline __device__ half2 gexp(half2 a) {
    return h2exp(a, b, c);
}

inline __device__ half gexp(half a) {
    return hexp(a, b, c);
}

template <>
inline __device__ float2 gexp(float2 a) {
    a.x = std::exp(a.x);
    a.y = std::exp(a.y);
    return a;
}

template <>
inline __device__ float4 gexp(float4 a) {
    a.x = std::exp(a.x);
    a.y = std::exp(a.y);
    a.z = std::exp(a.z);
    a.w = std::exp(a.w);
    return a;
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

template <typename T>
inline __device__ T fast_tanh(T x) {
    return c10::cuda::compat::tanh(x);
}

template <>
inline __device__ float fast_tanh(float x) {
    float y;
    // TODO: check whether volatile is must
    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

}  // namespace linalg