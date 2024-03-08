#pragma once
#include <ATen/core/Tensor.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdarg>
#include <numeric>
#include <type_traits>

namespace linalg {
#define CHECK_TYPE(x, st) TORCH_CHECK(x.scalar_type() == st, "Inconsistency of Tensor type: " #x)
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x, st)   \
    {                        \
        CHECK_CUDA(x);       \
        CHECK_CONTIGUOUS(x); \
        CHECK_TYPE(x, st);   \
    }
#define CHECK_INPUT_LOOSE(x, st) \
    {                            \
        CHECK_CUDA(x);           \
        CHECK_CONTIGUOUS(x);     \
    }
#define CHECK_CPU(x) TORCH_CHECK(!x.is_cuda(), #x " must be a CPU tensor")
#define CHECK_INPUT_CPU(x, st) \
    {                          \
        CHECK_CPU(x);          \
        CHECK_CONTIGUOUS(x);   \
        CHECK_TYPE(x, st);     \
    }

static const char* _cudaGetErrorEnum(cudaError_t error) {
    return cudaGetErrorString(error);
}

static const char* _cudaGetErrorEnum(cublasStatus_t error) {
    switch (error) {
    case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";

    case CUBLAS_STATUS_NOT_SUPPORTED:
        return "CUBLAS_STATUS_NOT_SUPPORTED";

    case CUBLAS_STATUS_LICENSE_ERROR:
        return "CUBLAS_STATUS_LICENSE_ERROR";
    }
    return "<unknown>";
}

template <typename T>
void check(T result, char const* const func, const char* const file, int const line) {
    if (result) {
        fprintf(stderr, "[ENCY][ERROR ]CUDA runtime error = %s at %s:%d '%s'\n", _cudaGetErrorEnum(result), file, line, func);
        exit(1);
    }
}

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
#define WARP_SIZE 32
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DIVIDE_ROUND_UP(a, b) (((a) + (b)-1) / (b))
#define FULL_MASK 0xFFFFFFFF

template <typename T>
inline T* get_ptr(at::Tensor& t) {
    return reinterpret_cast<T*>(t.data_ptr());
}

template <typename T>
inline const T* get_ptr(const at::Tensor& t) {
    return reinterpret_cast<const T*>(t.data_ptr());
}

template <typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t) * vec_size) aligned_vector {
    scalar_t val[vec_size];
};

template <typename scalar_t>
inline int can_vectorize_up_to(char* pointer) {
    uint64_t address = reinterpret_cast<uint64_t>(pointer);
    constexpr int vec2_alignment = std::alignment_of<aligned_vector<scalar_t, 2>>::value;
    constexpr int vec4_alignment = std::alignment_of<aligned_vector<scalar_t, 4>>::value;
    if (address % vec4_alignment == 0) {
        return 4;
    } else if (address % vec2_alignment == 0) {
        return 2;
    }
    return 1;
}

template <typename scalar_t>
int get_vector_size(const at::Tensor& t) {
    int vec_size = 4;
    if (!t.is_non_overlapping_and_dense()) {
        vec_size = 1;
    } else {
        vec_size = memory::can_vectorize_up_to<scalar_t>((char*)t.data_ptr());
    }
    bool can_vectorize = true;
    do {
        can_vectorize = t.numel() % vec_size == 0;
        if (!can_vectorize) vec_size /= 2;
    } while (vec_size > 1 && !can_vectorize);
    return can_vectorize ? vec_size : 1;
}

int cal_vector_size(int n_args, ...);

class GPUClock {
public:
    GPUClock();
    ~GPUClock();

    void start();
    float milliseconds();

private:
    cudaEvent_t start_, stop_;
};

}  // namespace linalg
