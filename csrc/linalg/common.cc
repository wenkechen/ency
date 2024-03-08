#include "common.h"

namespace linalg {
int cal_vector_size(int n_args, ...) {
    va_list ap;
    va_start(ap, n_args);
    int vec_size = va_arg(ap, int);
    for (int i = 1; i < n_args; ++i) {
        if (vec_size == 1) break;
        int n = va_arg(ap, int);
        vec_size = std::gcd(vec_size, n);
    }
    va_end(ap);
    return vec_size;
}

GPUClock::GPUClock() {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
    cudaEventRecord(start_);
}
GPUClock::~GPUClock() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
}
void GPUClock::start() {
    cudaEventRecord(start_);
}
float GPUClock::milliseconds() {
    cudaEventRecord(stop_);
    cudaEventSynchronize(stop_);
    float time;
    cudaEventElapsedTime(&time, start_, stop_);
    return time;
}

}  // namespace linalg