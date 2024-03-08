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
}  // namespace linalg