#include <arm_neon.h>
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <iomanip>

__attribute__((noinline, optimize("no-tree-vectorize")))
int64_t process_array_scalar(const int32_t* data, size_t n) {
    int64_t sum = 0;
    for (size_t i = 0; i < n; ++i) {
        int32_t val = data[i];
        if (val > 0)      sum += val;
        else if (val < 0) sum -= val;
    }
    return sum;
}

__attribute__((noinline))
int64_t process_array_neon(const int32_t* __restrict__ data, size_t n) {
    int64x2_t acc0 = vdupq_n_s64(0);
    int64x2_t acc1 = vdupq_n_s64(0);
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        __builtin_prefetch(data + i + 32, 0 /*read*/, 1 /*L2*/);

        int32x4_t vec0 = vld1q_s32(data + i);
        int32x4_t vec1 = vld1q_s32(data + i + 4);
        int32x4_t sign0 = vshrq_n_s32(vec0, 31);
        int32x4_t sign1 = vshrq_n_s32(vec1, 31);
        int32x4_t abs0 = vsubq_s32(veorq_s32(vec0, sign0), sign0);
        int32x4_t abs1 = vsubq_s32(veorq_s32(vec1, sign1), sign1);
        acc0 = vaddq_s64(acc0, vpaddlq_s32(abs0));
        acc1 = vaddq_s64(acc1, vpaddlq_s32(abs1));
    }
    for (; i + 3 < n; i += 4) {
        int32x4_t vec  = vld1q_s32(data + i);
        int32x4_t sign = vshrq_n_s32(vec, 31);
        int32x4_t abs_val = vsubq_s32(veorq_s32(vec, sign), sign);
        acc0 = vaddq_s64(acc0, vpaddlq_s32(abs_val));
    }
    int64x2_t acc = vaddq_s64(acc0, acc1);
    int64_t sum = vgetq_lane_s64(acc, 0) + vgetq_lane_s64(acc, 1);
    for (; i < n; ++i) {
        int32_t val = data[i];
        if (val > 0)      sum += val;
        else if (val < 0) sum -= val;
    }

    return sum;
}

template<typename Func>
double benchmark(Func func, const int32_t* data, size_t n, int iterations) {
    volatile int64_t warmup = func(data, n);
    (void)warmup;
    auto start = std::chrono::high_resolution_clock::now();
    volatile int64_t result = 0;
    for (int i = 0; i < iterations; ++i)
        result = func(data, n);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dur = end - start;
    return dur.count() / iterations;
}

int main() {
    constexpr size_t N = 1'000'000;
    constexpr int ITERATIONS = 200;
    int32_t* data = static_cast<int32_t*>(
        std::aligned_alloc(16, N * sizeof(int32_t)));
    for (size_t i = 0; i < N; ++i)
        data[i] = static_cast<int32_t>(i * 7 % 2001) - 1000;
    int64_t scalar_result = process_array_scalar(data, N);
    int64_t neon_result   = process_array_neon  (data, N);
    std::cout << "  Scalar: " << scalar_result << "\n";
    std::cout << "  NEON:   " << neon_result   << "\n";
    std::cout << "  Match:  " << (scalar_result == neon_result ? "YES ✓" : "NO ✗") << "\n\n";
    double scalar_ms = benchmark(process_array_scalar, data, N, ITERATIONS);
    double neon_ms   = benchmark(process_array_neon,   data, N, ITERATIONS);
    double speedup   = scalar_ms / neon_ms;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  Scalar time: " << scalar_ms << " ms\n";
    std::cout << "  NEON time:   " << neon_ms   << " ms\n";
    std::cout << "  Speedup:     " << speedup   << "x\n";

    std::free(data);
    return (scalar_result == neon_result) ? 0 : 1;
}
