#include <arm_neon.h>
#include <cstdint>
#include <cstddef>
#include <chrono>
#include <iostream>
#include <iomanip>

int64_t process_array_scalar(const int32_t* data, size_t n) {
    int64_t sum = 0;
    for (size_t i = 0; i < n; ++i) {
        int32_t val = data[i];
        if (val > 0) {
            sum += val;
        } else if (val < 0) {
            sum -= val;
        }
    }
    return sum;
}

int64_t process_array_neon(const int32_t* data, size_t n) {
    int64_t sum = 0;
    int32x4_t acc = vdupq_n_s32(0);
    size_t i = 0;
    
    for (; i + 3 < n; i += 4) {
        int32x4_t vec = vld1q_s32(data + i);
        uint32x4_t mask_pos = vcgtq_s32(vec, vdupq_n_s32(0));
        uint32x4_t mask_neg = vcltq_s32(vec, vdupq_n_s32(0));
        int32x4_t sign = vshrq_n_s32(vec, 31);
        int32x4_t abs_val = veorq_s32(vec, sign);
        abs_val = vsubq_s32(abs_val, sign);
        int32x4_t pos_part = vandq_s32(vec, reinterpret_cast<int32x4_t>(mask_pos));
        int32x4_t neg_part = vandq_s32(abs_val, reinterpret_cast<int32x4_t>(mask_neg));
        int32x4_t contrib = vorrq_s32(pos_part, neg_part);
        acc = vaddq_s32(acc, contrib);
    }
    
    int32_t temp[4];
    vst1q_s32(temp, acc);
    for (int j = 0; j < 4; ++j) {
        sum += temp[j];
    }
    
    for (; i < n; ++i) {
        int32_t val = data[i];
        if (val > 0) sum += val;
        else if (val < 0) sum -= val;
    }
    
    return sum;
}

template<typename Func>
double benchmark(Func func, const int32_t* data, size_t n, int iterations = 100) {
    auto start = std::chrono::high_resolution_clock::now();
    volatile int64_t result = 0;
    for (int i = 0; i < iterations; ++i) {
        result = func(data, n);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    return duration.count() / iterations;
}

int main() {
    constexpr size_t N = 100000;
    constexpr int ITERATIONS = 100;
    
    alignas(16) static int32_t* data = new int32_t[N];
    for (size_t i = 0; i < N; ++i) {
        data[i] = static_cast<int32_t>(i * 7 % 2000) - 1000;
    }
    
    int64_t scalar_result = process_array_scalar(data, N);
    int64_t neon_result = process_array_neon(data, N);
    
    std::cout << "  Scalar: " << scalar_result << "\n";
    std::cout << "  NEON:   " << neon_result << "\n";
    std::cout << "  Match:  " << (scalar_result == neon_result ? "YES" : "NO") << "\n\n";
    
    double scalar_time = benchmark(process_array_scalar, data, N, ITERATIONS);
    double neon_time = benchmark(process_array_neon, data, N, ITERATIONS);
    double speedup = scalar_time / neon_time;
    
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  Scalar time: " << scalar_time << " ms\n";
    std::cout << "  NEON time:   " << neon_time << " ms\n";
    std::cout << "  Speedup:     " << speedup << "x\n";
    
    delete[] data;
    return (scalar_result == neon_result) ? 0 : 1;
}
