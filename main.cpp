#include <arm_neon.h>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <chrono>
#include <random>

constexpr size_t N = 1000000;

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
    int32x4_t zero = vdupq_n_s32(0);

    size_t i = 0;

    for (; i + 3 < n; i += 4) {
        __builtin_prefetch(data + i + 16);

        int32x4_t vec = vld1q_s32(data + i);

        uint32x4_t mask_pos = vcgtq_s32(vec, zero);
        uint32x4_t mask_neg = vcltq_s32(vec, zero);

        int32x4_t sign = vshrq_n_s32(vec, 31);
        int32x4_t abs_val = veorq_s32(vec, sign);
        abs_val = vsubq_s32(abs_val, sign);

        int32x4_t pos_part = vbslq_s32(mask_pos, vec, zero);

        int32x4_t neg_part = vbslq_s32(mask_neg, abs_val, zero);

        int32x4_t contrib = vorrq_s32(pos_part, neg_part);

        acc = vaddq_s32(acc, contrib);
    }

    int32_t temp[4];
    vst1q_s32(temp, acc);

    sum += temp[0];
    sum += temp[1];
    sum += temp[2];
    sum += temp[3];

    for (; i < n; ++i) {
        int32_t val = data[i];

        if (val > 0) {
            sum += val;
        } else if (val < 0) {
            sum -= val;
        }
    }

    return sum;
}

std::vector<size_t> generate_sizes() {
    std::vector<size_t> sizes;

    for (int exp = 1; exp <= 7; ++exp) {
        size_t base = pow(10, exp);

        for (int i = 1; i <= 10; ++i) {
            sizes.push_back(base * i);
        }
    }

    return sizes;
}

struct Result {
    double n;
    double scalar_time;
    double neon_time;
};

std::vector<Result> benchmark() {
    std::vector<Result> results;
    auto sizes = generate_sizes();

    std::mt19937 gen(42);
    std::uniform_int_distribution<int32_t> dist(-1000, 1000);

    for (size_t n : sizes) {
        std::vector<int32_t> data(n);
        for (size_t i = 0; i < n; ++i)
            data[i] = dist(gen);

        auto t1 = std::chrono::high_resolution_clock::now();
        process_array_scalar(data.data(), n);
        auto t2 = std::chrono::high_resolution_clock::now();

        auto t3 = std::chrono::high_resolution_clock::now();
        process_array_neon(data.data(), n);
        auto t4 = std::chrono::high_resolution_clock::now();

        double scalar_sec = std::chrono::duration<double>(t2 - t1).count();
        double neon_sec   = std::chrono::duration<double>(t4 - t3).count();

        results.push_back({(double)n, scalar_sec, neon_sec});
    }

    return results;
}

int main() {
    alignas(16) int32_t* data = new int32_t[N];

    std::mt19937 gen(42);
    std::uniform_int_distribution<int32_t> dist(-1000, 1000);

    for (size_t i = 0; i < N; ++i) {
        data[i] = dist(gen);
    }

    auto start_scalar = std::chrono::high_resolution_clock::now();
    int64_t scalar_result = process_array_scalar(data, N);
    auto end_scalar = std::chrono::high_resolution_clock::now();

    auto start_neon = std::chrono::high_resolution_clock::now();
    int64_t neon_result = process_array_neon(data, N);
    auto end_neon = std::chrono::high_resolution_clock::now();

    auto scalar_time = std::chrono::duration_cast<std::chrono::microseconds>(
        end_scalar - start_scalar
    ).count();

    auto neon_time = std::chrono::duration_cast<std::chrono::microseconds>(
        end_neon - start_neon
    ).count();

    std::cout << "Scalar result: " << scalar_result << std::endl;
    std::cout << "NEON result:   " << neon_result << std::endl;

    std::cout << "Scalar time: " << scalar_time << " us" << std::endl;
    std::cout << "NEON time:   " << neon_time << " us" << std::endl;

    if (neon_time > 0) {
        std::cout << "Speedup: " 
                  << static_cast<double>(scalar_time) / neon_time 
                  << "x" << std::endl;
    }

    if (scalar_result == neon_result) {
        std::cout << "Results are equal" << std::endl;
    } else {
        std::cout << "Error: results are different" << std::endl;
    }

    delete[] data;

    return 0;
}
