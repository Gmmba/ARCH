#include <arm_neon.h>
#include <cstdint>
#include <cstddef>

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
        int32x4_t mask_pos = vcgtq_s32(vec, vdupq_n_s32(0));
        int32x4_t mask_neg = vcltq_s32(vec, vdupq_n_s32(0));
        int32x4_t sign = vshrq_n_s32(vec, 31);
        int32x4_t abs_val = veorq_s32(vec, sign);
        abs_val = vsubq_s32(abs_val, sign);
        int32x4_t pos_part = vandq_s32(vec, mask_pos);
        int32x4_t neg_part = vandq_s32(abs_val, mask_neg);
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

int main() {
    alignas(16) static int32_t data[1024];
    for (int i = 0; i < 1024; ++i) {
        data[i] = i - 512;
    }
    
    int64_t scalar_result = process_array_scalar(data, 1024);
    int64_t neon_result = process_array_neon(data, 1024);
    
    return (scalar_result == neon_result) ? 0 : 1;
}