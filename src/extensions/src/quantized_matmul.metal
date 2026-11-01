#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"

template <typename T>
[[kernel]] void quantized_matmul_w4a16_g128(  // output.shape=[768,768]
    device const T* scales [[buffer(0)]],
    device const T* biases [[buffer(1)]],
    device const T* a [[buffer(2)]],
    device const uint32_t* b [[buffer(3)]],
    device T* out [[buffer(4)]],
    constant const int& M [[buffer(5)]],
    constant const int& N [[buffer(6)]],
    constant const int& K [[buffer(7)]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 thread_id [[thread_position_in_threadgroup]],
    uint3 threads_per_threadgroup [[threads_per_threadgroup]]) {
    constexpr int bits = 4;
    constexpr int group_size = 128;
    constexpr int packs_per_item = 32 / bits;
    constexpr uint32_t item_mask = (1 << bits) - 1;

    const int i = group_id.x * threads_per_threadgroup.x + thread_id.x;
    const int k = group_id.y * threads_per_threadgroup.y + thread_id.y;

    if (i >= M || k >= K) {
        return;
    }

    const int groups_per_row = N / group_size;
    int a_loc = i * N;
    int b_loc = k * (N / packs_per_item);
    int scale_bias_loc = k * groups_per_row;

    float sum = 0.0f;
    for (int group_idx = 0; group_idx < groups_per_row; group_idx++) {
        const float scale = static_cast<float>(scales[scale_bias_loc]);
        const float bias = static_cast<float>(biases[scale_bias_loc]);

        for (int item_idx = 0; item_idx < group_size; item_idx += packs_per_item) {
            uint32_t packed = b[b_loc];

            for (int pack_idx = 0; pack_idx < packs_per_item; pack_idx++) {
                uint32_t q = (packed >> (pack_idx * bits)) & item_mask;
                float w = static_cast<float>(q) * scale + bias;
                sum += static_cast<float>(a[a_loc + pack_idx]) * w; //flaot32
            }

            a_loc += packs_per_item;
            b_loc += 1;
        }

        scale_bias_loc += 1;
    }

    out[i * K + k] = static_cast<T>(sum);
}

instantiate_kernel("quantized_matmul_w4a16_g128_f16", quantized_matmul_w4a16_g128, half);
instantiate_kernel("quantized_matmul_w4a16_g128_bf16", quantized_matmul_w4a16_g128, bfloat16_t);

template <typename T>
[[kernel]] void quantized_matvec_x2_w4a16_g128(
    device const T* scales [[buffer(0)]],
    device const T* biases [[buffer(1)]],
    device const T* a [[buffer(2)]],
    device const uint32_t* b [[buffer(3)]],
    device T* out [[buffer(4)]],
    constant const int& M [[buffer(5)]],
    constant const int& N [[buffer(6)]],
    constant const int& K [[buffer(7)]],
    uint3 group_id [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simd_id [[simdgroup_index_in_threadgroup]]) {
    const int i = group_id.y;
    const int k0 = group_id.x * 4 + simd_id * 2;
    const int k1 = k0 + 1;
    const int packs_per_row = N / 8;
    const int groups_per_row = N / 128;
    float sum0 = 0.0f;
    float sum1 = 0.0f;

    // Lanes stride over packed words; one activation load serves both outputs.
    for (int p = lane; p < packs_per_row; p += 32) {
        const int g = p / 16;
        const uint32_t packed0 = k0 < K ? b[k0 * packs_per_row + p] : 0;
        const uint32_t packed1 = k1 < K ? b[k1 * packs_per_row + p] : 0;
        const float scale0 = k0 < K ? static_cast<float>(scales[k0 * groups_per_row + g]) : 0.0f;
        const float scale1 = k1 < K ? static_cast<float>(scales[k1 * groups_per_row + g]) : 0.0f;
        const float bias0 = k0 < K ? static_cast<float>(biases[k0 * groups_per_row + g]) : 0.0f;
        const float bias1 = k1 < K ? static_cast<float>(biases[k1 * groups_per_row + g]) : 0.0f;
        for (int j = 0; j < 8; j++) {
            const float x = static_cast<float>(a[i * N + p * 8 + j]);
            const float q0 = static_cast<float>((packed0 >> (j * 4)) & 0xF);
            const float q1 = static_cast<float>((packed1 >> (j * 4)) & 0xF);
            sum0 += x * (q0 * scale0 + bias0);
            sum1 += x * (q1 * scale1 + bias1);
        }
    }

    // Even lanes without a packed word must participate in both reductions.
    sum0 = simd_sum(sum0);
    sum1 = simd_sum(sum1);
    if (lane == 0 && i < M) {
        if (k0 < K) out[i * K + k0] = static_cast<T>(sum0);
        if (k1 < K) out[i * K + k1] = static_cast<T>(sum1);
    }
}

instantiate_kernel("quantized_matvec_x2_w4a16_g128_f16", quantized_matvec_x2_w4a16_g128, half);
instantiate_kernel("quantized_matvec_x2_w4a16_g128_bf16", quantized_matvec_x2_w4a16_g128, bfloat16_t);
