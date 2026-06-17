#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"

template <typename T>
[[kernel]] void quantized_matmul_w4a16_g128(
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
                sum += static_cast<float>(a[a_loc + pack_idx]) * w;
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
