#include "tiny_llm_ext.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "mlx/backend/cpu/encoder.h"
#include "mlx/utils.h"

namespace tiny_llm_ext {

mx::array flash_attention(
    const mx::array &q,
    const mx::array &k,
    const mx::array &v,
    const mx::array &mask,
    float scale,
    bool is_causal,
    int num_kv_heads,
    int num_heads,
    mx::StreamOrDevice s
) {
    if (q.dtype() != mx::float32 || k.dtype() != mx::float32 ||
        v.dtype() != mx::float32 || mask.dtype() != mx::float32) {
        throw std::runtime_error("flash_attention: all input arrays must be float32");
    }

    if (q.shape().size() != 3 || k.shape().size() != 3 || v.shape().size() != 3) {
        throw std::runtime_error("flash_attention: q, k, v must be 3D");
    }

    if (mask.shape().size() != 3) {
        throw std::runtime_error("flash_attention: mask must be 3D");
    }

    if (num_heads % num_kv_heads != 0) {
        throw std::runtime_error("flash_attention: num_heads must be divisible by num_kv_heads");
    }

    const auto &q_shape = q.shape();
    const auto &k_shape = k.shape();
    const auto &v_shape = v.shape();
    const auto &mask_shape = mask.shape();

    if (q_shape[0] % num_heads != 0) {
        throw std::runtime_error("flash_attention: q batch-head dim mismatch");
    }

    if (k_shape[0] % num_kv_heads != 0 || v_shape[0] % num_kv_heads != 0) {
        throw std::runtime_error("flash_attention: k/v batch-head dim mismatch");
    }

    if (q_shape[0] / num_heads != k_shape[0] / num_kv_heads ||
        q_shape[0] / num_heads != v_shape[0] / num_kv_heads) {
        throw std::runtime_error("flash_attention: q and kv batch size mismatch");
    }

    if (q_shape[2] != k_shape[2] || q_shape[2] != v_shape[2]) {
        throw std::runtime_error("flash_attention: head_dim mismatch");
    }

    if (k_shape[1] != v_shape[1]) {
        throw std::runtime_error("flash_attention: key/value sequence length mismatch");
    }

    if (mask_shape[0] != q_shape[0] || mask_shape[1] != q_shape[1] || mask_shape[2] != k_shape[1]) {
        throw std::runtime_error("flash_attention: mask must have shape [N, L, S]");
    }

    return mx::array(
        q.shape(),
        mx::float32,
        std::make_shared<FlashAttention>(
            mx::to_stream(s),
            scale,
            is_causal,
            num_kv_heads,
            num_heads
        ),
        {q, k, v, mask}
    );
}

void FlashAttention::eval_cpu(
    const std::vector<mx::array> &inputs,
    std::vector<mx::array> &outputs
) {
    const auto &q = inputs[0];
    const auto &k = inputs[1];
    const auto &v = inputs[2];
    const auto &mask = inputs[3];
    auto &out = outputs[0];

    if (out.dtype() != mx::float32) {
        throw std::runtime_error("flash_attention: output dtype must be float32");
    }
    if (!q.flags().row_contiguous || !k.flags().row_contiguous ||
        !v.flags().row_contiguous || !mask.flags().row_contiguous) {
        throw std::runtime_error("flash_attention: inputs must be contiguous");
    }

    out.set_data(mx::allocator::malloc(out.nbytes()));

    auto &encoder = mx::cpu::get_command_encoder(stream());
    encoder.set_input_array(q);
    encoder.set_input_array(k);
    encoder.set_input_array(v);
    encoder.set_input_array(mask);
    encoder.set_output_array(out);

    encoder.dispatch([
        out_ptr = out.data<float>(),
        q = mx::array::unsafe_weak_copy(q),
        k = mx::array::unsafe_weak_copy(k),
        v = mx::array::unsafe_weak_copy(v),
        mask = mx::array::unsafe_weak_copy(mask),
        scale = scale_,
        is_causal = is_causal_,
        num_heads = num_heads_,
        num_kv_heads = num_kv_heads_
    ]() {
        const int64_t N = q.shape()[0];
        const int64_t L = q.shape()[1];
        const int64_t S = k.shape()[1];
        const int64_t E = q.shape()[2];

        constexpr int64_t Br = 32;
        constexpr int64_t Bc = 32;
        const int64_t Tr = (L + Br - 1) / Br;
        const int64_t Tc = (S + Bc - 1) / Bc;
        const int64_t q_kv_heads_ratio = num_heads / num_kv_heads;
        const int64_t causal_offset = S - L;

        const int64_t q_head_size = L * E;
        const int64_t kv_head_size = S * E;
        const float *q_ptr = q.data<float>();
        const float *k_ptr = k.data<float>();
        const float *v_ptr = v.data<float>();
        const float *mask_ptr = mask.data<float>();

        for (int64_t n = 0; n < N; n++) {
            const float *q_head = q_ptr + n * q_head_size;
            const int64_t kv_n = n / q_kv_heads_ratio;
            const float *k_head = k_ptr + kv_n * kv_head_size;
            const float *v_head = v_ptr + kv_n * kv_head_size;
            const float *mask_head = mask_ptr + n * L * S;
            float *out_head = out_ptr + n * q_head_size;

            for (int64_t i = 0; i < Tr; i++) {
                const int64_t q_start = i * Br;
                const int64_t br = std::min(Br, L - q_start);

                std::vector<float> o_i(br * E, 0.0f);
                std::vector<float> l_i(br, 0.0f);
                std::vector<float> m_i(br, -std::numeric_limits<float>::infinity());

                for (int64_t j = 0; j < Tc; j++) {
                    const int64_t k_start = j * Bc;
                    const int64_t bc = std::min(Bc, S - k_start);

                    if (is_causal && k_start > q_start + br - 1 + causal_offset) {
                        continue;
                    }

                    const bool block_all_valid =
                        is_causal && (k_start + bc - 1 <= q_start + causal_offset);

                    std::vector<float> scores(br * bc, 0.0f);
                    for (int64_t r = 0; r < br; r++) {
                        const int64_t q_pos = q_start + r;
                        for (int64_t c = 0; c < bc; c++) {
                            const int64_t k_pos = k_start + c;
                            float score = 0.0f;
                            for (int64_t e = 0; e < E; e++) {
                                score += q_head[q_pos * E + e] * k_head[k_pos * E + e];
                            }
                            score *= scale;
                            if (!block_all_valid) {
                                score += mask_head[q_pos * S + k_pos];
                            }
                            scores[r * bc + c] = score;
                        }
                    }

                    std::vector<float> probs(br * bc, 0.0f);
                    std::vector<float> old_m_shift(br, 0.0f);
                    for (int64_t r = 0; r < br; r++) {
                        float row_max = -std::numeric_limits<float>::infinity();
                        for (int64_t c = 0; c < bc; c++) {
                            row_max = std::max(row_max, scores[r * bc + c]);
                        }

                        const float new_m = std::max(m_i[r], row_max);
                        old_m_shift[r] = m_i[r] - new_m;
                        m_i[r] = new_m;

                        float row_sum = 0.0f;
                        for (int64_t c = 0; c < bc; c++) {
                            const float p = std::exp(scores[r * bc + c] - new_m);
                            probs[r * bc + c] = p;
                            row_sum += p;
                        }

                        l_i[r] = std::exp(old_m_shift[r]) * l_i[r] + row_sum;
                    }

                    for (int64_t r = 0; r < br; r++) {
                        const float previous_output_scale = std::exp(old_m_shift[r]);
                        for (int64_t e = 0; e < E; e++) {
                            float pv = 0.0f;
                            for (int64_t c = 0; c < bc; c++) {
                                pv += probs[r * bc + c] * v_head[(k_start + c) * E + e];
                            }
                            o_i[r * E + e] = previous_output_scale * o_i[r * E + e] + pv;
                        }
                    }
                }

                for (int64_t r = 0; r < br; r++) {
                    const int64_t q_pos = q_start + r;
                    for (int64_t e = 0; e < E; e++) {
                        out_head[q_pos * E + e] = o_i[r * E + e] / l_i[r];
                    }
                }
            }
        }
    });
}

void FlashAttention::eval_gpu(
    const std::vector<mx::array> &inputs,
    std::vector<mx::array> &outputs
) {
    throw std::runtime_error("FlashAttention GPU implementation is not implemented yet");
}

bool FlashAttention::is_equivalent(const mx::Primitive &other) const {
    const auto &primitive = static_cast<const FlashAttention &>(other);
    return scale_ == primitive.scale_ &&
           is_causal_ == primitive.is_causal_ &&
           num_kv_heads_ == primitive.num_kv_heads_ &&
           num_heads_ == primitive.num_heads_;
}

}  // namespace tiny_llm_ext
