#include "tiny_llm_ext.h"

#include <stdexcept>

#include <cstdint>

#include "mlx/backend/common/utils.h" 
#include "mlx/backend/cpu/encoder.h"
#include "mlx/utils.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#endif

namespace tiny_llm_ext {

mx::array quantized_matmul(
    const mx::array &scales,
    const mx::array &biases,
    int group_size,
    int bits,
    const mx::array &a,
    const mx::array &b,
    bool transpose_b,
    mx::StreamOrDevice s
) {  // a=x , b=w.weights  b 是 packed 4-bit 权重，物理 shape 是 [K, N/8]，逻辑上代表 [K, N]。
    if (!transpose_b) {
        throw std::runtime_error("quantized_matmul currently expects transpose_b=true");
    }

    if (bits != 4) {
        throw std::runtime_error("quantized_matmul currently only supports 4-bit weights");
    }

    if (group_size != 128) {
        throw std::runtime_error("quantized_matmul currently only supports group_size=128");
    }

    auto stream = mx::to_stream(s);

    auto out_shape = a.shape();
    out_shape[1] = b.shape()[0];

    return mx::array(
        out_shape,
        a.dtype(),
        std::make_shared<QuantizedMatmul>(stream, group_size, bits, transpose_b),
        {scales, biases, a, b}
    );
}

template <typename T>
void quantized_matmul_impl(
    const mx::array &scales,
    const mx::array &biases,
    const mx::array &a,
    const mx::array &b,
    mx::array &out,
    mx::Stream stream
) {
    out.set_data(mx::allocator::malloc(out.nbytes())); //表示输出需要多少个字节

    auto &encoder = mx::cpu::get_command_encoder(stream);
    encoder.set_input_array(scales); //放进容器里管理 array的生命周期
    encoder.set_input_array(biases);
    encoder.set_input_array(a);
    encoder.set_input_array(b);
    encoder.set_output_array(out);

    encoder.dispatch([
        out_ptr = out.data<T>(),
        out_shape = out.shape(),
        out_strides = out.strides(), //这里是步长
        scales = mx::array::unsafe_weak_copy(scales), //弱引用，不强行延长生命周期
        biases = mx::array::unsafe_weak_copy(biases),
        a = mx::array::unsafe_weak_copy(a),
        b = mx::array::unsafe_weak_copy(b)
    ]() {
        int M = a.shape()[0];
        int N = a.shape()[1];
        int K = b.shape()[0];

        constexpr int group_size = 128;
        constexpr int bits = 4;
        constexpr int packs_per_item = 32 / bits;

        int groups_per_row = N / group_size;  // 表示一行有多少组
        uint32_t item_mask = (1 << bits) - 1; //用来从packed uint32_t 里取出一个4-bit权重值

        const T *a_ptr = a.data<T>();
        const uint32_t *b_ptr = b.data<uint32_t>();
        const T *scales_ptr = scales.data<T>();
        const T *biases_ptr = biases.data<T>();
        //a [M,N] b[K,N/8] scale [K,N/128] bias [K,N/128]   out[M,K]
        for (int i = 0; i < M; i++) {
            for (int k = 0; k < K; k++) {
                float sum = 0.0f;

                for (int group_idx = 0; group_idx < groups_per_row; group_idx++) {
                    int64_t scale_loc = mx::elem_to_loc(
                        k * groups_per_row + group_idx,
                        scales.shape(),
                        scales.strides()
                    );
                    int64_t bias_loc = mx::elem_to_loc(
                        k * groups_per_row + group_idx,
                        biases.shape(),
                        biases.strides()
                    );

                    float scale = static_cast<float>(scales_ptr[scale_loc]); //取出scale 和 bias 并转float32
                    float bias = static_cast<float>(biases_ptr[bias_loc]);

                    int group_start = group_idx * group_size;

                    for (int item_idx = 0; item_idx < group_size; item_idx += packs_per_item) { //item_idx 是 0 8 16...120 每次处理8个权重
                        int n = group_start + item_idx; // n 表示没缩放前的位置

                        int64_t b_loc = mx::elem_to_loc(
                            k * (N / packs_per_item) + n / packs_per_item,
                            b.shape(),
                            b.strides()
                        );

                        uint32_t packed = b_ptr[b_loc];

                        for (int pack_idx = 0; pack_idx < packs_per_item; pack_idx++) { // uint32 里 取 8个 uint_4
                            uint32_t q = (packed >> (pack_idx * bits)) & item_mask;
                            float w = static_cast<float>(q) * scale + bias;

                            int64_t a_loc = mx::elem_to_loc(
                                i * N + n + pack_idx,
                                a.shape(),
                                a.strides()
                            );

                            sum += static_cast<float>(a_ptr[a_loc]) * w;
                        }
                    }
                }

                int64_t out_loc = mx::elem_to_loc(
                    i * K + k,
                    out_shape,
                    out_strides
                );
                out_ptr[out_loc] = static_cast<T>(sum);
            }
        }
    });
}

void QuantizedMatmul::eval_cpu( // 输入 a[M,N]  b:[K,N/8]  scale[K,N/128] biases[K,N/128]
    const std::vector<mx::array> &inputs,  // CPU版本是双重/三重 for loop
    std::vector<mx::array> &outputs
) {
        auto &scales = inputs[0];
        auto &biases = inputs[1];
        auto &a = inputs[2];
        auto &b = inputs[3];
        auto &out = outputs[0];

    if (out.dtype() == mx::float16) {
        return quantized_matmul_impl<mx::float16_t>(scales, biases, a, b, out, stream());
    }

    if (out.dtype() == mx::bfloat16) {
        return quantized_matmul_impl<mx::bfloat16_t>(scales, biases, a, b, out, stream());
    }

    throw std::runtime_error("QuantizedMatmul CPU only supports float16 and bfloat16");
}

void QuantizedMatmul::eval_gpu(
    const std::vector<mx::array> &inputs,
    std::vector<mx::array> &outputs
) {
#ifdef _METAL_
    auto &scales = inputs[0];
    auto &biases = inputs[1];
    auto &a = inputs[2];
    auto &b = inputs[3];
    auto &out = outputs[0];

    if (!a.flags().row_contiguous) {
        throw std::runtime_error("quantized_matmul: a must be contiguous");
    }
    if (!b.flags().row_contiguous) {
        throw std::runtime_error("quantized_matmul: b must be contiguous");
    }

    const int M = a.shape()[0];
    const int N = a.shape()[1];
    const int K = b.shape()[0];

    if (N % group_size_ != 0) {
        throw std::runtime_error("quantized_matmul: N must be divisible by group_size");
    }

    auto &s = stream();
    auto &d = mx::metal::device(s.device);
    out.set_data(mx::allocator::malloc(out.nbytes()));

    auto library = d.get_library("tiny_llm_ext");
    const char *kernel_name = nullptr;
    if (out.dtype() == mx::float16) { //根据dtype选择kernel
        kernel_name = "quantized_matmul_w4a16_g128_f16";
    } else if (out.dtype() == mx::bfloat16) {
        kernel_name = "quantized_matmul_w4a16_g128_bf16";
    } else {
        throw std::runtime_error("QuantizedMatmul GPU only supports float16 and bfloat16");
    }
    auto kernel = d.get_kernel(kernel_name, library);

    auto &compute_encoder = d.get_command_encoder(s.index);
    compute_encoder.set_compute_pipeline_state(kernel);

    compute_encoder.set_input_array(scales, 0);
    compute_encoder.set_input_array(biases, 1);
    compute_encoder.set_input_array(a, 2);
    compute_encoder.set_input_array(b, 3);
    compute_encoder.set_output_array(out, 4);

    compute_encoder.set_bytes(M, 5);
    compute_encoder.set_bytes(N, 6);
    compute_encoder.set_bytes(K, 7);

    size_t tgp_size = kernel->maxTotalThreadsPerThreadgroup();  //获取一个threadgroup最多能放多少线程 
    int x_size = 32;
    if (M <= 1) {
        x_size = 1;
    } else if (M <= 2) {
        x_size = 2;
    } else if (M <= 4) {
        x_size = 4;
    } else if (M <= 8) {
        x_size = 8;
    } else if (M <= 16) {
        x_size = 16;
    }
    int y_size = static_cast<int>(tgp_size) / x_size;

    MTL::Size num_threadgroups = MTL::Size(
        (M + x_size - 1) / x_size,
        (K + y_size - 1) / y_size,
        1
    );
    MTL::Size threads_per_threadgroup = MTL::Size(x_size, y_size, 1);
    compute_encoder.dispatch_threadgroups(num_threadgroups, threads_per_threadgroup);
#else
    throw std::runtime_error("QuantizedMatmul has no GPU implementation.");
#endif
}

bool QuantizedMatmul::is_equivalent(const mx::Primitive &other) const {
    const auto &primitive = static_cast<const QuantizedMatmul &>(other);
    return group_size_ == primitive.group_size_ &&
           bits_ == primitive.bits_ &&
           transpose_b_ == primitive.transpose_b_;
}

}  // namespace tiny_llm_ext
