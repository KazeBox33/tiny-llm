#include "tiny_llm_ext.h"

#include <stdexcept>

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
) {
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

void QuantizedMatmul::eval_cpu(
    const std::vector<mx::array> &inputs,
    std::vector<mx::array> &outputs
) {
    throw std::runtime_error("QuantizedMatmul CPU implementation is not implemented yet");
}

void QuantizedMatmul::eval_gpu(
    const std::vector<mx::array> &inputs,
    std::vector<mx::array> &outputs
) {
    throw std::runtime_error("QuantizedMatmul GPU implementation is not implemented yet");
}

bool QuantizedMatmul::is_equivalent(const mx::Primitive &other) const {
    const auto &primitive = static_cast<const QuantizedMatmul &>(other);
    return group_size_ == primitive.group_size_ &&
           bits_ == primitive.bits_ &&
           transpose_b_ == primitive.transpose_b_;
}

}  // namespace tiny_llm_ext
