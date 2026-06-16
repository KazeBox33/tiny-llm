#pragma once

#include <mlx/array.h>
#include <mlx/stream.h>
#include <vector>
#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mx = mlx::core;

namespace tiny_llm_ext {

void load_library(mx::Device d, const char *path);

mx::array quantized_matmul(
    const mx::array &scales,
    const mx::array &biases,
    int group_size,
    int bits,
    const mx::array &a,
    const mx::array &b,
    bool transpose_b=false,
    mx::StreamOrDevice s={} 
);

class QuantizedMatmul : public mx::Primitive{ //这是一个MLX自定义的底层算子
public: 
    explicit QuantizedMatmul(mx::Stream stream, int group_size , int bits, bool transpose_b):mx::Primitive(stream),group_size_(group_size),
    bits_(bits),transpose_b_(transpose_b){}

    void eval_cpu(const std::vector<mx::array> & inputs,
        std::vector<mx::array>& outputs) override;
    void eval_gpu(const std::vector<mx::array> & inputs,
        std::vector<mx::array>& outputs) override;

    const char* name() const override{return "QuantizedMatmul";}

    bool is_equivalent(const mx::Primitive &other) const override;

private:
    int group_size_;
    int bits_;
    bool transpose_b_;
};

}  // namespace tiny_llm_ext
