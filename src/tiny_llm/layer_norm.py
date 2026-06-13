import mlx.core as mx


class RMSNorm:
    def __init__(self, dim: int, weight: mx.array, eps: float = 1e-5):
        self.dim=dim
        self.weight=weight
        self.eps=eps
        
    def __call__(self, x: mx.array) -> mx.array:
        dtype=x.dtype
        x_float=x.astype(mx.float32)

        variance=mx.mean(x_float*x_float,axis=-1,keepdims=True)
        x_norm=x_float*mx.rsqrt(variance+self.eps)

        return x_norm.astype(dtype)*self.weight.astype(dtype)
