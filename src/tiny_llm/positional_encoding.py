import mlx.core as mx


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        assert dims % 2==0

        self.dims=dims
        self.seq_len=seq_len
        self.base=base
        self.traditional=traditional
        self.half_dims=dims//2

        dim_idx=mx.arange(0,self.half_dims,dtype=mx.float32)
        freqs=mx.power(base,-dim_idx/self.half_dims)

        position=mx.arange(seq_len,dtype=mx.float32)
        angles=mx.outer(position,freqs)

        self.cos_freqs=mx.cos(angles)
        self.sin_freqs=mx.sin(angles)

    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
        N,L,H,D=x.shape
        dtype=x.dtype

        if offset is None:
            offset=slice(0,L)

        cos=self.cos_freqs[offset].reshape(1,L,1,self.half_dims)
        sin=self.sin_freqs[offset].reshape(1,L,1,self.half_dims)

        if not self.traditional:
            x1=x[...,:self.half_dims]
            x2=x[...,self.half_dims:self.dims]

            out1=x1*cos-x2*sin
            out2=x1*sin+x2*cos

            out=mx.concat([out1,out2],axis=-1)
            return out.astype(dtype)
        
        x=x.reshape(N,L,H,self.half_dims,2)
        x_even=x[...,0]
        x_odd=x[...,1]

        out_even=x_even*cos-x_odd*sin
        out_odd=x_even*sin+x_odd*cos

        out=mx.stack([out_even,out_odd],axis=-1)
        return out.reshape(N,L,H,D).astype(dtype)
