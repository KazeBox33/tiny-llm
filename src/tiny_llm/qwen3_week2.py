import mlx.core as mx
from .basics import silu,linear
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear, QuantizedWeights
from .kv_cache import TinyKvCache


class Qwen3MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        wq: QuantizedWeights,
        wk: QuantizedWeights,
        wv: QuantizedWeights,
        wo: QuantizedWeights,
        q_norm: mx.array,
        k_norm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        rms_norm_eps: float = 1e-5,
        use_flash_attention: bool = False,
    ):
        self.hidden_size=hidden_size
        self.num_heads=num_heads
        self.num_kv_heads=num_kv_heads
        self.head_dim=head_dim
        self.scale=mx.rsqrt(head_dim)

        self.wq=dequantize_linear(wq)
        self.wk=dequantize_linear(wk)
        self.wv=dequantize_linear(wv)
        self.wo=dequantize_linear(wo)

        self.q_norm=RMSNorm(head_dim,q_norm,eps=rms_norm_eps)
        self.k_norm=RMSNorm(head_dim,k_norm,eps=rms_norm_eps)

        self.rope=RoPE(head_dim,max_seq_len,theta,traditional=False)
        self.use_flash_attention=use_flash_attention

    def __call__(
        self,
        x: mx.array,
        offsets: list[int],
        cache: TinyKvCache,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        B,L,_=x.shape

        q=linear(x,self.wq).reshape(B,L,self.num_heads,self.head_dim)
        k=linear(x,self.wk).reshape(B,L,self.num_kv_heads,self.head_dim)
        v=linear(x,self.wv).reshape(B,L,self.num_kv_heads,self.head_dim)

        q=self.q_norm(q)
        k=self.k_norm(k)

        if isinstance(offsets,int): #单条输入
            offsets=slice(offsets,offsets+L)
        else:   #batch里多条请求
            offsets=[slice(i,i+L) for i in offsets]

        q=self.rope(q,offsets)
        k=self.rope(k,offsets)

        q=q.transpose(0,2,1,3)
        k=k.transpose(0,2,1,3)
        v=v.transpose(0,2,1,3)

        k,v,_,mask=cache.update_and_fetch(k,v,mask_length=L,mask=mask)

        out=scaled_dot_product_attention_grouped(
            q.astype(mx.float32),
            k.astype(mx.float32),
            v.astype(mx.float32),
            scale=self.scale,
            mask=mask,
        ).astype(x.dtype)

        out=out.transpose(0,2,1,3).reshape(B,L,self.num_heads*self.head_dim)
        return linear(out,self.wo)


class Qwen3MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: QuantizedWeights,
        w_up: QuantizedWeights,
        w_down: QuantizedWeights,
    ):
        self.dim=dim
        self.hidden_dim=hidden_dim
        self.w_gate=dequantize_linear(w_gate)
        self.w_up=dequantize_linear(w_up)
        self.w_down=dequantize_linear(w_down)

    def __call__(self, x: mx.array) -> mx.array:
        gate=silu(linear(x,self.w_gate))
        up=linear(x,self.w_up)
        return linear(gate*up,self.w_down)


class Qwen3TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: QuantizedWeights,
        wk: QuantizedWeights,
        wv: QuantizedWeights,
        wo: QuantizedWeights,
        q_norm: mx.array,
        k_norm: mx.array,
        w_gate: QuantizedWeights,
        w_up: QuantizedWeights,
        w_down: QuantizedWeights,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        use_flash_attention: bool = False,
    ):
        self.num_attention_heads=num_attention_heads
        self.hidden_size=hidden_size

        self.input_layernorm=RMSNorm(
            hidden_size,
            w_input_layernorm,
            eps=rms_norm_eps,
        )

        self.post_attention_layernorm=RMSNorm(
            hidden_size,
            w_post_attention_layernorm,
            eps=rms_norm_eps
        )

        self.self_attn=Qwen3MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            wq=wq,
            wk=wk,
            wv=wv,
            wo=wo,
            q_norm=q_norm,
            k_norm=k_norm,
            max_seq_len=max_seq_len,
            theta=theta,
            rms_norm_eps=rms_norm_eps,
            use_flash_attention=use_flash_attention,
        )

        self.mlp = Qwen3MLP(
            dim=hidden_size,
            hidden_dim=intermediate_size,
            w_gate=w_gate,
            w_up=w_up,
            w_down=w_down,
        )

    def __call__(
        self,
        x: mx.array,
        offset: int,
        cache: TinyKvCache,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        r=self.self_attn(
            self.input_layernorm(x),
            offset,
            cache,
            mask=mask
        )
        h=x+r

        r=self.mlp(self.post_attention_layernorm(h))
        out=h+r

        return out


class Qwen3ModelWeek2:
    def __init__(
        self,
        mlx_model: Any,
        enable_flash_attn: bool = False,
    ):
        self.num_hidden_layers = mlx_model.args.num_hidden_layers
        pass

    def __call__(
        self,
        inputs: mx.array,
        offset: int,
        cache: list[TinyKvCache],
    ) -> mx.array:
        pass
