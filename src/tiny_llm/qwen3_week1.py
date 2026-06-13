import mlx.core as mx
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear


class Qwen3MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        q_norm: mx.array,
        k_norm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        rms_norm_eps: float = 1e-5,
    ):
        self.hidden_size=hidden_size
        self.num_heads=num_heads
        self.num_kv_heads=num_kv_heads
        self.head_dim=head_dim
        self.scale=mx.rsqrt(head_dim)

        self.wq=wq
        self.wk=wk
        self.wv=wv
        self.wo=wo

        self.q_norm=q_norm
        self.k_norm=k_norm
        self.rms_norm_eps=rms_norm_eps

        self.rope=RoPE(head_dim,max_seq_len,theta,traditional=False)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        B,L,_=x.shape

        q=linear(x,self.wq).reshape(B,L,self.num_heads,self.head_dim)
        k=linear(x,self.wk).reshape(B,L,self.num_kv_heads,self.head_dim)
        v=linear(x,self.wv).reshape(B,L,self.num_kv_heads,self.head_dim)

        q=mx.fast.rms_norm(q,self.q_norm,eps=self.rms_norm_eps)
        k=mx.fast.rms_norm(k,self.k_norm,eps=self.rms_norm_eps)

        q=self.rope(q,offset=slice(0,L))
        k=self.rope(k,offset=slice(0,L))

        q=q.transpose(0,2,1,3)
        k=k.transpose(0,2,1,3)
        v=v.transpose(0,2,1,3)

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
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
    ):
        self.dim=dim
        self.hidden_dim=hidden_dim
        self.w_gate=w_gate
        self.w_up=w_up
        self.w_down=w_down

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
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        q_norm: mx.array,
        k_norm: mx.array,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        self.num_attention_heads=num_attention_heads
        self.hidden_size=hidden_size

        self.input_layernorm=RMSNorm(
            hidden_size,
            w_input_layernorm,
            eps=rms_norm_eps
        )

        self.post_attention_layernorm=RMSNorm(
            hidden_size,
            w_post_attention_layernorm,
            eps=rms_norm_eps,
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
        )

        self.mlp=Qwen3MLP(
            dim=hidden_size,
            hidden_dim=intermediate_size,
            w_gate=w_gate,
            w_up=w_up,
            w_down=w_down
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        r=self.self_attn(self.input_layernorm(x),mask=mask)
        h=x+r
        
        r=self.mlp(self.post_attention_layernorm(h))
        out=h+r

        return out


class Qwen3ModelWeek1:
    def __init__(self, mlx_model: Any):
        self.num_hidden_layers=mlx_model.args.num_hidden_layers
        self.hidden_size=mlx_model.args.hidden_size
        self.vocab_size=mlx_model.args.vocab_size
        self.precision=mx.bfloat16

        self.embedding=Embedding(
            vocab_size=self.vocab_size,
            embedding_dim=self.hidden_size,
            weight=dequantize_linear(mlx_model.model.embed_tokens) #解量化
        )
        self.layers_inner = []

        for i in range(mlx_model.args.num_hidden_layers):
            layer = Qwen3TransformerBlock(
                num_attention_heads=mlx_model.args.num_attention_heads,
                num_kv_heads=mlx_model.args.num_key_value_heads,
                hidden_size=mlx_model.args.hidden_size,
                head_dim=mlx_model.args.head_dim,
                intermediate_size=mlx_model.args.intermediate_size,
                rms_norm_eps=mlx_model.args.rms_norm_eps,
                wq=dequantize_linear(mlx_model.model.layers[i].self_attn.q_proj),
                wk=dequantize_linear(mlx_model.model.layers[i].self_attn.k_proj),
                wv=dequantize_linear(mlx_model.model.layers[i].self_attn.v_proj),
                wo=dequantize_linear(mlx_model.model.layers[i].self_attn.o_proj),
                q_norm=mlx_model.model.layers[i].self_attn.q_norm.weight,
                k_norm=mlx_model.model.layers[i].self_attn.k_norm.weight,
                w_gate=dequantize_linear(mlx_model.model.layers[i].mlp.gate_proj),
                w_up=dequantize_linear(mlx_model.model.layers[i].mlp.up_proj),
                w_down=dequantize_linear(mlx_model.model.layers[i].mlp.down_proj),
                w_input_layernorm=mlx_model.model.layers[i].input_layernorm.weight,
                w_post_attention_layernorm=mlx_model.model.layers[
                    i
                ].post_attention_layernorm.weight,
                max_seq_len=mlx_model.args.max_position_embeddings,
                theta=mlx_model.args.rope_theta,
            )
            self.layers_inner.append(layer)

        self.norm = RMSNorm(
            self.hidden_size,
            weight=mlx_model.model.norm.weight,
            eps=mlx_model.args.rms_norm_eps,
        )

        if mlx_model.args.tie_word_embeddings:
            self.w_lm_head = None
        else:
            self.w_lm_head = dequantize_linear(mlx_model.lm_head)

        self.mlx_model = mlx_model
    def __call__(
        self,
        inputs: mx.array,
    ) -> mx.array:
        h=self.embedding(inputs)

        mask="causal" if inputs.shape[-1] >1 else None

        for layer in self.layers_inner:
            h=layer(h,mask=mask)

        h=self.norm(h) #在这里自己最后归一化一下

        if self.w_lm_head is not None:
            return linear(h,self.w_lm_head)
        
        return self.embedding.as_linear(h)
