import mlx.core as mx

from .basics import softmax, linear
import math 

def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    if scale is None: #scale 是缩放系数
        scale =1.0/math.sqrt(query.shape[-1])

    scores=query@key.swapaxes(-1,-2)* scale

    if mask is not None:
        scores=scores+mask

    attention_weights=softmax(scores,axis=-1)
    return attention_weights@value

class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array, #wq 这里的 w 表示的是权重
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        self.hidden_size=hidden_size
        self.num_heads=num_heads
        self.head_dim=hidden_size//num_heads

        self.wq=wq
        self.wk=wk
        self.wv=wv
        self.wo=wo


    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        q=linear(query,self.wq) #没办法，推理的linear 并不带权重
        k=linear(key,self.wk)
        v=linear(value,self.wv)

        *batch_dims,seq_len,_=q.shape

        q=q.reshape(*batch_dims,seq_len,self.num_heads,self.head_dim)
        k=k.reshape(*batch_dims,seq_len,self.num_heads,self.head_dim)
        v=v.reshape(*batch_dims,seq_len,self.num_heads,self.head_dim)

        q=q.swapaxes(-3,-2)
        k=k.swapaxes(-3,-2)
        v=v.swapaxes(-3,-2)
        
        attention_output=scaled_dot_product_attention_simple(q,k,v,mask=mask)
        attention_output=attention_output.swapaxes(-3,-2)
        attention_output=attention_output.reshape(*batch_dims,seq_len,self.hidden_size)

        return linear(attention_output,self.wo)


def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    pass


def scaled_dot_product_attention_grouped(  # k v 复用头
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    if scale is None:
        scale=1.0/math.sqrt(query.shape[-1])

    expected_shape=query.shape

    *batch_dims,H_q,L,D=query.shape
    H,S,_=key.shape[-3:]
    n_repeats=H_q//H

    query=query.reshape(*batch_dims,H,n_repeats,L,D)
    key=key.reshape(*batch_dims,H,1,S,D)
    value=value.reshape(*batch_dims,H,1,S,D)

    scores=query@key.swapaxes(-1,-2) *scale

    if mask is not None:
        mask=mx.broadcast_to(mask,(*batch_dims,H_q,L,S))
        mask=mask.reshape(*batch_dims,H,n_repeats,L,S)
        scores=scores+mask

    attention_weights=softmax(scores,axis=-1)
    output=attention_weights@value
    return output.reshape(expected_shape)


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    pass
