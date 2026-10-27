import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper
from .qwen3_week1 import Qwen3ModelWeek1
from .qwen3_week2 import Qwen3ModelWeek2
from typing import Callable

def simple_generate(
    model: Qwen3ModelWeek1,
    tokenizer: TokenizerWrapper,
    prompt: str,
    sampler: Callable[[mx.array], mx.array] | None,
    max_tokens: int = 256,
) -> None:
    if (
        not isinstance(max_tokens, int)
        or isinstance(max_tokens, bool)
        or max_tokens < 0
    ):
        raise ValueError("max_tokens must be a non-negative integer")
    if max_tokens == 0:
        return

    def _step(model, y):
        logits=model(y[None])
        logits=logits[:,-1,:]

        #转换成logprob
        logprobs=logits-mx.logsumexp(logits,keepdims=True)

        if sampler is None:
            y=mx.argmax(logprobs,axis=-1)
        else:
            y=sampler(logprobs)
        return y
    
    tokens=mx.array(tokenizer.encode(prompt,add_special_tokens=False))
    if tokens.size == 0:
        raise ValueError("prompt must encode to at least one token")

    detokenizer=tokenizer.detokenizer
    detokenizer.reset()

    for _ in range(max_tokens):
        token=_step(model,tokens)
        mx.eval(token) #这里是MLX强制把这个token算出来 , 否则后面的token.item()拿不到数值

        tokens=mx.concat([tokens,token])

        if token.item()==tokenizer.eos_token_id:
            break

        detokenizer.add_token(token.item())
        print(detokenizer.last_segment,end="",flush=True)

    detokenizer.finalize()
    print(detokenizer.last_segment, end="", flush=True)

def simple_generate_with_kv_cache(
    model: Qwen3ModelWeek2,
    tokenizer: TokenizerWrapper,
    prompt: str,
    max_tokens: int = 256,
    use_bounded_kv_capacity: bool = True,
) -> None:
    if (
        not isinstance(max_tokens, int)
        or isinstance(max_tokens, bool)
        or max_tokens < 0
    ):
        raise ValueError("max_tokens must be a non-negative integer")
    if not isinstance(use_bounded_kv_capacity, bool):
        raise ValueError("use_bounded_kv_capacity must be a bool")
    if max_tokens == 0:
        return

    def _step(model, y, offset, kv_cache):
        logits=model(y[None],offset,kv_cache)
        logits=logits[:,-1,:]

        logprobs=logits-mx.logsumexp(logits,keepdims=True)
        y=mx.argmax(logprobs,axis=-1).astype(mx.int32)
        return y

    tokens=mx.array(
        tokenizer.encode(prompt,add_special_tokens=False), dtype=mx.int32
    )
    if tokens.size == 0:
        raise ValueError("prompt must encode to at least one token")
    capacity = int(tokens.size) + max_tokens if use_bounded_kv_capacity else None
    kv_cache = model.create_kv_cache(capacity=capacity)

    detokenizer=tokenizer.detokenizer
    detokenizer.reset()

    offset=0

    for _ in range(max_tokens):
        token=_step(model,tokens,offset,kv_cache)
        mx.eval(token)

        if token.item()==tokenizer.eos_token_id:
            break

        detokenizer.add_token(token.item())
        print(detokenizer.last_segment,end="",flush=True)

        offset+=tokens.size
        tokens=token

    detokenizer.finalize()
    print(detokenizer.last_segment,end="",flush=True)

def speculative_generate(
    draft_model: Qwen3ModelWeek2,
    model: Qwen3ModelWeek2,
    draft_tokenizer: TokenizerWrapper,
    tokenizer: TokenizerWrapper,
    prompt: str,
) -> str:
    pass
