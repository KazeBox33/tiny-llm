import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper
from .qwen3_week1 import Qwen3ModelWeek1
from .qwen3_week2 import Qwen3ModelWeek2
from typing import Callable
from .kv_cache import TinyKvFullCache

def simple_generate(
    model: Qwen3ModelWeek1,
    tokenizer: TokenizerWrapper,
    prompt: str,
    sampler: Callable[[mx.array], mx.array] | None,
) -> str:
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

    detokenizer=tokenizer.detokenizer
    detokenizer.reset()

    while True:
        token=_step(model,tokens)
        mx.eval(token) #这里是MLX强制把这个token算出来 , 否则后面的token.item()拿不到数值

        tokens=mx.concat([tokens,token])

        if token.item()==tokenizer.eos_token_id:
            break

        detokenizer.add_token(token.item())
        print(detokenizer.last_segment,end="",flush=True)

def simple_generate_with_kv_cache(
    model: Qwen3ModelWeek2, tokenizer: TokenizerWrapper, prompt: str
) -> str:
    kv_cache=[TinyKvFullCache() for _ in range(model.num_hidden_layers)] #不同的层数对应不同的kv_cache

    def _step(model, y, offset, kv_cache):
        logits=model(y[None],offset,kv_cache)
        logits=logits[:,-1,:]

        logprobs=logits-mx.logsumexp(logits,keepdims=True)
        y=mx.argmax(logprobs,axis=-1)
        return y

    tokens=mx.array(tokenizer.encode(prompt,add_special_tokens=False))

    detokenizer=tokenizer.detokenizer
    detokenizer.reset()

    offset=0

    while True:  # 这里最大的改动就是 1. 加入并随着步骤更新tokens ， 2 再把下一轮输入改成刚生成的单个token
        token=_step(model,tokens,offset,kv_cache)
        mx.eval(token)

        if token.item()==tokenizer.eos_token_id:
            break

        detokenizer.add_token(token.item())
        print(detokenizer.last_segment,end="",flush=True)

        offset+=tokens.size
        tokens=token

def speculative_generate(
    draft_model: Qwen3ModelWeek2,
    model: Qwen3ModelWeek2,
    draft_tokenizer: TokenizerWrapper,
    tokenizer: TokenizerWrapper,
    prompt: str,
) -> str:
    pass
