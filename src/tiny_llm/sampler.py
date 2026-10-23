import mlx.core as mx
import copy


def make_sampler(temp: float, top_p: float | None, top_k: int | None):
    def sample(logprobs: mx.array):
        if temp == 0:
            return mx.argmax(logprobs, axis=-1)
        
        logprobs=copy.copy(logprobs)

        if top_k is not None and top_k>0:
            mask_elements=mx.argpartition(-logprobs,kth=top_k-1,axis=-1)[:,top_k:] # 部分排序 第top_k-1小的在左边 返回下标
            logprobs[:,mask_elements]=-mx.inf
        if top_p is not None and 0 < top_p < 1:
            sorted_idx=mx.argsort(-logprobs,axis=-1) #从大到小排序，因为是负数就是从小到大排序
            sorted_logprobs=logprobs[:,sorted_idx]

            sorted_probs = mx.exp(sorted_logprobs)
            cumsum = mx.cumsum(sorted_probs, axis=-1)
            # 用当前候选之前的累计概率筛选，保留跨过阈值的候选。
            mask_elements = cumsum - sorted_probs < top_p

            logprobs[:,sorted_idx]=mx.where( #为True 就保留原来的值  ,为False就变为负无穷大
                mask_elements,
                sorted_logprobs,
                -mx.inf,
            )
        logprobs=logprobs/temp #温度  越小差距越大 越大差距越小
        return mx.random.categorical(logprobs,axis=-1)
    
    return sample
