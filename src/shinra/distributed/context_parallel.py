"""Exact local-query/global-key attention using differentiable KV all-gather.

This transport replicates KV, not a ring or a memory-efficiency claim. Logical
context partitioning is real; model-level CP assembly is a separate contract.
"""
import torch
import torch.distributed as dist
from shinra.model.attention import _BlockwiseGQA, require_ieee_matmul
from .communication import Group


class _Gather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, sizes, group):
        group.validate_device(value)
        if len(sizes) != group.size or any(type(n) is not int or n < 0 for n in sizes) or value.shape[0] != sizes[group.rank]:
            raise ValueError("context shard lengths mismatch")
        maximum = max(sizes)
        if maximum == 0:
            raise ValueError("global context cannot be empty")
        padded = value.new_zeros((maximum, *value.shape[1:]))
        padded[:value.shape[0]].copy_(value)
        gathered = [torch.empty_like(padded) for _ in sizes]
        dist.all_gather(gathered, padded.contiguous(), group=group.process_group)
        ctx.sizes, ctx.group = sizes, group
        return torch.cat([part[:size] for part, size in zip(gathered, sizes)], dim=0)

    @staticmethod
    def backward(ctx, gradient):
        gradient = gradient.contiguous().clone()
        dist.all_reduce(gradient, op=dist.ReduceOp.SUM, group=ctx.group.process_group)
        start = sum(ctx.sizes[:ctx.group.rank])
        return gradient[start:start+ctx.sizes[ctx.group.rank]], None, None


def context_attention(q, local_k, local_v, qpos, local_kpos, qseg, local_kseg, *,
                      sizes: tuple[int, ...], group: Group, query_chunk: int, key_chunk: int):
    if type(query_chunk) is not int or query_chunk <= 0 or type(key_chunk) is not int or key_chunk <= 0:
        raise ValueError("explicit positive attention chunk sizes required")
    if q.ndim != 3 or q.shape[1:] != (32,128) or local_k.shape[1:] != (8,128) or local_k.shape != local_v.shape:
        raise ValueError("CP GQA shape mismatch")
    if q.dtype != local_k.dtype or q.dtype != local_v.dtype or q.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("CP attention dtype mismatch")
    for metadata, length in ((qpos,q.shape[0]), (qseg,q.shape[0]), (local_kpos,local_k.shape[0]), (local_kseg,local_k.shape[0])):
        if metadata.dtype != torch.int64 or metadata.shape != (length,) or metadata.device != q.device:
            raise ValueError("CP metadata requires same-device int64 positions and segments")
    require_ieee_matmul(q.device)
    keys = _Gather.apply(local_k, sizes, group)
    values = _Gather.apply(local_v, sizes, group)
    positions = _Gather.apply(local_kpos, sizes, group)
    segments = _Gather.apply(local_kseg, sizes, group)
    # Explicit global positions determine mask; no sliding window or re-RoPE.
    return _BlockwiseGQA.apply(q, keys, values, qpos, positions, qseg, segments, query_chunk, key_chunk)
