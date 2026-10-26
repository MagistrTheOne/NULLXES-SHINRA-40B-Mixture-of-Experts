"""Logical tensor partitions, real TP linear algebra, summed DP gradients."""
from dataclasses import dataclass
import torch
import torch.distributed as dist
from torch.nn import functional as F
from .communication import Group


@dataclass(frozen=True)
class TensorPartition:
    logical_shape: tuple[int, int]
    axis: int
    offsets: tuple[int, ...]

    def validate(self, group: Group):
        if self.axis not in (0,1) or len(self.logical_shape) != 2 or any(type(n) is not int or n <= 0 for n in self.logical_shape):
            raise ValueError("invalid logical matrix partition")
        if len(self.offsets) != group.size + 1 or self.offsets[0] != 0 or self.offsets[-1] != self.logical_shape[self.axis]:
            raise ValueError("partition must cover entire logical axis")
        if any(type(n) is not int for n in self.offsets) or any(a >= b for a,b in zip(self.offsets,self.offsets[1:])):
            raise ValueError("partition offsets must strictly increase")


class _Copy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, group):
        ctx.group = group
        return value.clone()

    @staticmethod
    def backward(ctx, gradient):
        result = gradient.contiguous().clone()
        dist.all_reduce(result, op=dist.ReduceOp.SUM, group=ctx.group.process_group)
        return result, None


class _Reduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, group):
        result = value.contiguous().clone()
        dist.all_reduce(result, op=dist.ReduceOp.SUM, group=group.process_group)
        return result

    @staticmethod
    def backward(ctx, gradient):
        # Replicated output represents one logical loss, not independent losses.
        return gradient, None


def partitioned_linear(value, weight, partition: TensorPartition, group: Group):
    partition.validate(group)
    group.validate_device(value)
    extent = partition.offsets[group.rank+1] - partition.offsets[group.rank]
    expected = list(partition.logical_shape)
    expected[partition.axis] = extent
    if tuple(weight.shape) != tuple(expected):
        raise ValueError("local weight does not match logical tensor partition")
    if partition.axis == 0:
        return F.linear(_Copy.apply(value, group), weight)
    if value.shape[-1] != extent:
        raise ValueError("row-parallel input must already be partitioned")
    return _Reduce.apply(F.linear(value, weight), group)


def synchronize_gradients(parameters: dict, accumulated: dict, group: Group) -> dict:
    """Sum DP gradients in canonical name order; explicit zero for unused experts.

    All participants must own the same full parameter inventory. This is DP, not
    a fallback for TP/EP shards. Global token normalization happens afterwards.
    """
    if not set(accumulated).issubset(parameters):
        raise ValueError("unknown accumulated gradient")
    result = {}
    for name in sorted(parameters):
        parameter = parameters[name]
        value = accumulated.get(name)
        if value is None:
            value = torch.zeros_like(parameter, dtype=torch.float32)
        if value.shape != parameter.shape or value.dtype != torch.float32:
            raise ValueError("gradient partition/precision mismatch")
        result[name] = group.sum(value)
    return result
