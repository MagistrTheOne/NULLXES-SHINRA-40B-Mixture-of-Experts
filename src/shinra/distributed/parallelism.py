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
        if result.numel():
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


def synchronize_gradients(parameters: dict, accumulated: dict, group: Group, *, owned_names=frozenset()) -> dict:
    """Sum DP gradients in canonical name order; explicit zero for unused experts.

    Replicated parameters must have identical inventories across members. The
    explicit owned_names inventory excludes owner-only EP parameters from SUM.
    Global token normalization happens afterwards; TP shards are not admitted.
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
        # Owner-only experts already accumulated all input-source gradients via
        # inverse EP communication; reducing them again would double-count.
        result[name] = value if name in owned_names else group.sum(value)
    return result


class _GatherFeatures(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, group):
        group.validate_device(value)
        ctx.group, ctx.width = group, value.shape[-1]
        if value.numel() == 0:
            return value.new_empty((*value.shape[:-1], value.shape[-1]*group.size))
        parts = [torch.empty_like(value) for _ in group.ranks]
        dist.all_gather(parts, value.contiguous(), group=group.process_group)
        ctx.group, ctx.width = group, value.shape[-1]
        return torch.cat(parts, dim=-1)

    @staticmethod
    def backward(ctx, gradient):
        # Every TP member represents the same logical output/loss. Summing
        # replicated output gradients here would multiply the derivative by TP.
        start = ctx.group.rank * ctx.width
        return gradient[..., start:start+ctx.width].contiguous(), None


@dataclass(frozen=True)
class TensorExecution:
    """Weight-sharded TP with replicated full activations and full-vocab logits.

    Gathered Q/K/V retain global head ordering. This deliberately does not claim
    activation-memory efficiency or use a different SHINRA attention algorithm.
    """
    group: Group

    def __post_init__(self):
        if self.group.size not in (2, 4, 8):
            raise ValueError("TP must partition all 8 KV heads and corresponding 32 Q heads exactly")

    def partition(self, spec):
        if spec.role == "norm" or spec.role == "router":
            return None  # Explicitly replicated, including independent QK gamma.
        if len(spec.shape) != 2:
            raise ValueError("TP weight must have a two-dimensional logical shape")
        axis = 1 if spec.name == "embeddings.weight" else 0
        if spec.shape[axis] % self.group.size:
            raise ValueError("logical tensor dimension does not divide TP membership")
        extent = spec.shape[axis] // self.group.size
        return TensorPartition(spec.shape, axis, tuple(i*extent for i in range(self.group.size+1)))

    def initialize(self, spec, recipe, *, device):
        from shinra.core.initialization import initialize_parameter
        partition = self.partition(spec)
        # Canonical logical sampling is independent of slice boundaries. Only
        # the source generates the FULL logical stream, with the legacy seed
        # and FP32 sampling/cast semantics. Other members receive those bytes.
        # Full temporary allocation is intentional and is not a VRAM claim.
        with torch.no_grad():
            if self.group.rank == 0:
                full = initialize_parameter(spec, recipe, device=device).detach()
            else:
                dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[spec.dtype]
                full = torch.empty(spec.shape, dtype=dtype, device=device)
            self.group.validate_device(full)
            dist.broadcast(full, src=self.group.ranks[0], group=self.group.process_group)
            if partition is None:
                local = full
            else:
                start, stop = partition.offsets[self.group.rank:self.group.rank+2]
                local = full.narrow(partition.axis, start, stop-start).contiguous().clone()
        return torch.nn.Parameter(local)

    def linear(self, value, weight):
        # All non-embedding matrix shards own output rows. Input-gradient SUM
        # and output-feature gather have deliberately different adjoints.
        local = F.linear(_Copy.apply(value, self.group), weight)
        return _GatherFeatures.apply(local, self.group)

    def embedding(self, token_ids, weight):
        return _GatherFeatures.apply(F.embedding(token_ids, weight, padding_idx=None), self.group)

    def inventory(self, specs):
        result = {}
        for spec in specs:
            partition = self.partition(spec)
            shape = list(spec.shape)
            if partition is None:
                entry = {"logical_shape": list(spec.shape), "axis": None, "start": None, "stop": None}
            else:
                start, stop = partition.offsets[self.group.rank:self.group.rank+2]
                shape[partition.axis] = stop-start
                entry = {"logical_shape": list(spec.shape), "axis": partition.axis, "start": start, "stop": stop}
            result[spec.name] = {**entry, "local_shape": shape, "dtype": spec.dtype}
        return result


class MixedExecution:
    """Validated caller-provided groups; never creates or guesses a topology."""
    def __init__(self, layout, groups):
        from .ownership import ParallelLayout, ExpertOwnership
        from .expert_parallel import ExpertExecution
        from .context_parallel import ContextExecution
        roles = {"world", "tensor", "expert", "context", "shared_parameters", "expert_parameters", "source_replicas"}
        if not isinstance(layout, ParallelLayout) or set(groups) != roles or any(not isinstance(g, Group) for g in groups.values()):
            raise ValueError("all explicit logical process groups are required, including singleton groups")
        rank = dist.get_rank()
        for role, group in groups.items():
            if group.ranks != layout.members(rank, role):
                raise ValueError("process-group membership/order differs from logical layout: " + role)
        if len({g.backend for g in groups.values()}) != 1:
            raise ValueError("mixed backend policy is not admitted")
        self.layout, self.groups, self.rank = layout, dict(groups), rank
        self.coordinate = layout.coordinates[layout.ranks.index(rank)]
        self.tensor = TensorExecution(groups["tensor"]) if layout.dimensions[3] > 1 else None
        ownership = ExpertOwnership(layout.expert_owners, layout.dimensions[1], layout.approval_reference)
        self.expert = ExpertExecution(ownership, groups["expert"]) if layout.dimensions[1] > 1 else None
        self.context = ContextExecution(groups["context"]) if layout.dimensions[2] > 1 else None

    def replica_group(self, name):
        return self.groups["expert_parameters" if ".moe.experts." in name else "shared_parameters"]

    def unique_names(self, inventory):
        return frozenset(name for name, spec in inventory.items()
                         if self.replica_group(name).rank == 0 and (spec["axis"] is not None or self.coordinate[3] == 0))

    def gradients(self, parameters, accumulated):
        result = {}
        if not set(accumulated).issubset(parameters): raise ValueError("unknown gradient owner")
        for name, parameter in sorted(parameters.items()):
            value = accumulated.get(name)
            if value is None: value = torch.zeros_like(parameter, dtype=torch.float32)
            if value.shape != parameter.shape or value.dtype != torch.float32:
                raise ValueError("gradient shard shape/precision mismatch")
            result[name] = self.replica_group(name).sum(value)
        return result

    def split_batch(self, batch, *, tokenizer_identity, manifest_identity, device):
        from shinra.data.packing import ContextBatch, partition_context_batch
        from dataclasses import asdict
        source = self.groups["source_replicas"]
        source.agree(None if batch is None else asdict(batch), device=device)
        if batch is None:
            return ContextBatch((), (), (), (), (), (), tokenizer_identity, manifest_identity, 0, 0, 0)
        n, degree = len(batch.token_ids), self.layout.dimensions[2]
        offsets = tuple(n*i//degree for i in range(degree+1))
        return partition_context_batch(batch, offsets, self.coordinate[2])

    def ownership_metadata(self):
        return {"mode": "mixed", "ranks": list(self.layout.ranks), "rank": self.groups["world"].rank,
                "layout": self.layout.to_dict(), "layout_identity": self.layout.identity}
