"""Real synchronous collectives and explicitly paired autograd communication."""
from dataclasses import dataclass
import hashlib
import json
import torch
import torch.distributed as dist


@dataclass(frozen=True)
class Group:
    process_group: object
    ranks: tuple[int, ...]
    backend: str

    def __post_init__(self):
        if not dist.is_available() or not dist.is_initialized() or self.process_group is None:
            raise RuntimeError("BLOCKED: explicit initialized process group required")
        if self.backend not in ("gloo", "nccl") or str(dist.get_backend(self.process_group)) != self.backend:
            raise RuntimeError("BLOCKED: unsupported or mismatched backend")
        if tuple(dist.get_process_group_ranks(self.process_group)) != self.ranks:
            raise ValueError("process-group membership mismatch")
        if dist.get_rank() not in self.ranks:
            raise ValueError("caller is not a group member")

    @property
    def size(self):
        return len(self.ranks)

    @property
    def rank(self):
        return self.ranks.index(dist.get_rank())

    def validate_device(self, tensor):
        if self.backend == "nccl" and tensor.device.type != "cuda":
            raise ValueError("NCCL tensors must be on an explicitly selected CUDA device")
        if self.backend == "nccl" and tensor.device.index != torch.cuda.current_device():
            raise ValueError("caller must select the tensor CUDA device before NCCL operations")
        if self.backend == "gloo" and tensor.device.type != "cpu":
            raise RuntimeError("BLOCKED: this Gloo adapter admits CPU tensors only")

    def sum(self, tensor):
        self.validate_device(tensor)
        result = tensor.detach().clone()
        dist.all_reduce(result, op=dist.ReduceOp.SUM, group=self.process_group)
        return result

    def agree(self, metadata, *, device):
        """Reject divergent collective contracts before tensor mutation."""
        payload = json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        digest = torch.tensor(list(hashlib.sha256(payload).digest()), dtype=torch.uint8, device=device)
        self.validate_device(digest)
        gathered = [torch.empty_like(digest) for _ in self.ranks]
        dist.all_gather(gathered, digest, group=self.process_group)
        if any(not torch.equal(digest, other) for other in gathered):
            raise RuntimeError("distributed metadata disagreement; group restart required")

    def integer_sum(self, tensor):
        if tensor.dtype != torch.int64:
            raise ValueError("integer statistics require int64")
        self.validate_device(tensor)
        gathered = [torch.empty_like(tensor) for _ in self.ranks]
        dist.all_gather(gathered, tensor.contiguous(), group=self.process_group)
        rows = [x.cpu().reshape(-1).tolist() for x in gathered]
        values = [sum(column) for column in zip(*rows)]
        if any(v < 0 or v > 2**63-1 for v in values) or any(v < 0 for row in rows for v in row):
            raise OverflowError("global integer count outside nonnegative int64")
        return torch.tensor(values, dtype=torch.int64, device=tensor.device).reshape(tensor.shape)


def _transfer(value, send_sizes, receive_sizes, group):
    """Ordered nonempty P2P edges; no reliance on zero-count payload support."""
    group.validate_device(value)
    if (len(send_sizes) != group.size or len(receive_sizes) != group.size or
            any(type(n) is not int or n < 0 for n in (*send_sizes,*receive_sizes)) or sum(send_sizes) != value.shape[0]):
        raise ValueError("invalid assignment exchange sizes")
    admitted = exchange_sizes(torch.tensor(send_sizes, dtype=torch.int64, device=value.device), group)
    if admitted != tuple(receive_sizes): raise ValueError("receive inventory differs from actual send counts")
    result = value.new_empty((sum(receive_sizes), *value.shape[1:]))
    send_offsets = [sum(send_sizes[:i]) for i in range(group.size)]
    recv_offsets = [sum(receive_sizes[:i]) for i in range(group.size)]
    for source in range(group.size):
        for destination in range(group.size):
            if source == destination == group.rank:
                n = send_sizes[destination]
                if n: result[recv_offsets[source]:recv_offsets[source]+n].copy_(value[send_offsets[destination]:send_offsets[destination]+n])
            elif group.rank == source and send_sizes[destination]:
                payload = value[send_offsets[destination]:send_offsets[destination]+send_sizes[destination]].contiguous()
                dist.isend(payload, dst=group.ranks[destination], group=group.process_group).wait()
            elif group.rank == destination and receive_sizes[source]:
                target = result[recv_offsets[source]:recv_offsets[source]+receive_sizes[source]]
                dist.irecv(target, src=group.ranks[source], group=group.process_group).wait()
    # Zero-work members must not enter another group while peers still execute
    # this exchange. This is a real control collective, not a training token.
    completed = group.integer_sum(torch.ones((), dtype=torch.int64, device=value.device))
    if completed.item() != group.size:
        raise RuntimeError("incomplete expert exchange participation")
    return result


class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, send_sizes, receive_sizes, group):
        ctx.send_sizes, ctx.receive_sizes, ctx.group = send_sizes, receive_sizes, group
        return _transfer(value, send_sizes, receive_sizes, group)

    @staticmethod
    def backward(ctx, gradient):
        return _transfer(gradient, ctx.receive_sizes, ctx.send_sizes, ctx.group), None, None, None


def exchange(value, send_sizes: tuple[int, ...], receive_sizes: tuple[int, ...], group: Group):
    return _AllToAll.apply(value, send_sizes, receive_sizes, group)


def exchange_sizes(send_sizes: torch.Tensor, group: Group) -> tuple[int, ...]:
    group.validate_device(send_sizes)
    if send_sizes.dtype != torch.int64 or send_sizes.shape != (group.size,):
        raise ValueError("expected int64 per-owner assignment counts")
    if bool((send_sizes < 0).any()): raise ValueError("negative assignment count")
    rows = [torch.empty_like(send_sizes) for _ in group.ranks]
    dist.all_gather(rows, send_sizes.contiguous(), group=group.process_group)
    return tuple(int(row[group.rank].item()) for row in rows)
