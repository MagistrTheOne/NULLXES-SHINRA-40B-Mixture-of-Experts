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


class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, send_sizes, receive_sizes, group):
        group.validate_device(value)
        if len(send_sizes) != group.size or len(receive_sizes) != group.size or sum(send_sizes) != value.shape[0]:
            raise ValueError("all-to-all split contract mismatch")
        if any(type(n) is not int or n < 0 for n in (*send_sizes, *receive_sizes)):
            raise ValueError("invalid all-to-all split size")
        result = value.new_empty((sum(receive_sizes), *value.shape[1:]))
        dist.all_to_all_single(result, value.contiguous(), output_split_sizes=list(receive_sizes),
                               input_split_sizes=list(send_sizes), group=group.process_group)
        ctx.send_sizes, ctx.receive_sizes, ctx.group = send_sizes, receive_sizes, group
        return result

    @staticmethod
    def backward(ctx, gradient):
        result = gradient.new_empty((sum(ctx.send_sizes), *gradient.shape[1:]))
        dist.all_to_all_single(result, gradient.contiguous(), output_split_sizes=list(ctx.send_sizes),
                               input_split_sizes=list(ctx.receive_sizes), group=ctx.group.process_group)
        return result, None, None, None


def exchange(value, send_sizes: tuple[int, ...], receive_sizes: tuple[int, ...], group: Group):
    return _AllToAll.apply(value, send_sizes, receive_sizes, group)


def exchange_sizes(send_sizes: torch.Tensor, group: Group) -> tuple[int, ...]:
    group.validate_device(send_sizes)
    if send_sizes.dtype != torch.int64 or send_sizes.shape != (group.size,):
        raise ValueError("expected int64 per-owner assignment counts")
    receive = torch.empty_like(send_sizes)
    dist.all_to_all_single(receive, send_sizes.contiguous(), group=group.process_group)
    return tuple(receive.cpu().tolist())
