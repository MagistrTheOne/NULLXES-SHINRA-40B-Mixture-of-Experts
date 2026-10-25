"""Stable local assignment permutation; no simulated distributed transport."""
from dataclasses import dataclass
import torch

from .router import Routing


@dataclass(frozen=True)
class DispatchPlan:
    assignment_order: torch.Tensor
    inverse_order: torch.Tensor
    token_indices: torch.Tensor
    expert_counts: torch.Tensor
    tokens: int
    slots: int


def plan_dispatch(routing: Routing, token_uids: torch.Tensor, *, num_experts: int) -> DispatchPlan:
    """Sort assignments by (expert ID, globally unique token UID).

    token_uids identify compact tokens within the logical routing event. The
    distributed caller owns event identity and global UID uniqueness.
    """
    ids = routing.expert_ids
    if ids.ndim != 2 or ids.dtype != torch.int64 or ids.shape[1] != 8:
        raise ValueError("expected int64 [T,8] expert assignments")
    if type(num_experts) is not int or num_experts != 128:
        raise ValueError("SHINRA requires 128 routed experts")
    t, k = ids.shape
    if token_uids.shape != (t,) or token_uids.dtype != torch.int64 or token_uids.device != ids.device:
        raise ValueError("expected same-device int64 [T] token UIDs")
    if token_uids.unique().numel() != t or bool((token_uids < 0).any()):
        raise ValueError("token UIDs must be unique and nonnegative")
    if bool(((ids < 0) | (ids >= num_experts)).any()) or bool((ids[:, 1:] <= ids[:, :-1]).any()):
        raise ValueError("expert assignments must be in range, unique and ascending")
    flat_ids = ids.reshape(-1)
    token_indices = torch.arange(t, device=ids.device).repeat_interleave(k)
    uid_order = torch.argsort(token_uids[token_indices], stable=True)
    expert_order = torch.argsort(flat_ids[uid_order], stable=True)
    order = uid_order[expert_order]
    inverse = torch.argsort(order, stable=True)
    counts = torch.bincount(flat_ids, minlength=num_experts)
    return DispatchPlan(order, inverse, token_indices[order], counts, t, k)


def dispatch(hidden: torch.Tensor, plan: DispatchPlan) -> torch.Tensor:
    """Gather [T,D] into expert-major [8*T,D], retaining autograd paths."""
    if hidden.ndim != 2 or hidden.shape[0] != plan.tokens or hidden.device != plan.token_indices.device:
        raise ValueError("hidden tensor does not match dispatch plan")
    return hidden.index_select(0, plan.token_indices)
