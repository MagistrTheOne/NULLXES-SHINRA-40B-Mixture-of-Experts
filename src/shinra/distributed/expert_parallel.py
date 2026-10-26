"""Real dropless EP dispatch and reverse exchange with autograd."""
import torch
from dataclasses import dataclass
from .communication import Group, exchange, exchange_sizes
from .ownership import ExpertOwnership


@dataclass(frozen=True)
class ExpertExecution:
    ownership: ExpertOwnership
    group: Group

    def __post_init__(self):
        if self.group.size < 2 or self.ownership.members != self.group.size:
            raise ValueError("EP requires an explicit multi-member ownership group")

    def local_experts(self, layer):
        return self.ownership.local_experts(layer, self.group.rank)

    def compute(self, assigned, routing, plan, token_uids, *, layer, experts, chunk_tokens):
        expert_ids = routing.expert_ids.reshape(-1).index_select(0, plan.assignment_order)
        return compute_experts(assigned, expert_ids, token_uids.index_select(0, plan.token_indices),
                               layer=layer, experts=experts, ownership=self.ownership,
                               group=self.group, chunk_tokens=chunk_tokens)


def compute_experts(assigned: torch.Tensor, expert_ids: torch.Tensor, token_uids: torch.Tensor, *,
                    layer: int, experts, ownership: ExpertOwnership, group: Group, chunk_tokens: int):
    if ownership.members != group.size or not 0 <= layer < 48:
        raise ValueError("expert ownership/group mismatch")
    expected = {str(e) for e in ownership.local_experts(layer, group.rank)}
    if set(experts.keys()) != expected:
        raise ValueError("local expert inventory does not match ownership")
    if assigned.ndim != 2 or assigned.shape[1] != 2560:
        raise ValueError("expected [assignments,2560] expert input")
    for tensor in (expert_ids, token_uids):
        if tensor.dtype != torch.int64 or tensor.shape != (assigned.shape[0],) or tensor.device != assigned.device:
            raise ValueError("EP metadata shape/dtype/device mismatch")
    if bool(((expert_ids < 0) | (expert_ids >= 128) | (token_uids < 0)).any()):
        raise ValueError("invalid EP assignment metadata")
    owner_map = torch.tensor(ownership.owners[layer], device=assigned.device, dtype=torch.int64)
    owners = owner_map[expert_ids]
    order = torch.argsort(owners, stable=True)
    sends = torch.bincount(owners, minlength=group.size)
    receive_sizes = exchange_sizes(sends, group)
    send_sizes = tuple(sends.cpu().tolist())
    received = exchange(assigned[order], send_sizes, receive_sizes, group)
    metadata = exchange(torch.stack((expert_ids, token_uids), dim=1)[order], send_sizes, receive_sizes, group)
    # Stable per-expert UID order across all source ranks.
    uid_order = torch.argsort(metadata[:, 1], stable=True)
    expert_order = uid_order[torch.argsort(metadata[uid_order, 0], stable=True)]
    sorted_ids = metadata[expert_order, 0]
    sorted_metadata = metadata[expert_order]
    if sorted_metadata.shape[0] > 1 and bool((sorted_metadata[1:] == sorted_metadata[:-1]).all(dim=1).any()):
        raise ValueError("duplicate global token/expert assignment; replicated inputs must not be counted twice")
    pieces, offset = [], 0
    for expert in sorted(expected, key=int):
        count = int((sorted_ids == int(expert)).sum().item())
        if count:
            pieces.append(experts[expert](received[expert_order[offset:offset+count]], chunk_tokens=chunk_tokens))
        offset += count
    if offset != received.shape[0]:
        raise RuntimeError("received assignments for an unowned expert")
    computed = torch.cat(pieces, dim=0) if pieces else received
    original_receive_order = computed[torch.argsort(expert_order, stable=True)]
    returned = exchange(original_receive_order, receive_sizes, send_sizes, group)
    return returned[torch.argsort(order, stable=True)]
