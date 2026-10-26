"""Full-vocabulary summed causal loss; normalization belongs to global step."""
import torch
from torch.nn import functional as F


def causal_loss_sum(logits: torch.Tensor, token_ids: torch.Tensor, loss_mask: torch.Tensor):
    if logits.ndim != 2 or logits.shape != (token_ids.numel(), 131072):
        raise ValueError("expected complete [T,131072] vocabulary logits")
    if token_ids.ndim != 1 or token_ids.dtype != torch.int64:
        raise ValueError("expected int64 compact token IDs")
    if loss_mask.dtype != torch.bool or loss_mask.shape != (max(token_ids.numel()-1, 0),):
        raise ValueError("expected boolean shifted-target mask [T-1]")
    if logits.device != token_ids.device or loss_mask.device != logits.device:
        raise ValueError("loss tensor device mismatch")
    selected = loss_mask.nonzero(as_tuple=False).flatten()
    if selected.numel() == 0:
        return logits[:0].sum(), 0
    with torch.autocast(device_type=logits.device.type, enabled=False):
        prediction = logits.index_select(0, selected).float()
        targets = token_ids.index_select(0, selected + 1)
        loss = F.cross_entropy(prediction, targets, reduction="sum", label_smoothing=0.0)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("nonfinite causal loss")
    return loss, selected.numel()
