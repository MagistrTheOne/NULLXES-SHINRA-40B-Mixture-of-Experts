"""Assignment-preserving FP32 combination in ascending expert-ID order."""
import torch

from .dispatch import DispatchPlan
from .router import Routing


def combine(expert_outputs: torch.Tensor, routing: Routing, plan: DispatchPlan,
            *, output_dtype: torch.dtype) -> torch.Tensor:
    if expert_outputs.ndim != 2 or expert_outputs.shape[0] != plan.tokens * plan.slots:
        raise ValueError("expert output count does not match dispatch plan")
    if routing.weights.shape != (plan.tokens, plan.slots) or routing.weights.dtype != torch.float32:
        raise ValueError("expected FP32 [T,8] combination weights")
    if expert_outputs.device != routing.weights.device or expert_outputs.device != plan.inverse_order.device:
        raise ValueError("combine device mismatch")
    if output_dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("unsupported combine output dtype")
    if not bool(torch.isfinite(expert_outputs).all()) or not bool(torch.isfinite(routing.weights).all()):
        raise FloatingPointError("nonfinite expert outputs/weights")
    ordered = expert_outputs.index_select(0, plan.inverse_order).reshape(plan.tokens, plan.slots, expert_outputs.shape[1])
    with torch.autocast(device_type=expert_outputs.device.type, enabled=False):
        result = ordered[:, 0].float() * routing.weights[:, 0, None]
        for slot in range(1, plan.slots):
            result = result + ordered[:, slot].float() * routing.weights[:, slot, None]
    result = result.to(output_dtype)
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("nonfinite combined MoE output")
    return result
