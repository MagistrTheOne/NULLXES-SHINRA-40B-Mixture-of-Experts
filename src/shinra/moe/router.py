"""FP32 selected-logit routing; balancing updates belong to the step transaction."""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

from shinra.core.configuration import Configuration
from shinra.core.initialization import InitializationRecipe, ParameterSpec, initialize_parameter
from shinra.core.validation import validate_architecture


@dataclass(frozen=True)
class Routing:
    expert_ids: torch.Tensor  # [T, 8], ascending expert ID
    weights: torch.Tensor  # [T, 8], FP32, differentiable


class Router(nn.Module):
    def __init__(self, config: Configuration, recipe: InitializationRecipe, *,
                 layer: int, device: str, tensor_execution=None):
        super().__init__()
        validate_architecture(config)
        m = config.get("model")
        if type(layer) is not int or not 0 <= layer < m["num_hidden_layers"]:
            raise ValueError("invalid router layer")
        self.experts = m["num_routed_experts"]
        self.top_k = m["num_experts_per_tok"]
        self.hidden = m["hidden_size"]
        self.tensor_execution = tensor_execution
        spec = ParameterSpec(f"layers.{layer}.moe.router.weight",
                             (self.experts, self.hidden), "float32", "router")
        self.weight = initialize_parameter(spec, recipe, device=device, tensor_execution=tensor_execution)

    def forward(self, hidden: torch.Tensor, balancing_bias: torch.Tensor, *,
                replay_ids: torch.Tensor | None = None) -> Routing:
        """Replay retains original selection but recomputes differentiable weights.

        Caller supplies a frozen balancing snapshot and records load only for the
        original forward. This function never mutates or counts balancing state.
        """
        if hidden.ndim != 2 or hidden.shape[1] != self.hidden:
            raise ValueError("router requires compact [T, hidden_size] input")
        if hidden.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("router input must be BF16 or FP32")
        if self.weight.dtype != torch.float32 or balancing_bias.dtype != torch.float32:
            raise ValueError("router weights and balancing bias must remain FP32")
        if balancing_bias.shape != (self.experts,) or balancing_bias.requires_grad:
            raise ValueError("balancing bias must be nontrainable [128]")
        if hidden.device != self.weight.device or balancing_bias.device != hidden.device:
            raise ValueError("router device mismatch")
        if hidden.device.type == "cuda" and torch.backends.cuda.matmul.allow_tf32:
            raise ValueError("router requires IEEE FP32 matmul, disable TF32 explicitly")
        if not all(bool(torch.isfinite(x).all()) for x in (hidden, self.weight, balancing_bias)):
            raise FloatingPointError("nonfinite router input/state")
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            logits = F.linear(hidden.float(), self.weight)
            scores = logits + balancing_bias
            if not bool(torch.isfinite(scores).all()):
                raise FloatingPointError("nonfinite router logits/selection scores")
            if replay_ids is None:
                ranked = torch.argsort(scores, dim=-1, descending=True, stable=True)
                ids = ranked[:, :self.top_k].sort(dim=-1).values
            else:
                ids = replay_ids
                if ids.shape != (hidden.shape[0], self.top_k) or ids.dtype != torch.int64:
                    raise ValueError("invalid replay assignment shape/dtype")
                if ids.device != hidden.device:
                    raise ValueError("replay assignment device mismatch")
                if bool(((ids < 0) | (ids >= self.experts)).any()):
                    raise ValueError("replay expert ID out of range")
                if bool((ids[:, 1:] <= ids[:, :-1]).any()):
                    raise ValueError("replay assignments must be unique and ascending")
            selected = logits.gather(1, ids)
            shifted = selected - selected.max(dim=-1, keepdim=True).values
            numerator = shifted.exp()
            denominator = numerator.sum(dim=-1, keepdim=True)
            weights = numerator / denominator
            if not bool(torch.isfinite(weights).all()) or bool((denominator <= 0).any()):
                raise FloatingPointError("invalid selected-logit softmax")
        if self.tensor_execution is not None:
            # Identical assignments are a prerequisite for the per-expert TP
            # collective schedule, including during full block recomputation.
            self.tensor_execution.group.agree(
                {"ids": ids.detach().cpu().tolist(), "weights": weights.detach().cpu().tolist()},
                device=hidden.device)
        return Routing(ids, weights)
