"""RMSNorm and independent per-head Q/K normalization (D02/D04)."""
from __future__ import annotations

import torch
from torch import nn

from shinra.core.configuration import Configuration
from shinra.core.initialization import InitializationRecipe, ParameterSpec, initialize_parameter
from shinra.core.validation import validate_architecture


class RMSNorm(nn.Module):
    def __init__(self, config: Configuration, recipe: InitializationRecipe, *, name: str,
                 device: str, head_kind: str | None = None, tensor_execution=None):
        super().__init__()
        validate_architecture(config)
        m = config.get("model")
        if head_kind is None:
            shape = (m["hidden_size"],)
        elif head_kind in ("query", "key"):
            heads = m["num_attention_heads"] if head_kind == "query" else m["num_key_value_heads"]
            shape = (heads, m["head_dim"])
        else:
            raise ValueError("head_kind must be query, key or None")
        self.weight = initialize_parameter(ParameterSpec(name, shape, "bfloat16", "norm"), recipe, device=device, tensor_execution=tensor_execution)
        self.eps = m["rms_norm_eps"]
        self.normalized_shape = shape

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if tuple(value.shape[-len(self.normalized_shape):]) != self.normalized_shape:
            raise ValueError(f"RMSNorm expected trailing shape {self.normalized_shape}")
        if value.dtype not in (torch.bfloat16, torch.float32) or value.device != self.weight.device:
            raise ValueError("RMSNorm dtype/device mismatch")
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError("nonfinite RMSNorm input")
        with torch.autocast(device_type=value.device.type, enabled=False):
            x = value.float()
            normalized = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
            result = normalized * self.weight.float()
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError("nonfinite RMSNorm result")
        return result.to(value.dtype)
