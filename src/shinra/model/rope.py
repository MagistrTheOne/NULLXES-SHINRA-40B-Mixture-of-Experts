"""Fixed-frequency, adjacent-pair RoPE. No dynamic context modification."""
from __future__ import annotations

import torch
from torch import nn

from shinra.core.configuration import Configuration, ConfigurationError
from shinra.core.validation import validate_architecture


class RotaryEmbedding(nn.Module):
    def __init__(self, config: Configuration):
        super().__init__()
        validate_architecture(config)
        if config.get("rope.approval_reference") is None or config.get("rope.theta") is None:
            raise ConfigurationError("rope", "numeric contract requires explicit approval and theta")
        self.theta = config.get("rope.theta")
        self.head_dim = config.get("model.head_dim")
        self.max_positions = config.get("model.max_position_embeddings")

    def forward(self, value: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """value [..., heads,128]; positions [...] (no head axis)."""
        if value.shape[-1] != self.head_dim or tuple(value.shape[:-2]) != tuple(positions.shape):
            raise ValueError("RoPE value/position shapes do not match")
        if positions.dtype != torch.int64 or positions.device != value.device:
            raise ValueError("RoPE positions must be int64 on the input device")
        if value.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("RoPE requires BF16 or FP32")
        if bool(((positions < 0) | (positions >= self.max_positions)).any()):
            raise ValueError("RoPE position exceeds architectural range")
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError("nonfinite RoPE input")
        with torch.autocast(device_type=value.device.type, enabled=False):
            index = torch.arange(0, self.head_dim, 2, device=value.device, dtype=torch.float64)
            frequencies = torch.pow(self.theta, -index / self.head_dim)
            phases = positions.to(torch.float64).unsqueeze(-1) * frequencies
            cos = phases.cos().float().unsqueeze(-2)
            sin = phases.sin().float().unsqueeze(-2)
            even, odd = value.float()[..., 0::2], value.float()[..., 1::2]
            result = torch.stack((even*cos - odd*sin, even*sin + odd*cos), dim=-1).flatten(-2)
        return result.to(value.dtype)
