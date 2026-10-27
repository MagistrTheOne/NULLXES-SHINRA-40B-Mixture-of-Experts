"""Canonical individual SwiGLU experts, with explicit dropless chunking."""
import torch
from torch import nn
from torch.nn import functional as F

from shinra.core.configuration import Configuration
from shinra.core.initialization import InitializationRecipe, ParameterSpec, initialize_parameter
from shinra.core.validation import validate_architecture


class Expert(nn.Module):
    def __init__(self, config: Configuration, recipe: InitializationRecipe, *,
                 layer: int, expert_id: int, device: str, tensor_execution=None):
        super().__init__()
        validate_architecture(config)
        m = config.get("model")
        if type(layer) is not int or not 0 <= layer < m["num_hidden_layers"]:
            raise ValueError("invalid expert layer")
        if type(expert_id) is not int or not 0 <= expert_id < m["num_routed_experts"]:
            raise ValueError("invalid expert ID")
        self.tensor_execution = tensor_execution
        self.hidden = m["hidden_size"]
        width = m["intermediate_size"]
        prefix = f"layers.{layer}.moe.experts.{expert_id}"
        for name, shape, role in (("gate", (width, self.hidden), "projection"),
                                  ("up", (width, self.hidden), "projection"),
                                  ("down", (self.hidden, width), "residual_projection")):
            setattr(self, name, nn.ParameterDict({"weight": initialize_parameter(
                ParameterSpec(f"{prefix}.{name}.weight", shape, "bfloat16", role), recipe, device=device, tensor_execution=tensor_execution)}))

    def forward(self, hidden: torch.Tensor, *, chunk_tokens: int) -> torch.Tensor:
        if type(chunk_tokens) is not int or chunk_tokens <= 0:
            raise ValueError("explicit positive expert chunk size required")
        if hidden.ndim != 2 or hidden.shape[1] != self.hidden:
            raise ValueError("expert requires [assignments,2560]")
        if hidden.dtype != self.gate["weight"].dtype or hidden.device != self.gate["weight"].device:
            raise ValueError("expert dtype/device mismatch")
        if not bool(torch.isfinite(hidden).all()):
            raise FloatingPointError("nonfinite expert input")
        if hidden.shape[0] == 0:
            # No assignments: no expert arithmetic, retain the empty input graph.
            return hidden
        chunks = []
        for start in range(0, hidden.shape[0], chunk_tokens):
            portion = hidden[start:start + chunk_tokens]
            linear = F.linear if self.tensor_execution is None else self.tensor_execution.linear
            gate = F.silu(linear(portion, self.gate["weight"]))
            up = linear(portion, self.up["weight"])
            output = linear(gate * up, self.down["weight"])
            if not bool(torch.isfinite(output).all()):
                raise FloatingPointError("nonfinite expert output; abort routing event")
            chunks.append(output)
        return torch.cat(chunks, dim=0)
