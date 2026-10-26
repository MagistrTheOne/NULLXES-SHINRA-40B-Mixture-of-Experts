"""Pre-norm GQA and routed MoE residual block."""
import torch
from torch import nn

from shinra.core.configuration import Configuration
from shinra.core.initialization import InitializationRecipe
from shinra.moe.layer import MoELayer
from .attention import Attention
from .normalization import RMSNorm


class DecoderBlock(nn.Module):
    def __init__(self, config: Configuration, recipe: InitializationRecipe, *, layer: int, device: str, expert_execution=None, context_execution=None):
        super().__init__()
        prefix = f"layers.{layer}"
        self.attention_norm = RMSNorm(config, recipe, name=f"{prefix}.attention_norm.weight", device=device)
        self.attention = Attention(config, recipe, layer_id=layer, device=device, context_execution=context_execution)
        self.moe_norm = RMSNorm(config, recipe, name=f"{prefix}.moe_norm.weight", device=device)
        self.moe = MoELayer(config, recipe, layer=layer, device=device, expert_execution=expert_execution)

    def forward(self, hidden, positions, segments, token_uids, balancing_bias, *, replay_ids=None):
        hidden = hidden + self.attention(self.attention_norm(hidden), positions, segments)
        expert_output, routing, counts = self.moe(self.moe_norm(hidden), token_uids, balancing_bias,
                                                  replay_ids=replay_ids)
        hidden = hidden + expert_output
        if not bool(torch.isfinite(hidden).all()):
            raise FloatingPointError("nonfinite decoder residual")
        return hidden, routing.expert_ids, counts
