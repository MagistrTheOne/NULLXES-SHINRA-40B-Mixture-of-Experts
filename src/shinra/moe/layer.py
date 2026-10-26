"""Canonical resident or explicitly owner-sharded expert bank; no fallback."""
import torch
from torch import nn

from shinra.core.configuration import Configuration, ConfigurationError
from shinra.core.initialization import InitializationRecipe
from shinra.core.validation import validate_architecture
from .router import Router, Routing
from .dispatch import plan_dispatch, dispatch
from .combine import combine
from .experts import Expert


class MoELayer(nn.Module):
    """Fixed 128-expert logical MoE, with explicit optional EP ownership/transport."""
    def __init__(self, config: Configuration, recipe: InitializationRecipe, *, layer: int, device: str, expert_execution=None):
        super().__init__()
        validate_architecture(config)
        self.chunk = config.get("moe.chunk_tokens")
        if self.chunk is None:
            raise ConfigurationError("moe.chunk_tokens", "explicit chunk size required")
        self.router = Router(config, recipe, layer=layer, device=device)
        self.layer, self.expert_execution = layer, expert_execution
        self.num_experts = config.get("model.num_routed_experts")
        owned = range(self.num_experts) if expert_execution is None else expert_execution.local_experts(layer)
        self.experts = nn.ModuleDict({str(e): Expert(config, recipe, layer=layer, expert_id=e, device=device)
                                      for e in owned})

    def forward(self, hidden: torch.Tensor, token_uids: torch.Tensor, balancing_bias: torch.Tensor,
                *, replay_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, Routing, torch.Tensor]:
        routing = self.router(hidden, balancing_bias, replay_ids=replay_ids)
        plan = plan_dispatch(routing, token_uids, num_experts=self.num_experts)
        assigned = dispatch(hidden, plan)
        if self.expert_execution is not None:
            result = self.expert_execution.compute(assigned, routing, plan, token_uids,
                                                  layer=self.layer, experts=self.experts, chunk_tokens=self.chunk)
            return combine(result, routing, plan, output_dtype=hidden.dtype), routing, plan.expert_counts
        counts = plan.expert_counts.detach().cpu().tolist()
        outputs = []
        offset = 0
        for expert, count in zip(self.experts.values(), counts):
            if count:
                outputs.append(expert(assigned[offset:offset + count], chunk_tokens=self.chunk))
            offset += count
        if offset != assigned.shape[0]:
            raise RuntimeError("assignment conservation failure")
        result = torch.cat(outputs, dim=0) if outputs else assigned
        return combine(result, routing, plan, output_dtype=hidden.dtype), routing, plan.expert_counts
