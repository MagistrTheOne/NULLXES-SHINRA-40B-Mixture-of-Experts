"""Full resident expert bank; no implicit distributed execution or fallback."""
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
    """Resident computation primitive with externally managed balancing snapshot.

    Distributed sharding must call explicit transport/ownership components. This
    class does not infer a world size or claim to implement expert parallelism.
    """
    def __init__(self, config: Configuration, recipe: InitializationRecipe, *, layer: int, device: str):
        super().__init__()
        validate_architecture(config)
        self.chunk = config.get("moe.chunk_tokens")
        if self.chunk is None:
            raise ConfigurationError("moe.chunk_tokens", "explicit chunk size required")
        self.router = Router(config, recipe, layer=layer, device=device)
        self.experts = nn.ModuleList(Expert(config, recipe, layer=layer, expert_id=e, device=device)
                                     for e in range(config.get("model.num_routed_experts")))

    def forward(self, hidden: torch.Tensor, token_uids: torch.Tensor, balancing_bias: torch.Tensor,
                *, replay_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, Routing, torch.Tensor]:
        routing = self.router(hidden, balancing_bias, replay_ids=replay_ids)
        plan = plan_dispatch(routing, token_uids, num_experts=len(self.experts))
        assigned = dispatch(hidden, plan)
        counts = plan.expert_counts.detach().cpu().tolist()
        outputs = []
        offset = 0
        for expert, count in zip(self.experts, counts):
            if count:
                outputs.append(expert(assigned[offset:offset + count], chunk_tokens=self.chunk))
            offset += count
        if offset != assigned.shape[0]:
            raise RuntimeError("assignment conservation failure")
        result = torch.cat(outputs, dim=0) if outputs else assigned
        return combine(result, routing, plan, output_dtype=hidden.dtype), routing, plan.expert_counts
