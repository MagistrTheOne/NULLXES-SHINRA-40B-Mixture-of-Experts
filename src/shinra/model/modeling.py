"""Canonical 48-block model for complete, compact document segments.

No instance is constructed on import. Distributed execution and KV-cache decode
are separate integrations; this full-sequence primitive does not emulate them.
"""
from dataclasses import dataclass
import torch
from torch import nn

from shinra.core.configuration import Configuration
from shinra.core.initialization import InitializationRecipe
from shinra.core.validation import require_operation
from .embeddings import TokenEmbeddings
from .decoder import DecoderBlock
from .normalization import RMSNorm
from .lm_head import LanguageModelHead


@dataclass(frozen=True)
class ModelOutput:
    logits: torch.Tensor
    assignments: tuple[torch.Tensor, ...]
    expert_counts: tuple[torch.Tensor, ...]


class ShinraForCausalLM(nn.Module):
    def __init__(self, config: Configuration, *, device: str):
        super().__init__()
        require_operation(config, "model")
        recipe = InitializationRecipe.from_configuration(config)
        self.context = config.get("model.max_position_embeddings")
        self.layer_count = config.get("model.num_hidden_layers")
        self.expert_count = config.get("model.num_routed_experts")
        self.configuration_identity = config.identity
        self.embeddings = TokenEmbeddings(config, recipe, device=device)
        self.layers = nn.ModuleList(DecoderBlock(config, recipe, layer=i, device=device)
                                    for i in range(self.layer_count))
        self.final_norm = RMSNorm(config, recipe, name="final_norm.weight", device=device)
        self.lm_head = LanguageModelHead(config, recipe, device=device)

    def forward(self, token_ids: torch.Tensor, positions: torch.Tensor, segments: torch.Tensor,
                token_uids: torch.Tensor, balancing_bias: torch.Tensor, *,
                replay_assignments: tuple[torch.Tensor, ...] | None = None) -> ModelOutput:
        """Full document-aware causal forward; padding must be compacted by caller.

        Each contiguous segment begins at position zero. Arbitrary context shards
        and decode continuations require their own explicit attention integration.
        """
        if token_ids.ndim != 1:
            raise ValueError("expected compact token IDs [T]")
        for value in (token_ids, positions, segments, token_uids):
            if value.shape != token_ids.shape or value.dtype != torch.int64 or value.device != self.embeddings.weight.device:
                raise ValueError("token metadata must be same-device int64 [T]")
        if balancing_bias.shape != (self.layer_count, self.expert_count):
            raise ValueError("expected balancing snapshots [48,128]")
        if replay_assignments is not None and len(replay_assignments) != self.layer_count:
            raise ValueError("recomputation requires assignments for all 48 layers")
        if bool(((positions < 0) | (positions >= self.context) | (segments < 0)).any()):
            raise ValueError("position or segment ID outside allowed range")
        seen = set()
        previous_segment = None
        previous_position = -1
        for segment, position in zip(segments.detach().cpu().tolist(), positions.detach().cpu().tolist()):
            if segment != previous_segment:
                if segment in seen or position != 0:
                    raise ValueError("full-sequence segments must be contiguous and start at position zero")
                seen.add(segment)
            elif position != previous_position + 1:
                raise ValueError("segment positions must advance exactly once per token")
            previous_segment, previous_position = segment, position
        hidden = self.embeddings(token_ids)
        assignments, counts = [], []
        for index, block in enumerate(self.layers):
            replay = None if replay_assignments is None else replay_assignments[index]
            hidden, selected, load = block(hidden, positions, segments, token_uids, balancing_bias[index], replay_ids=replay)
            assignments.append(selected)
            counts.append(load)
        return ModelOutput(self.lm_head(self.final_norm(hidden)), tuple(assignments), tuple(counts))
