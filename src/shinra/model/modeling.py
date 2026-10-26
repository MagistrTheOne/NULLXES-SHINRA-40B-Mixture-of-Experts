"""Canonical 48-block model for complete, compact document segments.

No instance is constructed on import. Distributed execution and KV-cache decode
are separate integrations; this full-sequence primitive does not emulate them.
"""
from dataclasses import dataclass
import hashlib
import json
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
    def __init__(self, config: Configuration, *, device: str, expert_execution=None, context_execution=None):
        super().__init__()
        require_operation(config, "model")
        recipe = InitializationRecipe.from_configuration(config)
        if expert_execution is not None:
            from shinra.distributed.expert_parallel import ExpertExecution
            if not isinstance(expert_execution, ExpertExecution):
                raise ValueError("explicit ExpertExecution required")
            if config.get("distributed.ownership_manifest") is None or config.get("distributed.backend") != expert_execution.group.backend:
                raise ValueError("approved ownership manifest and matching backend required")
            from shinra.distributed.ownership import ExpertOwnership
            admitted = ExpertOwnership.from_manifest(config.get("distributed.ownership_manifest"))
            if admitted.identity != expert_execution.ownership.identity:
                raise ValueError("EP ownership differs from configured manifest")
            expert_execution.group.agree({"config": config.identity, "ownership": expert_execution.ownership.identity}, device=device)
        self.expert_execution = expert_execution
        self.context_execution = context_execution
        if context_execution is not None:
            from shinra.distributed.context_parallel import ContextExecution
            if not isinstance(context_execution, ContextExecution) or config.get("distributed.backend") != context_execution.group.backend:
                raise ValueError("explicit context group with matching backend required")
            context_execution.group.agree({"config": config.identity, "context_ranks": context_execution.group.ranks}, device=device)
        self.context = config.get("model.max_position_embeddings")
        self.layer_count = config.get("model.num_hidden_layers")
        self.expert_count = config.get("model.num_routed_experts")
        self.configuration_identity = config.identity
        self.rope_identity = hashlib.sha256(json.dumps(dict(config.get("rope")), sort_keys=True,
                                                       separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
        self.embeddings = TokenEmbeddings(config, recipe, device=device)
        self.layers = nn.ModuleList(DecoderBlock(config, recipe, layer=i, device=device, expert_execution=expert_execution,
                                                context_execution=context_execution)
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
        checked_positions, checked_segments = positions, segments
        if self.context_execution is not None:
            checked_positions, checked_segments = self.context_execution.metadata(positions, segments, token_uids)
        for segment, position in zip(checked_segments.detach().cpu().tolist(), checked_positions.detach().cpu().tolist()):
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
