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
    def __init__(self, config: Configuration, *, device: str, expert_execution=None, context_execution=None, tensor_execution=None, parallel_execution=None):
        super().__init__()
        require_operation(config, "model")
        recipe = InitializationRecipe.from_configuration(config)
        self.parallel_execution = parallel_execution
        if parallel_execution is not None:
            from shinra.distributed.parallelism import MixedExecution
            from shinra.distributed.ownership import ParallelLayout
            if not isinstance(parallel_execution, MixedExecution) or any(x is not None for x in (expert_execution, context_execution, tensor_execution)):
                raise ValueError("mixed execution owns all component groups explicitly")
            admitted = ParallelLayout.from_manifest(config.get("distributed.ownership_manifest"))
            if admitted.identity != parallel_execution.layout.identity or config.get("distributed.backend") != parallel_execution.groups["world"].backend:
                raise ValueError("mixed layout differs from canonical configuration")
            parallel_execution.groups["world"].agree({"configuration": config.identity, "layout": admitted.to_dict()}, device=device)
            tensor_execution, expert_execution, context_execution = parallel_execution.tensor, parallel_execution.expert, parallel_execution.context
        self.tensor_execution = tensor_execution
        if tensor_execution is not None:
            from shinra.distributed.parallelism import TensorExecution
            if not isinstance(tensor_execution, TensorExecution) or config.get("distributed.backend") != tensor_execution.group.backend:
                raise ValueError("explicit TP group with matching configured backend required")
            if parallel_execution is None and (expert_execution is not None or context_execution is not None):
                raise ValueError("BLOCKED: combined TP/EP/CP requires the explicit MixedExecution contract")
            tensor_execution.group.agree({"configuration": config.identity, "tp_ranks": tensor_execution.group.ranks}, device=device)
        if expert_execution is not None:
            from shinra.distributed.expert_parallel import ExpertExecution
            if not isinstance(expert_execution, ExpertExecution):
                raise ValueError("explicit ExpertExecution required")
            if config.get("distributed.ownership_manifest") is None or config.get("distributed.backend") != expert_execution.group.backend:
                raise ValueError("approved ownership manifest and matching backend required")
            from shinra.distributed.ownership import ExpertOwnership
            admitted = (ExpertOwnership.from_manifest(config.get("distributed.ownership_manifest")) if parallel_execution is None
                        else expert_execution.ownership)
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
        self.embeddings = TokenEmbeddings(config, recipe, device=device, tensor_execution=tensor_execution)
        self.layers = nn.ModuleList(DecoderBlock(config, recipe, layer=i, device=device, expert_execution=expert_execution,
                                                context_execution=context_execution, tensor_execution=tensor_execution)
                                    for i in range(self.layer_count))
        self.final_norm = RMSNorm(config, recipe, name="final_norm.weight", device=device, tensor_execution=tensor_execution)
        self.lm_head = LanguageModelHead(config, recipe, device=device, tensor_execution=tensor_execution)

    def validate_inputs(self, token_ids: torch.Tensor, positions: torch.Tensor, segments: torch.Tensor,
                token_uids: torch.Tensor, balancing_bias: torch.Tensor, *,
                replay_assignments: tuple[torch.Tensor, ...] | None = None, context_range=None) -> None:
        """Canonical original-forward admission, never called by block replay.

        Collective order is TP input agreement, then CP bias/metadata agreement.
        This boundary reads metadata only and does not update routing state.
        """
        if token_ids.ndim != 1:
            raise ValueError("expected compact token IDs [T]")
        for value in (token_ids, positions, segments, token_uids):
            if value.shape != token_ids.shape or value.dtype != torch.int64 or value.device != self.embeddings.weight.device:
                raise ValueError("token metadata must be same-device int64 [T]")
        if (balancing_bias.shape != (self.layer_count, self.expert_count) or
                balancing_bias.dtype != torch.float32 or balancing_bias.device != token_ids.device or
                balancing_bias.requires_grad or not bool(torch.isfinite(balancing_bias).all())):
            raise ValueError("expected finite nontrainable same-device FP32 balancing snapshots [48,128]")
        if bool(((token_ids < 0) | (token_ids >= self.embeddings.vocab_size)).any()):
            raise ValueError("token ID outside vocabulary")
        if token_ids.numel() > self.context or bool((token_uids < 0).any()) or token_uids.unique().numel() != token_uids.numel():
            raise ValueError("context bound or unique nonnegative token UID contract violated")
        if replay_assignments is not None and len(replay_assignments) != self.layer_count:
            raise ValueError("recomputation requires assignments for all 48 layers")
        if replay_assignments is not None:
            for selected in replay_assignments:
                if (selected.shape != (token_ids.numel(), 8) or selected.dtype != torch.int64 or
                        selected.device != token_ids.device or bool(((selected < 0) | (selected >= 128)).any()) or
                        bool((selected[:, 1:] <= selected[:, :-1]).any())):
                    raise ValueError("invalid original Top-8 replay assignments")
        if self.parallel_execution is not None:
            execution = self.parallel_execution
            for role, group in execution.groups.items():
                if group.ranks != execution.layout.members(execution.rank, role):
                    raise ValueError("logical process-group membership changed")
            if (self.tensor_execution is not execution.tensor or self.expert_execution is not execution.expert or
                    self.context_execution is not execution.context):
                raise ValueError("model execution components differ from admitted mixed groups")
        if self.tensor_execution is not None:
            self.tensor_execution.group.agree({"tokens": token_ids.detach().cpu().tolist(),
                                               "positions": positions.detach().cpu().tolist(),
                                               "segments": segments.detach().cpu().tolist(),
                                               "uids": token_uids.detach().cpu().tolist(),
                                               "bias": balancing_bias.detach().cpu().tolist(),
                                               "replay": None if replay_assignments is None else
                                               [value.detach().cpu().tolist() for value in replay_assignments]}, device=token_ids.device)
        if bool(((positions < 0) | (positions >= self.context) | (segments < 0)).any()):
            raise ValueError("position or segment ID outside allowed range")
        seen = set()
        previous_segment = None
        previous_position = -1
        checked_positions, checked_segments = positions, segments
        if self.context_execution is not None:
            self.context_execution.group.agree(balancing_bias.detach().cpu().tolist(), device=token_ids.device)
            checked_positions, checked_segments = self.context_execution.metadata(
                positions, segments, token_uids, context_range=context_range)
        elif context_range is not None and context_range != (0, token_ids.numel(), token_ids.numel()):
            raise ValueError("resident context must own the complete sequence")
        for segment, position in zip(checked_segments.detach().cpu().tolist(), checked_positions.detach().cpu().tolist()):
            if segment != previous_segment:
                if segment in seen or position != 0:
                    raise ValueError("full-sequence segments must be contiguous and start at position zero")
                seen.add(segment)
            elif position != previous_position + 1:
                raise ValueError("segment positions must advance exactly once per token")
            previous_segment, previous_position = segment, position

    def forward(self, token_ids: torch.Tensor, positions: torch.Tensor, segments: torch.Tensor,
                token_uids: torch.Tensor, balancing_bias: torch.Tensor, *,
                replay_assignments: tuple[torch.Tensor, ...] | None = None, context_range=None) -> ModelOutput:
        """Full document-aware forward with explicit global CP ownership bounds."""
        self.validate_inputs(token_ids, positions, segments, token_uids, balancing_bias,
                             replay_assignments=replay_assignments, context_range=context_range)
        hidden = self.embeddings(token_ids)
        assignments, counts = [], []
        for index, block in enumerate(self.layers):
            replay = None if replay_assignments is None else replay_assignments[index]
            hidden, selected, load = block(hidden, positions, segments, token_uids, balancing_bias[index], replay_ids=replay)
            assignments.append(selected)
            counts.append(load)
        return ModelOutput(self.lm_head(self.final_norm(hidden)), tuple(assignments), tuple(counts))
