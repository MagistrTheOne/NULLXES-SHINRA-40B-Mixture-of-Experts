"""Generation over identity-checked native model/tokenizer/cache components."""
from dataclasses import dataclass
import hashlib
import torch
from shinra.core.validation import require_operation
from shinra.tokenizer.serialization import canonical_bytes, TokenizerArtifact
from .kv_cache import KVCache
from .decoding import append_tokens
from .sampling import sample


@dataclass(frozen=True)
class GenerationResult:
    token_ids: tuple[int, ...]
    stop_reason: str
    structured_output: tuple


def generate(model, artifact: TokenizerArtifact, config, prompt: tuple[int, ...], *,
             checkpoint_identity: str, balancing_bias: torch.Tensor) -> GenerationResult:
    require_operation(config, "infer")
    if model.expert_execution is not None or model.context_execution is not None:
        raise ValueError("single-sequence generation requires resident experts; distributed generation is not admitted")
    if artifact.identity != config.get("tokenizer.artifact_identity") or not artifact.tokenizer.vocabulary.complete:
        raise ValueError("native tokenizer identity mismatch")
    if getattr(model, "checkpoint_identity", None) != checkpoint_identity or model.configuration_identity != config.identity:
        raise ValueError("loaded model identity mismatch")
    maximum = config.get("model.max_position_embeddings")
    if not prompt or len(prompt) > maximum or any(type(t) is not int or not 0 <= t < 131072 for t in prompt):
        raise ValueError("invalid prompt; no implicit truncation")
    registry = artifact.tokenizer.vocabulary.registry
    if any(t == registry.token("pad") or 15 <= t < 32 for t in prompt):
        raise ValueError("PAD/reserved token in prompt")
    rope_identity = hashlib.sha256(canonical_bytes(dict(config.get("rope")))).hexdigest()
    cache = KVCache(checkpoint_identity=checkpoint_identity, rope_identity=rope_identity,
                    max_positions=maximum, layers=48)
    device = model.embeddings.weight.device
    generator = None
    if config.get("inference.sampling") == "categorical":
        generator = torch.Generator(device=device)
        generator.manual_seed(config.get("inference.seed"))
    generated, reason = [], "max_new_tokens"
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            logits = append_tokens(model, torch.tensor(prompt, dtype=torch.int64, device=device), balancing_bias, cache)
            for _ in range(config.get("inference.max_new_tokens")):
                if len(prompt) + len(generated) >= maximum:
                    reason = "context_limit"
                    break
                token = sample(logits, config, generator=generator)
                if token == registry.token("pad") or 15 <= token < 32:
                    raise ValueError("generated forbidden structural ID; masking policy requires approval")
                generated.append(token)
                if token in config.get("inference.stop_token_ids"):
                    reason = "stop_token"
                    break
                logits = append_tokens(model, torch.tensor([token], dtype=torch.int64, device=device), balancing_bias, cache)
    finally:
        model.train(was_training)
    return GenerationResult(tuple(generated), reason, artifact.tokenizer.decode_structured(generated))
