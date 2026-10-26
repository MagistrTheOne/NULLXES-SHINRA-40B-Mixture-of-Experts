"""Single-sequence prefill and incremental decode through canonical blocks."""
import torch
from .kv_cache import KVCache


def append_tokens(model, token_ids: torch.Tensor, balancing_bias: torch.Tensor, cache: KVCache):
    if torch.is_grad_enabled() or model.training:
        raise RuntimeError("decode requires explicit no-grad context and eval mode")
    if getattr(model, "checkpoint_identity", None) != cache.checkpoint_identity:
        raise ValueError("model/cache checkpoint identity mismatch")
    if getattr(model, "rope_identity", None) != cache.rope_identity:
        raise ValueError("model/cache RoPE identity mismatch")
    if token_ids.ndim != 1 or token_ids.dtype != torch.int64 or not token_ids.numel():
        raise ValueError("prefill/decode requires nonempty int64 tokens")
    if balancing_bias.shape != (48,128) or balancing_bias.dtype != torch.float32 or balancing_bias.requires_grad:
        raise ValueError("inference requires frozen FP32 router biases")
    start = cache.length
    cache.begin(token_ids.numel())
    try:
        positions = torch.arange(start, start + token_ids.numel(), dtype=torch.int64, device=token_ids.device)
        query_segments = torch.zeros_like(positions)  # one logical sequence
        all_positions = torch.arange(start + token_ids.numel(), dtype=torch.int64, device=token_ids.device)
        all_segments = torch.zeros_like(all_positions)
        hidden = model.embeddings(token_ids)
        for index, block in enumerate(model.layers):
            q, new_k, new_v = block.attention.project(block.attention_norm(hidden), positions)
            layer_cache = cache.append(index, new_k, new_v)
            hidden = hidden + block.attention.attend(q, layer_cache.keys, layer_cache.values,
                                                      positions, all_positions, query_segments, all_segments)
            expert_output, _, _ = block.moe(block.moe_norm(hidden), positions, balancing_bias[index])
            hidden = hidden + expert_output
        logits = model.lm_head(model.final_norm(hidden[-1:]))[0]
        cache.commit()
        return logits
    except Exception:
        cache.abort()
        raise
