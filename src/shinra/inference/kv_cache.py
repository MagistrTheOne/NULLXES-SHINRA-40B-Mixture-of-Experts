"""Transactional per-sequence GQA cache; committed keys are never rotated twice."""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class LayerCache:
    keys: torch.Tensor
    values: torch.Tensor


class KVCache:
    def __init__(self, *, checkpoint_identity: str, rope_identity: str, max_positions: int, layers: int):
        if not checkpoint_identity or not rope_identity or max_positions != 262144 or layers != 48:
            raise ValueError("explicit SHINRA cache identities and dimensions required")
        self.checkpoint_identity, self.rope_identity = checkpoint_identity, rope_identity
        self.max_positions, self.layers = max_positions, layers
        self.committed: tuple[LayerCache, ...] = ()
        self.length = 0
        self.pending: dict[int, LayerCache] | None = None
        self.append_length = 0

    def begin(self, tokens: int):
        if self.pending is not None or type(tokens) is not int or tokens <= 0 or self.length + tokens > self.max_positions:
            raise ValueError("invalid cache append or context boundary exceeded")
        self.pending, self.append_length = {}, tokens

    def append(self, layer: int, keys: torch.Tensor, values: torch.Tensor) -> LayerCache:
        if self.pending is None or layer in self.pending or not 0 <= layer < self.layers:
            raise RuntimeError("invalid cache layer transaction")
        if keys.shape != (self.append_length,8,128) or values.shape != keys.shape or keys.dtype != torch.bfloat16 or values.dtype != keys.dtype or values.device != keys.device:
            raise ValueError("cache requires same-device BF16 [S,8,128] keys/values")
        if keys.requires_grad or values.requires_grad:
            raise ValueError("KV cache is inference-only")
        if self.length:
            previous = self.committed[layer]
            if previous.keys.device != keys.device:
                raise ValueError("cache device cannot change within a sequence")
            keys = torch.cat((previous.keys, keys), dim=0)
            values = torch.cat((previous.values, values), dim=0)
        result = LayerCache(keys, values)
        self.pending[layer] = result
        return result

    def commit(self):
        if self.pending is None or set(self.pending) != set(range(self.layers)):
            raise RuntimeError("all 48 cache layers must complete before commit")
        self.committed = tuple(self.pending[i] for i in range(self.layers))
        self.length += self.append_length
        self.pending, self.append_length = None, 0

    def abort(self):
        self.pending, self.append_length = None, 0
