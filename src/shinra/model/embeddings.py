"""Untied native input token embeddings; PAD row is not silently frozen."""
import torch
from torch import nn
from torch.nn import functional as F

from shinra.core.configuration import Configuration
from shinra.core.initialization import InitializationRecipe, ParameterSpec, initialize_parameter
from shinra.core.validation import validate_architecture


class TokenEmbeddings(nn.Module):
    def __init__(self, config: Configuration, recipe: InitializationRecipe, *, device: str):
        super().__init__()
        validate_architecture(config)
        self.vocab_size = config.get("model.vocab_size")
        self.weight = initialize_parameter(ParameterSpec("embeddings.weight", (
            self.vocab_size, config.get("model.hidden_size")), "bfloat16", "embedding"), recipe, device=device)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.dtype != torch.int64 or token_ids.device != self.weight.device:
            raise ValueError("embedding IDs must be int64 on the weight device")
        if bool(((token_ids < 0) | (token_ids >= self.vocab_size)).any()):
            raise ValueError("token ID outside vocabulary")
        return F.embedding(token_ids, self.weight, padding_idx=None)
