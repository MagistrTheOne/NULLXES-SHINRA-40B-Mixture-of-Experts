"""Independent full-vocabulary language-model output projection."""
import torch
from torch import nn
from torch.nn import functional as F

from shinra.core.configuration import Configuration
from shinra.core.initialization import InitializationRecipe, ParameterSpec, initialize_parameter
from shinra.core.validation import validate_architecture


class LanguageModelHead(nn.Module):
    def __init__(self, config: Configuration, recipe: InitializationRecipe, *, device: str, tensor_execution=None):
        super().__init__()
        validate_architecture(config)
        self.tensor_execution = tensor_execution
        self.hidden_size = config.get("model.hidden_size")
        self.weight = initialize_parameter(ParameterSpec("lm_head.weight", (
            config.get("model.vocab_size"), self.hidden_size), "bfloat16", "output"), recipe, device=device, tensor_execution=tensor_execution)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.shape[-1] != self.hidden_size or hidden.dtype != self.weight.dtype or hidden.device != self.weight.device:
            raise ValueError("LM head hidden shape/dtype/device mismatch")
        if not bool(torch.isfinite(hidden).all()):
            raise FloatingPointError("nonfinite LM head input")
        result = (F.linear(hidden, self.weight) if self.tensor_execution is None
                  else self.tensor_execution.linear(hidden, self.weight))
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError("nonfinite LM head logits")
        return result
