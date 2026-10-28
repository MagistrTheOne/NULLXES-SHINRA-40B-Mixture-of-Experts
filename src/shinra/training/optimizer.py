"""Explicit AdamW with FP32 master parameters and summed FP32 gradients."""
import torch
from shinra.core.configuration import Configuration, ConfigurationError
from shinra.core.validation import validate_architecture
from .precision import validate_model_precision


class MasterOptimizer:
    def __init__(self, model, config: Configuration):
        validate_architecture(config)
        validate_model_precision(model)
        required = ("optimizer", "optimizer_precision", "parameter_group_policy", "learning_rate",
                    "beta1", "beta2", "epsilon", "weight_decay")
        for name in required:
            if config.get(f"training.{name}") is None:
                raise ConfigurationError(f"training.{name}", "UNKNOWN")
        if config.get("training.optimizer") != "adamw" or config.get("training.optimizer_precision") != "fp32_master" or config.get("training.parameter_group_policy") != "uniform_decay":
            raise ConfigurationError("training", "explicit adamw/fp32_master/uniform_decay selection required")
        self.parameters = dict(model.named_parameters())
        self.masters = {name: torch.nn.Parameter(parameter.detach().float().clone())
                        for name, parameter in self.parameters.items()}
        self.optimizer = torch.optim.AdamW(
            list(self.masters.values()), lr=config.get("training.learning_rate"),
            betas=(config.get("training.beta1"), config.get("training.beta2")),
            eps=config.get("training.epsilon"), weight_decay=config.get("training.weight_decay"),
            amsgrad=False, maximize=False, foreach=False, capturable=False, differentiable=False, fused=False)
        self.accumulated: dict[str, torch.Tensor] = {}

    def clear(self):
        self.optimizer.zero_grad(set_to_none=True)
        for parameter in self.parameters.values():
            parameter.grad = None
        self.accumulated.clear()

    def accumulate(self):
        for name, parameter in self.parameters.items():
            if parameter.grad is None:
                continue
            grad = parameter.grad.detach().float()
            if not bool(torch.isfinite(grad).all()):
                raise FloatingPointError(f"nonfinite gradient: {name}")
            if name in self.accumulated:
                self.accumulated[name].add_(grad)
            else:
                self.accumulated[name] = grad.clone()
            parameter.grad = None

    def install_gradients(self, gradients: dict[str, torch.Tensor], *, global_tokens: int, max_norm: float,
                          group=None, owned_names=frozenset(), unique_names=None):
        if type(global_tokens) is not int or global_tokens <= 0:
            raise ValueError("global loss target count must be positive")
        if set(gradients) != set(self.masters):
            raise ValueError("synchronized gradient inventory mismatch")
        for name, master in self.masters.items():
            value = gradients[name]
            if value.dtype != torch.float32 or value.shape != master.shape or value.device != master.device:
                raise ValueError("FP32 gradient shape/device mismatch")
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError("nonfinite synchronized gradient: " + name)
            master.grad = value / global_tokens
        if group is not None:
            # Replicated non-expert tensors count once, owner expert tensors
            # count once globally. Never count every replica in clipping norm.
            device = next(iter(self.masters.values())).device
            square = torch.zeros((), dtype=torch.float64, device=device)
            for name, master in self.masters.items():
                selected = name in unique_names if unique_names is not None else (name in owned_names or group.rank == 0)
                if selected:
                    square += master.grad.double().square().sum()
            norm = group.sum(square).sqrt()
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError("nonfinite global EP gradient norm")
            scale = torch.clamp(max_norm / (norm + 1e-6), max=1.0)
            for master in self.masters.values():
                master.grad.mul_(scale.to(master.grad.dtype))
            return norm
        return torch.nn.utils.clip_grad_norm_(list(self.masters.values()), max_norm=max_norm,
                                             norm_type=2.0, error_if_nonfinite=True, foreach=False)

    def step(self, learning_rate: float):
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        self.optimizer.step()
        with torch.no_grad():
            for name, parameter in self.parameters.items():
                master = self.masters[name]
                if not bool(torch.isfinite(master).all()):
                    raise FloatingPointError(f"nonfinite updated master: {name}; recovery required")
                stored = master.to(parameter.dtype)
                if not bool(torch.isfinite(stored).all()):
                    raise FloatingPointError(f"storage dtype overflow: {name}; recovery required")
                parameter.copy_(stored)
