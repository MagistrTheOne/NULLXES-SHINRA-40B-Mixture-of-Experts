"""Initialization recipes and logical parameter inventory, without allocation."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from collections.abc import Iterator

from .configuration import Configuration, ConfigurationError
from .validation import require_operation, validate_architecture


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str
    role: str

    @property
    def numel(self) -> int:
        return math.prod(self.shape)


@dataclass(frozen=True, slots=True)
class InitializationRecipe:
    approval_reference: str
    seed: int
    distribution: str
    std: float
    residual_scale: float

    def __post_init__(self) -> None:
        if not isinstance(self.approval_reference, str) or not self.approval_reference.strip():
            raise ConfigurationError("initialization.approval_reference", "required")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ConfigurationError("initialization.seed", "expected integer in [0, 2**63)")
        if self.distribution != "normal":
            raise ConfigurationError("initialization.distribution", "explicit normal recipe required")
        for name in ("std", "residual_scale"):
            value = getattr(self, name)
            if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
                raise ConfigurationError(f"initialization.{name}", "finite positive value required")

    @classmethod
    def from_configuration(cls, config: Configuration) -> InitializationRecipe:
        require_operation(config, "initialize")
        values = config.get("model.initialization")
        if values["distribution"] != "normal":
            raise ConfigurationError("model.initialization.distribution", "only explicitly selected normal is supported")
        if values["seed"] >= 2**63:
            raise ConfigurationError("model.initialization.seed", "must fit signed 63-bit seed range")
        return cls(values["approval_reference"], values["seed"], values["distribution"],
                   float(values["std"]), float(values["residual_scale"]))

    def tensor_seed(self, logical_name: str) -> int:
        if not logical_name:
            raise ConfigurationError("initialization.logical_name", "must be nonempty")
        payload = f"shinra.init.v1:{self.seed}:{logical_name}".encode("utf-8")
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 2**63

    def scale_for(self, parameter: ParameterSpec) -> float:
        if parameter.role == "norm":
            return 1.0
        return self.std * (self.residual_scale if parameter.role == "residual_projection" else 1.0)


def parameter_specs(config: Configuration) -> Iterator[ParameterSpec]:
    """Yield every trainable logical tensor in the fixed model, no tensor creation."""
    validate_architecture(config)
    m = config.get("model")
    d, f, v = m["hidden_size"], m["intermediate_size"], m["vocab_size"]
    q, k, h = m["num_attention_heads"], m["num_key_value_heads"], m["head_dim"]
    yield ParameterSpec("embeddings.weight", (v, d), "bfloat16", "embedding")
    for layer in range(m["num_hidden_layers"]):
        prefix = f"layers.{layer}"
        for name in ("attention_norm", "moe_norm"):
            yield ParameterSpec(f"{prefix}.{name}.weight", (d,), "bfloat16", "norm")
        for name, shape, role in (
            ("q_proj", (q*h, d), "projection"), ("k_proj", (k*h, d), "projection"),
            ("v_proj", (k*h, d), "projection"), ("o_proj", (d, q*h), "residual_projection"),
        ):
            yield ParameterSpec(f"{prefix}.attention.{name}.weight", shape, "bfloat16", role)
        yield ParameterSpec(f"{prefix}.attention.q_norm.weight", (q, h), "bfloat16", "norm")
        yield ParameterSpec(f"{prefix}.attention.k_norm.weight", (k, h), "bfloat16", "norm")
        yield ParameterSpec(f"{prefix}.moe.router.weight", (m["num_routed_experts"], d), "float32", "router")
        for expert in range(m["num_routed_experts"]):
            for name, shape, role in (("gate", (f,d), "projection"), ("up", (f,d), "projection"),
                                      ("down", (d,f), "residual_projection")):
                yield ParameterSpec(f"{prefix}.moe.experts.{expert}.{name}.weight", shape, "bfloat16", role)
    yield ParameterSpec("final_norm.weight", (d,), "bfloat16", "norm")
    yield ParameterSpec("lm_head.weight", (v,d), "bfloat16", "output")


def initialize_parameter(spec: ParameterSpec, recipe: InitializationRecipe, *, device: str):
    """Explicit runtime allocation; never called by import or configuration parsing.

    Full logical tensors only. Partition-invariant initialization is not claimed.
    """
    import torch

    if not device or device == "meta":
        raise ConfigurationError("initialization.device", "explicit real device required")
    dtypes = {"bfloat16": torch.bfloat16, "float32": torch.float32}
    if spec.dtype not in dtypes or any(type(n) is not int or n <= 0 for n in spec.shape):
        raise ConfigurationError(spec.name, "invalid parameter specification")
    if spec.role not in ("norm", "embedding", "projection", "residual_projection", "router", "output"):
        raise ConfigurationError(spec.name, "unknown initialization role")
    target = torch.device(device)
    generator = torch.Generator(device=target)
    generator.manual_seed(recipe.tensor_seed(spec.name))
    value = torch.empty(spec.shape, device=target, dtype=dtypes[spec.dtype])
    with torch.no_grad():
        if spec.role == "norm":
            value.fill_(1.0)
        else:
            value.normal_(mean=0.0, std=recipe.scale_for(spec), generator=generator)
    return torch.nn.Parameter(value)
