"""Closed schema, fixed D02 architecture and operation-specific prerequisites."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

from .configuration import Configuration, ConfigurationError, read_configuration


# These are rejection constraints, never values injected into configuration.
FIXED_MODEL = {
    "name": "SHINRA-40B-MoE", "architecture": "causal_decoder_moe",
    "vocab_size": 131072, "hidden_size": 2560, "num_hidden_layers": 48,
    "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
    "intermediate_size": 800, "num_routed_experts": 128, "num_experts_per_tok": 8,
    "num_shared_experts": 0, "max_position_embeddings": 262144,
    "tie_word_embeddings": False, "attention_bias": False, "expert_bias": False,
    "router_bias": False, "hidden_act": "swiglu", "rms_norm_eps": 1e-6,
}


@dataclass(frozen=True, slots=True)
class Field:
    kind: str
    nullable: bool = False
    choices: tuple[Any, ...] = ()
    minimum: float | None = None
    maximum: float | None = None


def exact(value: Any) -> Field:
    kind = {str: "str", int: "int", bool: "bool", float: "number"}[type(value)]
    return Field(kind, choices=(value,))


TEXT = Field("str", nullable=True)
POSITIVE = Field("number", nullable=True, minimum=0)
COUNT = Field("int", nullable=True, minimum=1)
NONNEGATIVE = Field("int", nullable=True, minimum=0)
SCHEMA = {
    "schema_version": exact(1),
    "model": {**{key: exact(value) for key, value in FIXED_MODEL.items()},
              "initialization": {"approval_reference": TEXT, "seed": NONNEGATIVE,
                                 "distribution": TEXT, "std": POSITIVE, "residual_scale": POSITIVE}},
    "attention": {"type": exact("gqa"), "qk_norm": exact("per_head_rms"),
                  "qk_norm_order": exact("before_rope"), "scale": exact("inverse_sqrt_head_dim"),
                  "causal": exact(True), "document_aware": exact(True),
                  "sliding_window": Field("bool", nullable=True, choices=(False,)), "backend": TEXT,
                  "dropout": Field("number", True, minimum=0, maximum=1),
                  "query_chunk_tokens": COUNT, "key_chunk_tokens": COUNT},
    "rope": {"approval_reference": TEXT, "theta": POSITIVE, "pairing": exact("adjacent"),
             "scaling": exact("identity"), "phase_dtype": exact("float64")},
    "moe": {"dispatch": exact("stable"), "combine_dtype": exact("float32"),
            "overflow": exact("chunk_same_expert"), "chunk_tokens": COUNT},
    "router": {"weight_dtype": exact("float32"), "logits_dtype": exact("float32"),
               "tie_break": exact("expert_id_ascending"), "normalization": exact("selected_logits_softmax"),
               "balancing": {"approval_reference": TEXT, "algorithm": exact("bounded_sign"),
                             "learning_rate": POSITIVE, "bound": POSITIVE}},
    "tokenizer": {"algorithm": exact("native_byte_bpe"), "contract": exact("shinra.bytebpe.v1"),
                  "approval_reference": TEXT, "special_token_ids": Field("special_ids", True),
                  "byte_start_id": NONNEGATIVE, "learned_start_id": COUNT,
                  "max_token_bytes": COUNT, "min_pair_frequency": COUNT,
                  "artifact_path": TEXT, "artifact_identity": Field("digest", True)},
    "data": {"approval_reference": TEXT, "manifest_path": TEXT,
             "manifest_identity": Field("digest", True), "language_policy": TEXT},
    "training": {"precision": exact("bf16"), "optimizer": TEXT, "learning_rate": POSITIVE,
                 "optimizer_precision": Field("str", True, ("fp32_master",)),
                 "parameter_group_policy": Field("str", True, ("uniform_decay",)),
                 "gradient_clip_norm_type": Field("str", True, ("l2",)),
                 "beta1": Field("number", True, minimum=0, maximum=1),
                 "beta2": Field("number", True, minimum=0, maximum=1),
                 "epsilon": POSITIVE, "weight_decay": POSITIVE, "scheduler": TEXT,
                 "warmup_steps": NONNEGATIVE, "total_steps": COUNT, "microbatch_tokens": COUNT,
                 "accumulation_steps": COUNT, "max_grad_norm": POSITIVE, "token_budget": COUNT,
                 "recomputation": Field("bool", True)},
    "distributed": {"backend": TEXT, "ownership_manifest": TEXT, "timeout_seconds": POSITIVE},
    "checkpoint": {"format": exact("safetensors"), "storage_backend": TEXT,
                   "directory": TEXT, "resume_manifest": TEXT},
    "inference": {"checkpoint_manifest": TEXT, "sampling": Field("str", True, ("greedy", "categorical")),
                  "temperature": POSITIVE, "top_k": NONNEGATIVE,
                  "top_p": Field("number", True, minimum=0, maximum=1),
                  "max_new_tokens": COUNT, "stop_token_ids": Field("ids", True), "seed": NONNEGATIVE},
}

SPECIAL_NAMES = ("pad", "bos", "eos", "doc_sep", "role_system", "role_user", "role_assistant",
                 "role_tool", "header_end", "message_end", "turn_end", "fim_prefix",
                 "fim_suffix", "fim_middle", "fim_end") + tuple(f"reserved_{i}" for i in range(15, 32))


def _check(value: Any, schema: Any, path: str) -> None:
    if isinstance(schema, dict):
        if not isinstance(value, Mapping):
            raise ConfigurationError(path, "expected mapping")
        missing = set(schema) - set(value)
        unknown = set(value) - set(schema)
        if missing or unknown:
            raise ConfigurationError(path, f"missing={sorted(missing)}, unknown={sorted(unknown)}")
        for key, rule in schema.items():
            _check(value[key], rule, f"{path}.{key}")
        return
    rule: Field = schema
    if value is None:
        if rule.nullable or rule.kind == "null":
            return
        raise ConfigurationError(path, "required value is null")
    if rule.kind == "special_ids":
        _check(value, {name: Field("int", minimum=0, maximum=131071) for name in SPECIAL_NAMES}, path)
        if len(set(value.values())) != len(value):
            raise ConfigurationError(path, "special IDs must be unique")
        return
    if rule.kind == "ids":
        if not isinstance(value, (list, tuple)):
            raise ConfigurationError(path, "expected token ID sequence")
        for i, token in enumerate(value):
            _check(token, Field("int", minimum=0, maximum=131071), f"{path}[{i}]")
        if len(set(value)) != len(value):
            raise ConfigurationError(path, "duplicate token IDs")
        return
    expected = {"str": (str,), "digest": (str,), "int": (int,), "number": (int, float), "bool": (bool,), "null": ()}
    if type(value) not in expected[rule.kind]:
        raise ConfigurationError(path, f"expected {rule.kind}, got {type(value).__name__}")
    if isinstance(value, str) and not value.strip():
        raise ConfigurationError(path, "empty string; use explicit null for UNKNOWN")
    if rule.kind == "digest" and (len(value) != 64 or any(c not in "0123456789abcdef" for c in value)):
        raise ConfigurationError(path, "expected lowercase SHA-256")
    if rule.kind in ("number", "int"):
        if type(value) is float and not math.isfinite(value):
            raise ConfigurationError(path, "must be finite")
        if rule.minimum is not None and value < rule.minimum:
            raise ConfigurationError(path, f"must be >= {rule.minimum}")
        if rule.maximum is not None and value > rule.maximum:
            raise ConfigurationError(path, f"must be <= {rule.maximum}")
    if rule.choices and value not in rule.choices:
        raise ConfigurationError(path, f"expected one of {rule.choices!r}")


@dataclass(frozen=True, slots=True)
class ParameterAccounting:
    total: int
    conventional_active: int
    all_experts: int
    active_experts: int
    non_expert: int
    accessed_per_token: int
    arithmetic_per_token: int


def parameter_accounting(config: Configuration) -> ParameterAccounting:
    """Pure integer D02 algebra; no tensor allocation or framework import."""
    m = config.get("model")
    d, v, layers = m["hidden_size"], m["vocab_size"], m["num_hidden_layers"]
    q, k, h = m["num_attention_heads"], m["num_key_value_heads"], m["head_dim"]
    e, selected, f = m["num_routed_experts"], m["num_experts_per_tok"], m["intermediate_size"]
    expert = 3 * d * f
    non_expert = 2 * v * d + layers * (2*d*(q+k)*h + d*e + 2*d + (q+k)*h) + d
    active = non_expert + layers * selected * expert
    return ParameterAccounting(non_expert + layers*e*expert, active, layers*e*expert,
                               layers*selected*expert, non_expert, active-v*d+d, active-v*d)


def validate_architecture(config: Configuration) -> ParameterAccounting:
    _check(config.values, SCHEMA, "config")
    count = parameter_accounting(config)
    if (count.total, count.conventional_active) != (39694338560, 4304898560):
        raise ConfigurationError("model", "D02 parameter accounting mismatch")
    special = config.get("tokenizer.special_token_ids")
    byte_start = config.get("tokenizer.byte_start_id")
    learned = config.get("tokenizer.learned_start_id")
    if special is not None and tuple(special[name] for name in SPECIAL_NAMES) != tuple(range(32)):
        raise ConfigurationError("tokenizer.special_token_ids", "D05 candidate requires IDs 0..31 in registry order")
    if byte_start is not None and byte_start != 32:
        raise ConfigurationError("tokenizer.byte_start_id", "D05 candidate requires 32")
    if learned is not None and learned != 288:
        raise ConfigurationError("tokenizer.learned_start_id", "D05 candidate requires 288")
    for field in ("training.beta1", "training.beta2", "attention.dropout"):
        value = config.get(field)
        if value is not None and value >= 1:
            raise ConfigurationError(field, "must be < 1")
    for field in ("rope.theta", "model.initialization.std", "model.initialization.residual_scale",
                  "router.balancing.learning_rate", "router.balancing.bound", "training.learning_rate",
                  "training.epsilon", "training.max_grad_norm", "distributed.timeout_seconds",
                  "inference.temperature", "inference.top_p"):
        value = config.get(field)
        if value is not None and value <= 0:
            raise ConfigurationError(field, "must be positive")
    theta = config.get("rope.theta")
    if theta is not None and theta != 1000000:
        raise ConfigurationError("rope.theta", "D04 candidate is 1000000; a different base needs a contract revision")
    warmup, total = config.get("training.warmup_steps"), config.get("training.total_steps")
    if warmup is not None and total is not None and warmup >= total:
        raise ConfigurationError("training.warmup_steps", "must be less than total_steps")
    if config.get("inference.top_k") is not None and config.get("inference.top_k") > config.get("model.vocab_size"):
        raise ConfigurationError("inference.top_k", "exceeds vocabulary")
    if config.get("inference.max_new_tokens") is not None and config.get("inference.max_new_tokens") > config.get("model.max_position_embeddings"):
        raise ConfigurationError("inference.max_new_tokens", "exceeds context target")
    return count


def load_configuration(path: str | Path) -> Configuration:
    config = Configuration(read_configuration(path))
    validate_architecture(config)
    return config


def operation_blockers(config: Configuration, operation: str) -> tuple[str, ...]:
    """Configuration prerequisites only: this is not execution authorization."""
    validate_architecture(config)
    model = ("rope.approval_reference", "rope.theta", "attention.sliding_window",
             "attention.backend", "attention.dropout",
             "attention.query_chunk_tokens", "attention.key_chunk_tokens", "moe.chunk_tokens")
    tokenizer = tuple("tokenizer." + name for name in (
        "approval_reference", "special_token_ids", "byte_start_id", "learned_start_id",
        "max_token_bytes", "min_pair_frequency"))
    artifact = ("tokenizer.artifact_path", "tokenizer.artifact_identity")
    corpus = ("data.approval_reference", "data.manifest_path", "data.manifest_identity", "data.language_policy")
    initialization = tuple("model.initialization." + name for name in (
        "approval_reference", "seed", "distribution", "std", "residual_scale"))
    training = tuple("training." + name for name in SCHEMA["training"])
    distributed = tuple("distributed." + name for name in SCHEMA["distributed"])
    operations = {
        "architecture": (), "initialize": initialization,
        "model": model, "tokenizer": tokenizer, "tokenizer_train": tokenizer + corpus,
        "ingest": corpus, "pack": corpus + tokenizer + artifact,
        "train": model + tokenizer + artifact + corpus + training + distributed + (
            "router.balancing.approval_reference", "router.balancing.learning_rate", "router.balancing.bound",
            "checkpoint.storage_backend", "checkpoint.directory"),
        "checkpoint": ("checkpoint.storage_backend", "checkpoint.directory"),
        "resume": ("checkpoint.resume_manifest",) + distributed + artifact + corpus,
        "infer": model + tokenizer + artifact + ("inference.checkpoint_manifest", "inference.sampling",
                                                  "inference.max_new_tokens", "inference.stop_token_ids"),
    }
    if operation not in operations:
        raise ConfigurationError("operation", f"unknown operation {operation!r}")
    fields = operations[operation]
    if operation == "infer" and config.get("inference.sampling") == "categorical":
        fields += ("inference.temperature", "inference.top_k", "inference.top_p", "inference.seed")
    return tuple(f"{field}: UNKNOWN / not approved" for field in fields if config.get(field) is None)


def require_operation(config: Configuration, operation: str) -> None:
    blockers = operation_blockers(config, operation)
    if blockers:
        raise ConfigurationError(operation, "; ".join(blockers))
