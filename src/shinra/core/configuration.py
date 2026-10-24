"""Strict YAML decoding and immutable configuration values; no model imports."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any

import yaml
from yaml.nodes import MappingNode, ScalarNode, SequenceNode
from yaml.tokens import AliasToken, AnchorToken, DirectiveToken, TagToken


class ConfigurationError(ValueError):
    """Invalid, unknown or unapproved configuration with a field path."""

    def __init__(self, path: str, reason: str):
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


_INTEGER = re.compile(r"-?(?:0|[1-9][0-9]*)\Z")
_FLOAT = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+(?:[eE][+-]?[0-9]+)?|[eE][+-]?[0-9]+)\Z")


def _decode(node: Any, path: str) -> Any:
    if isinstance(node, MappingNode):
        result: dict[str, Any] = {}
        for key_node, value_node in node.value:
            if not isinstance(key_node, ScalarNode):
                raise ConfigurationError(path, "mapping keys must be strings")
            key = key_node.value
            if key == "<<" or key in result:
                raise ConfigurationError(f"{path}.{key}", "duplicate or merge key")
            if not key or not re.fullmatch(r"[a-z][a-z0-9_]*", key):
                raise ConfigurationError(path, f"invalid key {key!r}")
            result[key] = _decode(value_node, f"{path}.{key}")
        return result
    if isinstance(node, SequenceNode):
        return [_decode(child, f"{path}[{i}]") for i, child in enumerate(node.value)]
    if not isinstance(node, ScalarNode):
        raise ConfigurationError(path, "unsupported YAML node")
    value = node.value
    if node.style is not None:
        return value
    if value == "null":
        return None
    if value in ("true", "false"):
        return value == "true"
    if _INTEGER.fullmatch(value):
        return int(value)
    if _FLOAT.fullmatch(value):
        number = float(value)
        if not math.isfinite(number):
            raise ConfigurationError(path, "number must be finite")
        return number
    if value.lower() in (".nan", ".inf", "+.inf", "-.inf", "nan", "inf", "infinity"):
        raise ConfigurationError(path, "nonfinite number is forbidden")
    return value


def read_configuration(path: str | Path) -> dict[str, Any]:
    """Read one YAML document. Returned mapping still requires schema validation."""
    source = Path(path)
    if source.name != "shinra_40b_moe.yaml":
        raise ConfigurationError("config", "expected canonical shinra_40b_moe.yaml")
    text = source.read_text(encoding="utf-8", errors="strict")
    try:
        for token in yaml.scan(text, Loader=yaml.BaseLoader):
            if isinstance(token, (AliasToken, AnchorToken, DirectiveToken, TagToken)):
                raise ConfigurationError("config", "anchors, aliases, directives and explicit tags are forbidden")
        nodes = list(yaml.compose_all(text, Loader=yaml.BaseLoader))
    except yaml.YAMLError as exc:
        raise ConfigurationError("config", str(exc)) from exc
    if len(nodes) != 1 or not isinstance(nodes[0], MappingNode):
        raise ConfigurationError("config", "exactly one mapping document is required")
    return _decode(nodes[0], "config")


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class Configuration:
    """Immutable snapshot; validate_architecture must precede operational use."""

    values: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _freeze(self.values))

    def get(self, dotted_path: str) -> Any:
        value: Any = self.values
        for part in dotted_path.split("."):
            if not isinstance(value, Mapping) or part not in value:
                raise ConfigurationError(dotted_path, "missing field")
            value = value[part]
        return value

    def canonical_bytes(self) -> bytes:
        return json.dumps(_plain(self.values), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")

    @property
    def identity(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @property
    def architecture_identity(self) -> str:
        names = ("schema_version", "model", "attention", "rope", "moe", "router", "tokenizer")
        subset = {name: _plain(self.values[name]) for name in names}
        # Artifact locations are operational; tokenizer content identity is not.
        subset["tokenizer"].pop("artifact_path")
        encoded = json.dumps(subset, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
