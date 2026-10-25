"""Versioned, strict JSON metadata for Safetensors checkpoint transactions."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


SCHEMA_VERSION = 1
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_TRANSACTION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_GROUP = re.compile(r"[a-z][a-z0-9_]{0,31}\Z")
_DTYPES = frozenset(
    ("BOOL", "U8", "I8", "I16", "I32", "I64", "F16", "BF16", "F32", "F64",
     "F8_E4M3", "F8_E5M2")
)
_DTYPE_SIZES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2, "F16": 2, "BF16": 2, "I32": 4, "F32": 4, "I64": 8, "F64": 8,
}


class CheckpointError(ValueError):
    """Malformed, incomplete, unsupported, or inconsistent checkpoint state."""


def _require_keys(value: Any, keys: set[str], path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise CheckpointError(f"{path}: expected exactly keys {sorted(keys)}")
    if any(not isinstance(key, str) for key in value):
        raise CheckpointError(f"{path}: keys must be strings")
    return value


def _nonempty(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise CheckpointError(f"{path}: expected nonempty trimmed string")
    return value


def _digest(value: Any, path: str) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise CheckpointError(f"{path}: expected lowercase SHA-256")
    return value


def _safe_filename(value: Any, path: str, suffix: str) -> str:
    name = _nonempty(value, path)
    if (Path(name).name != name or "/" in name or "\\" in name
            or name in (".", "..") or not name.endswith(suffix)):
        raise CheckpointError(f"{path}: expected a plain filename ending in {suffix}")
    return name


def _json_value(value: Any, path: str) -> Any:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise CheckpointError(f"{path}: nonfinite JSON number")
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise CheckpointError(f"{path}: metadata keys must be nonempty strings")
            result[key] = _json_value(item, f"{path}.{key}")
        return result
    raise CheckpointError(f"{path}: unsupported metadata type {type(value).__name__}")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CheckpointError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise CheckpointError(f"nonfinite JSON value {value}")


@dataclass(frozen=True, slots=True)
class TensorRecord:
    name: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int
    shard: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "shape": list(self.shape), "dtype": self.dtype,
                "nbytes": self.nbytes, "shard": self.shard}

    @classmethod
    def from_dict(cls, value: Any, path: str) -> TensorRecord:
        item = _require_keys(value, {"name", "shape", "dtype", "nbytes", "shard"}, path)
        name = _nonempty(item["name"], f"{path}.name")
        shape_value = item["shape"]
        if not isinstance(shape_value, list) or any(type(dim) is not int or dim < 0 for dim in shape_value):
            raise CheckpointError(f"{path}.shape: expected nonnegative integer dimensions")
        dtype = _nonempty(item["dtype"], f"{path}.dtype")
        if dtype not in _DTYPES:
            raise CheckpointError(f"{path}.dtype: unsupported Safetensors dtype {dtype!r}")
        if type(item["nbytes"]) is not int or item["nbytes"] < 0:
            raise CheckpointError(f"{path}.nbytes: expected nonnegative integer")
        expected_nbytes = math.prod(shape_value) * _DTYPE_SIZES[dtype]
        if item["nbytes"] != expected_nbytes:
            raise CheckpointError(f"{path}.nbytes: inconsistent with shape and dtype")
        shard = _safe_filename(item["shard"], f"{path}.shard", ".safetensors")
        return cls(name, tuple(shape_value), dtype, item["nbytes"], shard)


@dataclass(frozen=True, slots=True)
class FileRecord:
    name: str
    size: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "size": self.size, "sha256": self.sha256}

    @classmethod
    def from_dict(cls, value: Any, path: str) -> FileRecord:
        item = _require_keys(value, {"name", "size", "sha256"}, path)
        name = _nonempty(item["name"], f"{path}.name")
        if Path(name).name != name or "/" in name or "\\" in name or name in (".", ".."):
            raise CheckpointError(f"{path}.name: expected a plain filename")
        if type(item["size"]) is not int or item["size"] < 0:
            raise CheckpointError(f"{path}.size: expected nonnegative integer")
        return cls(name, item["size"], _digest(item["sha256"], f"{path}.sha256"))


@dataclass(frozen=True, slots=True)
class TensorGroup:
    name: str
    index_file: str
    records: tuple[TensorRecord, ...]
    files: tuple[FileRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"index_file": self.index_file,
                "records": [record.to_dict() for record in self.records],
                "files": [record.to_dict() for record in self.files]}

    @classmethod
    def from_dict(cls, name: str, value: Any, path: str) -> TensorGroup:
        if not isinstance(name, str) or not _GROUP.fullmatch(name):
            raise CheckpointError(f"{path}: invalid tensor group name")
        item = _require_keys(value, {"index_file", "records", "files"}, path)
        index_file = _safe_filename(item["index_file"], f"{path}.index_file", ".index.json")
        if not isinstance(item["records"], list):
            raise CheckpointError(f"{path}.records: expected a list")
        if not isinstance(item["files"], list) or not item["files"]:
            raise CheckpointError(f"{path}.files: expected a nonempty list including the index")
        records = tuple(TensorRecord.from_dict(entry, f"{path}.records[{i}]")
                        for i, entry in enumerate(item["records"]))
        files = tuple(FileRecord.from_dict(entry, f"{path}.files[{i}]")
                      for i, entry in enumerate(item["files"]))
        if name == "model" and not records:
            raise CheckpointError(f"{path}.records: model tensor inventory cannot be empty")
        names = [record.name for record in records]
        if len(names) != len(set(names)):
            raise CheckpointError(f"{path}.records: duplicate tensor names")
        file_names = [record.name for record in files]
        if len(file_names) != len(set(file_names)) or index_file not in file_names:
            raise CheckpointError(f"{path}.files: duplicate file names")
        payload_files = set(file_names) - {index_file}
        if index_file != f"{name}.safetensors.index.json":
            raise CheckpointError(f"{path}.index_file: noncanonical Safetensors index name")
        if index_file not in file_names or {record.shard for record in records} != payload_files:
            raise CheckpointError(f"{path}: tensor shard ownership does not match file inventory")
        shard_pattern = re.compile(rf"{re.escape(name)}-(\d{{5,}})-of-(\d{{5,}})\.safetensors\Z")
        shard_parts = []
        for filename in payload_files:
            match = shard_pattern.fullmatch(filename)
            if match is None:
                raise CheckpointError(f"{path}: noncanonical Safetensors shard name {filename!r}")
            shard_parts.append((int(match.group(1)), int(match.group(2))))
        if shard_parts:
            total = shard_parts[0][1]
            if (any(part[1] != total for part in shard_parts)
                    or sorted(part[0] for part in shard_parts) != list(range(1, total + 1))
                    or total != len(shard_parts)):
                raise CheckpointError(f"{path}: incomplete or inconsistent shard numbering")
        elif name == "model":
            raise CheckpointError(f"{path}: model group requires at least one Safetensors shard")
        return cls(name, index_file, records, files)


@dataclass(frozen=True, slots=True)
class CheckpointManifest:
    transaction_id: str
    purpose: str
    successful_step: int
    config_identity: str
    architecture_identity: str
    required_groups: tuple[str, ...]
    groups: Mapping[str, TensorGroup]
    metadata: Mapping[str, Any]
    schema_version: int = SCHEMA_VERSION
    format: str = "safetensors"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "format": self.format,
            "transaction_id": self.transaction_id,
            "purpose": self.purpose,
            "successful_step": self.successful_step,
            "config_identity": self.config_identity,
            "architecture_identity": self.architecture_identity,
            "required_groups": list(self.required_groups),
            "groups": {name: group.to_dict() for name, group in sorted(self.groups.items())},
            "metadata": _json_value(self.metadata, "metadata"),
        }

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION or self.format != "safetensors":
            raise CheckpointError("unsupported checkpoint schema or tensor format")
        if not isinstance(self.transaction_id, str) or not _TRANSACTION_ID.fullmatch(self.transaction_id):
            raise CheckpointError("invalid transaction_id")
        if self.purpose not in ("model", "training_resume"):
            raise CheckpointError("purpose must be model or training_resume")
        if type(self.successful_step) is not int or self.successful_step < 0:
            raise CheckpointError("successful_step must be a nonnegative integer")
        _digest(self.config_identity, "config_identity")
        _digest(self.architecture_identity, "architecture_identity")
        if not isinstance(self.groups, Mapping) or "model" not in self.groups:
            raise CheckpointError("model tensor group is required")
        if any(not isinstance(name, str) or not isinstance(group, TensorGroup)
               for name, group in self.groups.items()):
            raise CheckpointError("tensor groups must map names to TensorGroup records")
        if (not isinstance(self.required_groups, tuple) or not self.required_groups
                or any(not isinstance(name, str) or not _GROUP.fullmatch(name) for name in self.required_groups)
                or len(set(self.required_groups)) != len(self.required_groups)
                or "model" not in self.required_groups
                or not set(self.required_groups).issubset(self.groups)):
            raise CheckpointError("required_groups must be unique existing tensor groups")
        if self.purpose == "training_resume":
            mandatory = {"model", "optimizer", "router", "rng"}
            if not mandatory.issubset(self.groups):
                raise CheckpointError(f"training resume requires tensor groups {sorted(mandatory)}")
            if not mandatory.issubset(set(self.required_groups)):
                raise CheckpointError("training resume must declare all mandatory tensor groups")
            if any(not self.groups[name].records for name in ("router", "rng")):
                raise CheckpointError("training resume requires router buffers and RNG tensors")
        for name, group in self.groups.items():
            if name != group.name:
                raise CheckpointError("tensor group key/name mismatch")
            TensorGroup.from_dict(name, group.to_dict(), f"groups.{name}")
        _json_value(self.metadata, "metadata")

    def canonical_bytes(self) -> bytes:
        self.validate()
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, value: Any) -> CheckpointManifest:
        fields = {"schema_version", "format", "transaction_id", "purpose", "successful_step",
                  "config_identity", "architecture_identity", "required_groups", "groups", "metadata"}
        data = _require_keys(value, fields, "manifest")
        if type(data["schema_version"]) is not int or data["schema_version"] != SCHEMA_VERSION:
            raise CheckpointError("manifest.schema_version: unsupported version")
        if data["format"] != "safetensors":
            raise CheckpointError("manifest.format: only safetensors is permitted")
        if not isinstance(data["required_groups"], list) or any(not isinstance(x, str) for x in data["required_groups"]):
            raise CheckpointError("manifest.required_groups: expected string list")
        if not isinstance(data["groups"], Mapping):
            raise CheckpointError("manifest.groups: expected mapping")
        groups = {name: TensorGroup.from_dict(name, item, f"manifest.groups.{name}")
                  for name, item in data["groups"].items()}
        metadata = _json_value(data["metadata"], "manifest.metadata")
        if not isinstance(metadata, Mapping):
            raise CheckpointError("manifest.metadata: expected object")
        result = cls(
            transaction_id=_nonempty(data["transaction_id"], "manifest.transaction_id"),
            purpose=data["purpose"],
            successful_step=data["successful_step"],
            config_identity=_digest(data["config_identity"], "manifest.config_identity"),
            architecture_identity=_digest(data["architecture_identity"], "manifest.architecture_identity"),
            required_groups=tuple(data["required_groups"]),
            groups=groups,
            metadata=metadata,
            schema_version=data["schema_version"],
            format=data["format"],
        )
        result.validate()
        return result


def read_manifest(path: str | Path) -> CheckpointManifest:
    source = Path(path)
    try:
        payload = source.read_text(encoding="utf-8", errors="strict")
    except OSError as exc:
        raise CheckpointError(f"cannot read manifest {source}: {exc}") from exc
    try:
        decoded = json.loads(payload, object_pairs_hook=_reject_duplicate_pairs,
                             parse_constant=_reject_nonfinite)
    except json.JSONDecodeError as exc:
        raise CheckpointError(f"invalid JSON manifest {source}: {exc}") from exc
    return CheckpointManifest.from_dict(decoded)


def _write_manifest_file(path: Path, payload: bytes) -> FileRecord:
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            import os
            os.fsync(stream.fileno())
    except OSError as exc:
        raise CheckpointError(f"cannot write checkpoint manifest {path}: {exc}") from exc
    return FileRecord(path.name, len(payload), hashlib.sha256(payload).hexdigest())
