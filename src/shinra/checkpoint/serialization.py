"""Strict tensor inventories and Safetensors-only tensor-group I/O."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from urllib.parse import quote
from typing import Any

from .manifest import CheckpointError, FileRecord, TensorGroup, TensorRecord


_TORCH_TO_SAFE = {
    "bool": "BOOL", "uint8": "U8", "int8": "I8", "int16": "I16", "int32": "I32",
    "int64": "I64", "float16": "F16", "bfloat16": "BF16", "float32": "F32", "float64": "F64",
    "float8_e4m3fn": "F8_E4M3", "float8_e4m3fnuz": "F8_E4M3FNUZ",
    "float8_e5m2": "F8_E5M2", "float8_e5m2fnuz": "F8_E5M2FNUZ",
}
_SAFE_TO_TORCH = {value: key for key, value in _TORCH_TO_SAFE.items()}


@dataclass(frozen=True, slots=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True, slots=True)
class OptimizerSnapshot:
    """Optimizer state keyed by stable logical parameter names, never ordinal IDs."""

    optimizer_type: str
    parameter_groups: tuple[Mapping[str, Any], ...]
    parameter_state: Mapping[str, Mapping[str, Any]]
    tensors: Mapping[str, Any]

    def metadata(self) -> dict[str, Any]:
        return {"optimizer_type": self.optimizer_type,
                "parameter_groups": [dict(group) for group in self.parameter_groups],
                "parameter_state": {name: dict(state) for name, state in self.parameter_state.items()}}


def _torch_dtype_name(tensor: Any) -> str:
    dtype_name = str(tensor.dtype).removeprefix("torch.")
    if dtype_name not in _TORCH_TO_SAFE:
        raise CheckpointError(f"unsupported tensor dtype {dtype_name!r}")
    return _TORCH_TO_SAFE[dtype_name]


def _validate_logical_name(name: Any, path: str) -> str:
    if not isinstance(name, str) or not name or name.strip() != name or "\x00" in name:
        raise CheckpointError(f"{path}: expected nonempty logical tensor name")
    return name


def tensor_inventory(tensors: Mapping[str, Any]) -> dict[str, TensorSpec]:
    """Describe actual tensors without materializing or serializing their values."""
    if not isinstance(tensors, Mapping):
        raise CheckpointError("tensor inventory must be a mapping")
    result: dict[str, TensorSpec] = {}
    storage_owners: dict[tuple[str, int, int], str] = {}
    for raw_name, tensor in tensors.items():
        name = _validate_logical_name(raw_name, "tensor name")
        if (not hasattr(tensor, "shape") or not hasattr(tensor, "dtype")
                or not hasattr(tensor, "layout") or not hasattr(tensor, "device")):
            raise CheckpointError(f"{name}: expected a tensor value")
        if str(tensor.layout) != "torch.strided" or bool(getattr(tensor, "is_quantized", False)):
            raise CheckpointError(f"{name}: only dense, non-quantized strided tensors are supported")
        if tensor.device.type == "meta":
            raise CheckpointError(f"{name}: meta tensors cannot be checkpointed")
        shape = tuple(tensor.shape)
        if any(type(dimension) is not int or dimension < 0 for dimension in shape):
            raise CheckpointError(f"{name}: invalid tensor shape")
        if int(tensor.numel()) > 0:
            storage = tensor.untyped_storage()
            storage_identity = (str(tensor.device), int(storage.data_ptr()), int(storage.nbytes()))
            previous_name = storage_owners.get(storage_identity)
            if previous_name is not None:
                raise CheckpointError(
                    f"{name} shares storage with {previous_name}; tied/aliased tensor state is not supported"
                )
            storage_owners[storage_identity] = name
        result[name] = TensorSpec(shape, _torch_dtype_name(tensor))
    return result


def module_state_dict(module: Any) -> Mapping[str, Any]:
    """Capture parameters and persistent buffers by the module's logical state names."""
    if not callable(getattr(module, "state_dict", None)):
        raise CheckpointError("module must provide state_dict()")
    state = module.state_dict()
    if not tensor_inventory(state):
        raise CheckpointError("module state_dict() must contain at least one tensor")
    return state


def _safe_api():
    try:
        from safetensors.torch import load_file, save_file
    except ImportError as exc:
        raise CheckpointError(
            "Safetensors is required for checkpoint tensor I/O; install the approved runtime dependency"
        ) from exc
    return load_file, save_file


def _validate_against_expected(tensors: Mapping[str, Any], expected: Mapping[str, TensorSpec],
                               group_name: str) -> dict[str, TensorSpec]:
    actual = tensor_inventory(tensors)
    if not isinstance(expected, Mapping) or (group_name == "model" and not expected):
        raise CheckpointError(f"{group_name}: explicit expected inventory is required")
    if set(actual) != set(expected):
        missing, unexpected = sorted(set(expected) - set(actual)), sorted(set(actual) - set(expected))
        raise CheckpointError(f"{group_name}: tensor inventory mismatch; missing={missing}, unexpected={unexpected}")
    for name, spec in expected.items():
        if not isinstance(spec, TensorSpec):
            raise CheckpointError(f"{group_name}.{name}: expected TensorSpec")
        if actual[name] != spec:
            raise CheckpointError(f"{group_name}.{name}: shape/dtype mismatch")
    return actual


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
                size += len(block)
    except OSError as exc:
        raise CheckpointError(f"cannot hash checkpoint file {path}: {exc}") from exc
    return size, digest.hexdigest()


def _write_json_new(path: Path, value: Mapping[str, Any]) -> FileRecord:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            import os
            os.fsync(stream.fileno())
    except OSError as exc:
        raise CheckpointError(f"cannot write checkpoint index {path}: {exc}") from exc
    return FileRecord(path.name, len(payload), hashlib.sha256(payload).hexdigest())


def save_tensor_group(directory: str | Path, group_name: str, tensors: Mapping[str, Any],
                      expected: Mapping[str, TensorSpec], *, max_shard_bytes: int) -> TensorGroup:
    """Write one exact tensor inventory into deterministic Safetensors shards."""
    from .manifest import _GROUP

    target = Path(directory)
    if not isinstance(group_name, str) or not _GROUP.fullmatch(group_name):
        raise CheckpointError("invalid tensor group name")
    if type(max_shard_bytes) is not int or max_shard_bytes <= 0:
        raise CheckpointError("max_shard_bytes must be a positive integer")
    actual = _validate_against_expected(tensors, expected, group_name)
    load_file, save_file = _safe_api()
    del load_file
    try:
        if target.is_symlink() or not target.is_dir():
            raise CheckpointError(f"tensor group directory does not exist: {target}")
    except OSError as exc:
        raise CheckpointError(f"cannot inspect tensor group directory {target}: {exc}") from exc

    ordered_names = sorted(tensors)
    shard_batches: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for name in ordered_names:
        tensor = tensors[name]
        size = int(tensor.numel()) * int(tensor.element_size())
        if current and current_bytes + size > max_shard_bytes:
            shard_batches.append(current)
            current, current_bytes = [], 0
        current.append(name)
        current_bytes += size
    if current:
        shard_batches.append(current)
    shard_total = len(shard_batches)
    file_records: list[FileRecord] = []
    tensor_records: list[TensorRecord] = []
    weight_map: dict[str, str] = {}
    total_bytes = 0

    for shard_number, names in enumerate(shard_batches, start=1):
        filename = f"{group_name}-{shard_number:05d}-of-{shard_total:05d}.safetensors"
        destination = target / filename
        cpu_tensors = {}
        for name in names:
            tensor = tensors[name]
            cpu_tensors[name] = tensor.detach().to(device="cpu").contiguous().clone()
        try:
            save_file(cpu_tensors, str(destination))
        except (OSError, RuntimeError, ValueError) as exc:
            raise CheckpointError(f"Safetensors save failed for {destination}: {exc}") from exc
        size, digest = _sha256_file(destination)
        file_records.append(FileRecord(filename, size, digest))
        total_bytes += size
        for name in names:
            spec = actual[name]
            tensor = tensors[name]
            byte_count = int(tensor.numel()) * int(tensor.element_size())
            tensor_records.append(TensorRecord(name, spec.shape, spec.dtype, byte_count, filename))
            weight_map[name] = filename

    index_filename = f"{group_name}.safetensors.index.json"
    index_path = target / index_filename
    index_record = _write_json_new(index_path, {
        "metadata": {"total_size": total_bytes},
        "weight_map": {name: weight_map[name] for name in sorted(weight_map)},
    })
    return TensorGroup(group_name, index_filename, tuple(tensor_records),
                       tuple(file_records + [index_record]))


def optimizer_snapshot_inventory(snapshot: OptimizerSnapshot) -> dict[str, TensorSpec]:
    if not isinstance(snapshot, OptimizerSnapshot):
        raise CheckpointError("expected OptimizerSnapshot")
    return tensor_inventory(snapshot.tensors)


def load_tensor_group(directory: str | Path, group: TensorGroup,
                      expected: Mapping[str, TensorSpec], *, device: str) -> dict[str, Any]:
    """Load and validate every tensor in an indexed group, with no partial result."""
    if not isinstance(device, str) or not device:
        raise CheckpointError("an explicit target device is required")
    load_file, _ = _safe_api()
    target = Path(directory)
    expected_by_name = dict(expected)
    if any(not isinstance(spec, TensorSpec) for spec in expected_by_name.values()):
        raise CheckpointError(f"{group.name}: expected inventory contains a non-TensorSpec value")
    record_by_name = {record.name: record for record in group.records}
    if set(record_by_name) != set(expected_by_name):
        raise CheckpointError(f"{group.name}: expected inventory differs from manifest")
    file_by_name = {record.name: record for record in group.files}
    loaded: dict[str, Any] = {}
    for shard_name in sorted({record.shard for record in group.records}):
        file_record = file_by_name.get(shard_name)
        if file_record is None:
            raise CheckpointError(f"{group.name}: missing shard record {shard_name}")
        shard_path = target / shard_name
        size, digest = _sha256_file(shard_path)
        if size != file_record.size or digest != file_record.sha256:
            raise CheckpointError(f"{shard_name}: size or SHA-256 mismatch")
        try:
            shard = load_file(str(shard_path), device="cpu")
        except (OSError, RuntimeError, ValueError) as exc:
            raise CheckpointError(f"Safetensors load failed for {shard_path}: {exc}") from exc
        expected_shard_names = {name for name, record in record_by_name.items() if record.shard == shard_name}
        if set(shard) != expected_shard_names:
            raise CheckpointError(f"{shard_name}: tensor keys differ from manifest")
        for name, tensor in shard.items():
            record = record_by_name[name]
            spec = expected_by_name[name]
            if tuple(tensor.shape) != spec.shape or _torch_dtype_name(tensor) != spec.dtype:
                raise CheckpointError(f"{name}: tensor shape/dtype differs from expected inventory")
            if tuple(record.shape) != spec.shape or record.dtype != spec.dtype:
                raise CheckpointError(f"{name}: manifest inventory differs from expected inventory")
            if record.nbytes != int(tensor.numel()) * int(tensor.element_size()):
                raise CheckpointError(f"{name}: byte-size metadata mismatch")
            loaded[name] = tensor.to(device=device)
    if set(loaded) != set(expected_by_name):
        raise CheckpointError(f"{group.name}: incomplete tensor group")
    return loaded


def _json_scalar(value: Any, path: str) -> Any:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise CheckpointError(f"{path}: nonfinite optimizer scalar")
        return value
    if isinstance(value, (list, tuple)):
        return [_json_scalar(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise CheckpointError(f"{path}: optimizer mapping keys must be nonempty strings")
            result[key] = _json_scalar(item, f"{path}.{key}")
        return result
    raise CheckpointError(f"{path}: unsupported optimizer metadata type {type(value).__name__}")


def _optimizer_scalar(value: Any, path: str) -> Any:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise CheckpointError(f"{path}: optimizer scalar state must be a finite JSON scalar")


def capture_optimizer_state(optimizer: Any, named_parameters: Mapping[str, Any]) -> OptimizerSnapshot:
    """Capture optimizer states by logical names and require JSON-representable group options."""
    if not hasattr(optimizer, "param_groups") or not hasattr(optimizer, "state"):
        raise CheckpointError("optimizer must expose param_groups and state")
    parameter_names: dict[int, str] = {}
    for name, parameter in named_parameters.items():
        _validate_logical_name(name, "parameter name")
        if id(parameter) in parameter_names:
            raise CheckpointError("one parameter object has multiple logical names")
        parameter_names[id(parameter)] = name

    groups: list[dict[str, Any]] = []
    seen_parameter_ids: set[int] = set()
    for group_index, group in enumerate(optimizer.param_groups):
        if not isinstance(group, Mapping) or "params" not in group:
            raise CheckpointError(f"optimizer.param_groups[{group_index}]: invalid group")
        names = []
        for parameter in group["params"]:
            parameter_id = id(parameter)
            if parameter_id not in parameter_names:
                raise CheckpointError(f"optimizer group {group_index} contains an unnamed parameter")
            if parameter_id in seen_parameter_ids:
                raise CheckpointError("optimizer parameter appears in more than one parameter group")
            seen_parameter_ids.add(parameter_id)
            names.append(parameter_names[parameter_id])
        options = {key: _json_scalar(value, f"optimizer.param_groups[{group_index}].{key}")
                   for key, value in group.items() if key != "params"}
        groups.append({"parameter_names": names, "options": options})

    tensor_states: dict[str, Any] = {}
    parameter_state: dict[str, dict[str, Any]] = {}
    for parameter, state in optimizer.state.items():
        parameter_id = id(parameter)
        if parameter_id not in parameter_names or parameter_id not in seen_parameter_ids:
            raise CheckpointError("optimizer state refers to a parameter outside named groups")
        parameter_name = parameter_names[parameter_id]
        if not isinstance(state, Mapping):
            raise CheckpointError(f"optimizer state for {parameter_name} must be a mapping")
        entries: dict[str, Any] = {}
        for state_key, value in state.items():
            if not isinstance(state_key, str) or not state_key:
                raise CheckpointError("optimizer state keys must be nonempty strings")
            if hasattr(value, "shape") and hasattr(value, "dtype") and hasattr(value, "layout"):
                tensor_name = f"optimizer.state.{quote(parameter_name, safe='')}.{quote(state_key, safe='')}"
                if tensor_name in tensor_states:
                    raise CheckpointError("optimizer tensor state name collision")
                if value.device.type not in ("cpu", parameter.device.type):
                    raise CheckpointError(
                        f"optimizer state for {parameter_name}.{state_key} uses an unsupported device placement"
                    )
                tensor_states[tensor_name] = value
                entries[state_key] = {
                    "kind": "tensor",
                    "tensor_name": tensor_name,
                    "device_policy": "cpu" if value.device.type == "cpu" else "parameter",
                }
            else:
                entries[state_key] = {"kind": "scalar", "value": _optimizer_scalar(
                    value, f"optimizer.state.{parameter_name}.{state_key}")}
        parameter_state[parameter_name] = entries
    optimizer_class = type(optimizer)
    return OptimizerSnapshot(
        f"{optimizer_class.__module__}.{optimizer_class.__qualname__}",
        tuple(groups), parameter_state, tensor_states,
    )


def restore_optimizer_state(optimizer: Any, named_parameters: Mapping[str, Any],
                            snapshot_metadata: Mapping[str, Any], tensors: Mapping[str, Any]) -> None:
    """Restore only into an exactly matching optimizer type, group layout, and options."""
    if not hasattr(optimizer, "param_groups") or not hasattr(optimizer, "state"):
        raise CheckpointError("optimizer must expose param_groups and state")
    expected = capture_optimizer_state(optimizer, named_parameters)
    actual_type = f"{type(optimizer).__module__}.{type(optimizer).__qualname__}"
    if snapshot_metadata.get("optimizer_type") != actual_type:
        raise CheckpointError("optimizer implementation identity mismatch")
    if set(snapshot_metadata) != {"optimizer_type", "parameter_groups", "parameter_state"}:
        raise CheckpointError("optimizer metadata has missing or unknown fields")
    saved_groups = snapshot_metadata.get("parameter_groups")
    if not isinstance(saved_groups, list) or saved_groups != list(expected.parameter_groups):
        raise CheckpointError("optimizer parameter groups or hyperparameters do not match")
    saved_state = snapshot_metadata.get("parameter_state")
    if not isinstance(saved_state, Mapping):
        raise CheckpointError("optimizer parameter_state metadata is missing")
    parameter_by_name = dict(named_parameters)
    optimizer_parameter_names = {
        name for group in expected.parameter_groups for name in group["parameter_names"]
    }
    if any(name not in optimizer_parameter_names for name in saved_state):
        raise CheckpointError("optimizer state references a parameter outside its saved groups")
    if not isinstance(tensors, Mapping):
        raise CheckpointError("optimizer tensor state must be a mapping")
    restored: dict[Any, dict[str, Any]] = {}
    for parameter_name, state in saved_state.items():
        if parameter_name not in parameter_by_name or not isinstance(state, Mapping):
            raise CheckpointError("optimizer state references an unknown logical parameter")
        parameter = parameter_by_name[parameter_name]
        values: dict[str, Any] = {}
        for state_key, entry in state.items():
            if not isinstance(entry, Mapping) or entry.get("kind") not in ("tensor", "scalar"):
                raise CheckpointError(f"invalid optimizer state entry for {parameter_name}.{state_key}")
            if entry["kind"] == "tensor":
                if set(entry) != {"kind", "tensor_name", "device_policy"}:
                    raise CheckpointError(f"invalid tensor optimizer metadata for {parameter_name}.{state_key}")
                tensor_name = entry.get("tensor_name")
                if not isinstance(tensor_name, str) or tensor_name not in tensors:
                    raise CheckpointError(f"missing optimizer tensor state {tensor_name!r}")
                device_policy = entry["device_policy"]
                if device_policy not in ("cpu", "parameter"):
                    raise CheckpointError(f"invalid optimizer device policy for {parameter_name}.{state_key}")
                target_device = "cpu" if device_policy == "cpu" else parameter.device
                values[state_key] = tensors[tensor_name].to(device=target_device)
            else:
                if set(entry) != {"kind", "value"}:
                    raise CheckpointError(f"invalid scalar optimizer metadata for {parameter_name}.{state_key}")
                values[state_key] = _optimizer_scalar(entry.get("value"), f"{parameter_name}.{state_key}")
        restored[parameter] = values
    expected_tensor_names = []
    for state in saved_state.values():
        if not isinstance(state, Mapping):
            raise CheckpointError("optimizer parameter state must be an object")
        for entry in state.values():
            if isinstance(entry, Mapping) and entry.get("kind") == "tensor":
                if not isinstance(entry.get("tensor_name"), str):
                    raise CheckpointError("optimizer tensor reference must be a string")
                expected_tensor_names.append(entry["tensor_name"])
    if len(expected_tensor_names) != len(set(expected_tensor_names)) or set(expected_tensor_names) != set(tensors):
        raise CheckpointError("optimizer tensor inventory contains missing or unexpected states")
    optimizer.state.clear()
    optimizer.state.update(restored)
