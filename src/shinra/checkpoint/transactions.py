"""Staged checkpoint writes and explicit, capability-gated publication."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any

from .manifest import (
    CheckpointError,
    CheckpointManifest,
    FileRecord,
    TensorGroup,
    _digest,
    _GROUP,
    _reject_nonfinite,
    _reject_duplicate_pairs,
    _require_keys,
    _safe_filename,
    _json_value,
    _write_manifest_file,
)
from .serialization import TensorSpec, save_tensor_group, tensor_inventory


_TRANSACTION = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_TRAINING_METADATA = frozenset(
    ("optimizer", "scheduler_state", "data_cursor", "tokenizer_identity",
     "corpus_manifest_identity", "ownership", "source_revision",
     "precision_policy", "state_schema_version", "state_phase", "committed_step")
)


@dataclass(frozen=True, slots=True)
class AtomicPublicationGuarantee:
    """Caller attests atomic same-directory replace semantics for this storage."""

    backend_identity: str
    approval_reference: str

    def __post_init__(self) -> None:
        for field, value in (("backend_identity", self.backend_identity),
                             ("approval_reference", self.approval_reference)):
            if not isinstance(value, str) or not value.strip():
                raise CheckpointError(f"publication guarantee {field} is required")


def _write_new(path: Path, payload: bytes) -> None:
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        raise CheckpointError(f"cannot write checkpoint metadata {path}: {exc}") from exc


def _read_json(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8", errors="strict")
    except OSError as exc:
        raise CheckpointError(f"cannot read checkpoint metadata {path}: {exc}") from exc
    try:
        return json.loads(text, object_pairs_hook=_reject_duplicate_pairs,
                          parse_constant=_reject_nonfinite)
    except json.JSONDecodeError as exc:
        raise CheckpointError(f"invalid checkpoint JSON {path}: {exc}") from exc


def _validate_training_metadata(purpose: str, groups: Mapping[str, frozenset[str]], metadata: Mapping[str, Any],
                                required_groups: tuple[str, ...], successful_step: int) -> None:
    if purpose != "training_resume":
        return
    if not {"model", "optimizer", "router", "rng"}.issubset(groups):
        raise CheckpointError("training resume requires model, optimizer, router, and rng tensor groups")
    if not {"model", "optimizer", "router", "rng"}.issubset(required_groups):
        raise CheckpointError("training resume must declare model, optimizer, router, and rng groups required")
    missing = sorted(_TRAINING_METADATA - set(metadata))
    if missing:
        raise CheckpointError(f"training resume metadata is incomplete: {missing}")
    optimizer = metadata["optimizer"]
    if not isinstance(optimizer, Mapping) or set(optimizer) != {
            "optimizer_type", "parameter_groups", "parameter_state"}:
        raise CheckpointError("training resume requires strict stable-name optimizer metadata")
    if not isinstance(optimizer["optimizer_type"], str) or not optimizer["optimizer_type"].strip():
        raise CheckpointError("optimizer_type must identify the concrete optimizer implementation")
    parameter_groups = optimizer["parameter_groups"]
    if not isinstance(parameter_groups, list) or not parameter_groups:
        raise CheckpointError("optimizer parameter_groups must be a nonempty list")
    grouped_names: set[str] = set()
    for index, group in enumerate(parameter_groups):
        if not isinstance(group, Mapping) or set(group) != {"parameter_names", "options"}:
            raise CheckpointError(f"optimizer.parameter_groups[{index}] has an invalid schema")
        names = group["parameter_names"]
        if not isinstance(names, list) or any(not isinstance(name, str) or not name for name in names):
            raise CheckpointError(f"optimizer.parameter_groups[{index}].parameter_names is invalid")
        if len(names) != len(set(names)) or grouped_names.intersection(names):
            raise CheckpointError("optimizer parameter names must be unique across groups")
        grouped_names.update(names)
        if not isinstance(group["options"], Mapping):
            raise CheckpointError(f"optimizer.parameter_groups[{index}].options must be an object")
    parameter_state = optimizer["parameter_state"]
    if not isinstance(parameter_state, Mapping):
        raise CheckpointError("optimizer.parameter_state must be an object")
    tensor_references: list[str] = []
    for parameter_name, state in parameter_state.items():
        if parameter_name not in grouped_names or not isinstance(state, Mapping):
            raise CheckpointError("optimizer state refers to an unnamed or ungrouped parameter")
        for state_name, entry in state.items():
            if not isinstance(state_name, str) or not state_name or not isinstance(entry, Mapping):
                raise CheckpointError("optimizer state entry has an invalid schema")
            if (entry.get("kind") == "tensor" and set(entry) == {"kind", "tensor_name", "device_policy"}
                    and isinstance(entry["tensor_name"], str) and entry["tensor_name"]):
                if entry["device_policy"] not in ("cpu", "parameter"):
                    raise CheckpointError("optimizer device_policy must be cpu or parameter")
                tensor_references.append(entry["tensor_name"])
            elif entry.get("kind") == "scalar" and set(entry) == {"kind", "value"}:
                value = entry["value"]
                if value is not None and type(value) not in (str, bool, int, float):
                    raise CheckpointError("optimizer scalar state must remain a JSON scalar")
                if type(value) is float and not math.isfinite(value):
                    raise CheckpointError("optimizer scalar state must be finite")
                continue
            else:
                raise CheckpointError(f"optimizer state entry {parameter_name}.{state_name} is invalid")
    if (len(tensor_references) != len(set(tensor_references))
            or set(tensor_references) != set(groups["optimizer"])):
        raise CheckpointError("optimizer tensor inventory does not match stable-name optimizer metadata")
    for field in ("tokenizer_identity", "corpus_manifest_identity", "source_revision"):
        value = metadata[field]
        if not isinstance(value, str) or not value.strip():
            raise CheckpointError(f"metadata.{field} must be an approved nonempty identity")
    for field in ("scheduler_state", "data_cursor", "ownership", "precision_policy"):
        if not isinstance(metadata[field], Mapping):
            raise CheckpointError(f"metadata.{field} must be a declarative object")
    if type(metadata["state_schema_version"]) is not int or metadata["state_schema_version"] <= 0:
        raise CheckpointError("metadata.state_schema_version must be a positive integer")
    if metadata["state_phase"] != "COMMITTED":
        raise CheckpointError("training resume snapshots must be captured at COMMITTED state")
    if type(metadata["committed_step"]) is not int or metadata["committed_step"] != successful_step:
        raise CheckpointError("training snapshot committed_step must match successful_step")


def write_checkpoint(root: str | Path, *, transaction_id: str, purpose: str,
                     successful_step: int, config_identity: str, architecture_identity: str,
                     tensor_groups: Mapping[str, Mapping[str, Any]],
                     expected_inventories: Mapping[str, Mapping[str, TensorSpec]],
                     required_groups: tuple[str, ...], metadata: Mapping[str, Any],
                     max_shard_bytes: int) -> Path:
    """Write a complete but unpublished transaction; failures remain uncommitted."""
    if not isinstance(transaction_id, str) or not _TRANSACTION.fullmatch(transaction_id):
        raise CheckpointError("invalid transaction_id")
    if purpose not in ("model", "training_resume"):
        raise CheckpointError("purpose must be model or training_resume")
    if type(successful_step) is not int or successful_step < 0:
        raise CheckpointError("successful_step must be a nonnegative integer")
    _digest(config_identity, "config_identity")
    _digest(architecture_identity, "architecture_identity")
    if type(max_shard_bytes) is not int or max_shard_bytes <= 0:
        raise CheckpointError("max_shard_bytes must be a positive integer")
    if not isinstance(tensor_groups, Mapping) or "model" not in tensor_groups:
        raise CheckpointError("model tensor group is required")
    if any(not isinstance(name, str) or not _GROUP.fullmatch(name) for name in tensor_groups):
        raise CheckpointError("tensor group names must be canonical lowercase identifiers")
    if set(tensor_groups) != set(expected_inventories):
        raise CheckpointError("tensor groups and expected inventories must have identical names")
    if not isinstance(metadata, Mapping):
        raise CheckpointError("metadata must be a mapping")
    _json_value(metadata, "metadata")
    if (not isinstance(required_groups, tuple) or not required_groups
            or any(not isinstance(name, str) for name in required_groups)
            or len(set(required_groups)) != len(required_groups)
            or not set(required_groups).issubset(tensor_groups)
            or "model" not in required_groups):
        raise CheckpointError("required_groups must be an explicit nonempty tuple")
    for group_name, tensors in tensor_groups.items():
        if not isinstance(expected_inventories[group_name], Mapping):
            raise CheckpointError(f"{group_name}: expected explicit tensor inventory")
        actual = tensor_inventory(tensors)
        if set(actual) != set(expected_inventories[group_name]):
            raise CheckpointError(f"{group_name}: source tensor names differ from expected inventory")
        for name, spec in expected_inventories[group_name].items():
            if not isinstance(spec, TensorSpec) or actual[name] != spec:
                raise CheckpointError(f"{group_name}.{name}: source tensor shape/dtype differs from inventory")

    group_names = {name: frozenset(tensors.keys()) for name, tensors in tensor_groups.items()}
    _validate_training_metadata(purpose, group_names, metadata, required_groups, successful_step)

    destination_root = Path(root)
    transaction_path = destination_root / transaction_id
    try:
        destination_root.mkdir(parents=True, exist_ok=True)
        transaction_path.mkdir(exist_ok=False)
    except OSError as exc:
        raise CheckpointError(f"cannot create checkpoint transaction directory {transaction_path}: {exc}") from exc

    groups: dict[str, TensorGroup] = {}
    for group_name in sorted(tensor_groups):
        groups[group_name] = save_tensor_group(
            transaction_path, group_name, tensor_groups[group_name],
            expected_inventories[group_name], max_shard_bytes=max_shard_bytes)
    manifest = CheckpointManifest(
        transaction_id=transaction_id,
        purpose=purpose,
        successful_step=successful_step,
        config_identity=config_identity,
        architecture_identity=architecture_identity,
        required_groups=required_groups,
        groups=groups,
        metadata=metadata,
    )
    manifest_bytes = manifest.canonical_bytes()
    _write_manifest_file(transaction_path / "manifest.json", manifest_bytes)
    return transaction_path


def _verify_transaction(transaction_path: Path, *, require_commit: bool) -> CheckpointManifest:
    if transaction_path.is_symlink() or not transaction_path.is_dir():
        raise CheckpointError(f"checkpoint transaction is not a regular directory: {transaction_path}")
    marker_path = transaction_path / "COMMITTED.json"
    if require_commit and (marker_path.is_symlink() or not marker_path.is_file()):
        raise CheckpointError("checkpoint has no committed publication marker")
    manifest_path = transaction_path / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise CheckpointError("checkpoint manifest is missing or not a regular file")
    manifest_bytes = manifest_path.read_bytes()
    manifest = CheckpointManifest.from_dict(_read_json(manifest_path))
    _validate_training_metadata(
        manifest.purpose,
        {name: frozenset(record.name for record in group.records) for name, group in manifest.groups.items()},
        manifest.metadata,
        manifest.required_groups, manifest.successful_step)
    if manifest.transaction_id != transaction_path.name:
        raise CheckpointError("manifest transaction identity does not match its directory")
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    if require_commit:
        marker = _require_keys(_read_json(marker_path),
                               {"schema_version", "transaction_id", "manifest", "manifest_sha256",
                                "publication_backend", "publication_approval"},
                               "commit marker")
        if type(marker["schema_version"]) is not int or marker["schema_version"] != 1:
            raise CheckpointError("unsupported commit marker schema")
        if marker["transaction_id"] != manifest.transaction_id:
            raise CheckpointError("commit marker transaction identity mismatch")
        if _safe_filename(marker["manifest"], "commit marker manifest", ".json") != "manifest.json":
            raise CheckpointError("commit marker must reference manifest.json")
        if _digest(marker["manifest_sha256"], "commit marker manifest_sha256") != manifest_hash:
            raise CheckpointError("manifest SHA-256 does not match commit marker")
        for field in ("publication_backend", "publication_approval"):
            if not isinstance(marker[field], str) or not marker[field].strip():
                raise CheckpointError(f"commit marker {field} is required")

    expected_files = {"manifest.json"}
    if require_commit:
        expected_files.add("COMMITTED.json")
    for group in manifest.groups.values():
        expected_files.add(group.index_file)
        expected_files.update(file.name for file in group.files)
        file_records = {record.name: record for record in group.files}
        for filename, record in file_records.items():
            path = transaction_path / filename
            if path.is_symlink() or not path.is_file():
                raise CheckpointError(f"required checkpoint file missing or unsafe: {filename}")
            size, digest = _hash_file(path)
            if size != record.size or digest != record.sha256:
                raise CheckpointError(f"checkpoint file integrity mismatch: {filename}")
        index = _require_keys(_read_json(transaction_path / group.index_file),
                              {"metadata", "weight_map"}, f"{group.name} index")
        index_metadata = _require_keys(index["metadata"], {"total_size"}, f"{group.name} index metadata")
        if type(index_metadata["total_size"]) is not int:
            raise CheckpointError(f"{group.name} index total_size must be an integer")
        expected_map = {record.name: record.shard for record in group.records}
        if index["weight_map"] != expected_map:
            raise CheckpointError(f"{group.name} index tensor-to-shard map mismatch")
        if index_metadata["total_size"] != sum(record.nbytes for record in group.records):
            raise CheckpointError(f"{group.name} index total_size mismatch")
    try:
        actual_files = {entry.name for entry in transaction_path.iterdir()}
    except OSError as exc:
        raise CheckpointError(f"cannot enumerate transaction files: {exc}") from exc
    if actual_files != expected_files:
        raise CheckpointError(
            f"transaction file inventory mismatch; missing={sorted(expected_files-actual_files)}, "
            f"unexpected={sorted(actual_files-expected_files)}"
        )
    return manifest


def publish_checkpoint(transaction_path: str | Path, *,
                       guarantee: AtomicPublicationGuarantee) -> CheckpointManifest:
    """Publish only after all indexed files and hashes validate.

    Atomic visibility is claimed only under the caller-supplied backend guarantee.
    This method does not claim crash durability of directory entries.
    """
    if not isinstance(guarantee, AtomicPublicationGuarantee):
        raise CheckpointError("explicit storage atomic-publication guarantee is required")
    path = Path(transaction_path)
    manifest = _verify_transaction(path, require_commit=False)
    marker = {
        "schema_version": 1,
        "transaction_id": manifest.transaction_id,
        "manifest": "manifest.json",
        "manifest_sha256": hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest(),
        "publication_backend": guarantee.backend_identity,
        "publication_approval": guarantee.approval_reference,
    }
    marker_bytes = json.dumps(marker, sort_keys=True, separators=(",", ":"),
                              ensure_ascii=False, allow_nan=False).encode("utf-8")
    temporary_marker = path / f".COMMITTED.{manifest.transaction_id}.tmp"
    final_marker = path / "COMMITTED.json"
    _write_new(temporary_marker, marker_bytes)
    try:
        os.replace(temporary_marker, final_marker)
    except OSError as exc:
        raise CheckpointError(f"atomic commit-marker publication failed: {exc}") from exc
    return _verify_transaction(path, require_commit=True)


def _hash_file(path: Path) -> tuple[int, str]:
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


def recover_checkpoint(root: str | Path, transaction_id: str, *,
                       config_identity: str, architecture_identity: str) -> CheckpointManifest:
    """Validate a committed transaction, returning metadata only."""
    if not isinstance(transaction_id, str) or not _TRANSACTION.fullmatch(transaction_id):
        raise CheckpointError("invalid transaction_id")
    if not isinstance(config_identity, str) or not isinstance(architecture_identity, str):
        raise CheckpointError("expected explicit configuration and architecture identities")
    path = Path(root) / transaction_id
    manifest = _verify_transaction(path, require_commit=True)
    if manifest.config_identity != config_identity:
        raise CheckpointError("checkpoint configuration identity mismatch")
    if manifest.architecture_identity != architecture_identity:
        raise CheckpointError("checkpoint architecture identity mismatch")
    return manifest


def write_training_snapshot(engine, root: str | Path, *, transaction_id: str,
                            source_revision: str, rng_devices: tuple[int, ...], max_shard_bytes: int) -> Path:
    """Stage one real DP replica snapshot; publication remains a separate action.

    The caller must freeze this engine and all writers for the entire snapshot.
    A distributed checkpoint set requires every member's matching transaction;
    staging one member does not publish a recoverable distributed training run.
    """
    from shinra.core.initialization import parameter_specs
    from shinra.training.precision import capture_rng
    from .serialization import capture_optimizer_state, module_state_dict

    progress = engine.state.metadata()
    if engine.stream.attempt is not None or engine.scheduler.completed_step != progress["successful_step"]:
        raise CheckpointError("training snapshot components are not committed together")
    router = {}
    balancing_metadata = []
    for layer, balance in enumerate(engine.balances):
        metadata = balance.checkpoint_metadata()
        if metadata["completed_step"] != progress["successful_step"]:
            raise CheckpointError("router step differs from committed training step")
        balancing_metadata.append(metadata)
        router[f"layers.{layer}.bias"] = balance.bias
        router[f"layers.{layer}.completed_step"] = balance.completed_step
    optimizer = capture_optimizer_state(engine.optimizer.optimizer, engine.optimizer.masters)
    groups = {"model": module_state_dict(engine.model), "optimizer": optimizer.tensors,
              "master": engine.optimizer.masters, "router": router, "rng": capture_rng(rng_devices)}
    dtype_names = {"bfloat16": "BF16", "float32": "F32"}
    model_expected = {spec.name: TensorSpec(spec.shape, dtype_names[spec.dtype]) for spec in parameter_specs(engine.config)}
    execution = engine.model.expert_execution
    if execution is not None:
        model_expected = {name: spec for name, spec in model_expected.items()
                          if ".moe.experts." not in name or
                          execution.ownership.owners[int(name.split(".")[1])][int(name.split(".")[4])] == engine.group.rank}
    expected = {name: tensor_inventory(tensors) for name, tensors in groups.items()}
    expected["model"] = model_expected
    expected["master"] = {name: TensorSpec(spec.shape, "F32") for name, spec in model_expected.items()}
    metadata = {
        "optimizer": optimizer.metadata(), "scheduler_state": engine.scheduler.metadata(),
        "data_cursor": engine.stream.committed.metadata(), "training_progress": progress,
        "tokenizer_identity": engine.config.get("tokenizer.artifact_identity"),
        "corpus_manifest_identity": engine.config.get("data.manifest_identity"),
        "ownership": {"mode": "full_replica_dp" if execution is None else "expert_parallel",
                      "ranks": list(engine.group.ranks), "rank": engine.group.rank,
                      "expert_identity": None if execution is None else execution.ownership.identity},
        "precision_policy": {"model": "bf16", "router": "float32", "optimizer": "fp32_master"},
        "source_revision": source_revision, "state_schema_version": 1, "state_phase": "COMMITTED",
        "committed_step": progress["successful_step"], "balancing": balancing_metadata,
        "rng_devices": list(rng_devices),
    }
    return write_checkpoint(root, transaction_id=transaction_id, purpose="training_resume",
                            successful_step=progress["successful_step"], config_identity=engine.config.identity,
                            architecture_identity=engine.config.architecture_identity, tensor_groups=groups,
                            expected_inventories=expected, required_groups=("model", "optimizer", "master", "router", "rng"),
                            metadata=metadata, max_shard_bytes=max_shard_bytes)


def _training_set_members(root, transaction_ids, *, config_identity, architecture_identity, ranks):
    if not transaction_ids or len(transaction_ids) != len(ranks) or len(set(transaction_ids)) != len(transaction_ids):
        raise CheckpointError("one unique transaction per logical rank is required")
    manifests = tuple(recover_checkpoint(root, tx, config_identity=config_identity,
                                        architecture_identity=architecture_identity) for tx in transaction_ids)
    shared_fields = ("training_progress", "scheduler_state", "tokenizer_identity", "corpus_manifest_identity",
                     "precision_policy", "state_schema_version", "state_phase", "committed_step", "balancing", "source_revision")
    first = manifests[0]
    ownership = first.metadata["ownership"]
    for rank, manifest in enumerate(manifests):
        if manifest.purpose != "training_resume" or set(manifest.groups) != {"model", "optimizer", "master", "router", "rng"}:
            raise CheckpointError("complete model/optimizer/master/router/RNG transaction required")
        if set(manifest.required_groups) != set(manifest.groups):
            raise CheckpointError("all training tensor groups must be required")
        if any(field not in manifest.metadata or manifest.metadata[field] != first.metadata.get(field) for field in shared_fields):
            raise CheckpointError("rank snapshots belong to different committed training versions")
        saved = manifest.metadata["ownership"]
        if set(saved) != {"mode", "ranks", "rank", "expert_identity"} or saved != {**ownership, "rank": rank} or saved["ranks"] != list(ranks):
            raise CheckpointError("rank ownership mismatch")
        if saved["mode"] not in ("full_replica_dp", "expert_parallel"):
            raise CheckpointError("unsupported training ownership")
        if manifest.successful_step != first.successful_step:
            raise CheckpointError("rank optimizer steps differ")
        progress = manifest.metadata["training_progress"]
        if (not isinstance(progress, Mapping) or set(progress) != {"successful_step", "consumed_tokens", "phase"}
                or progress["phase"] != "COMMITTED" or type(progress["successful_step"]) is not int
                or progress["successful_step"] != manifest.successful_step
                or type(progress["consumed_tokens"]) is not int or progress["consumed_tokens"] < 0
                or manifest.metadata["scheduler_state"].get("completed_step") != manifest.successful_step):
            raise CheckpointError("training/cursor/scheduler version is not committed consistently")
    intervals = []
    for manifest in manifests:
        cursor = manifest.metadata["data_cursor"]
        if set(cursor) != {"manifest_identity", "tokenizer_identity", "ordinal", "order", "start_ordinal", "end_ordinal"}:
            raise CheckpointError("invalid cursor schema")
        if (cursor["manifest_identity"] != first.metadata["corpus_manifest_identity"] or
                cursor["tokenizer_identity"] != first.metadata["tokenizer_identity"] or cursor["order"] != "manifest_order"):
            raise CheckpointError("cursor identity/order mismatch")
        if any(type(cursor[k]) is not int for k in ("ordinal", "start_ordinal", "end_ordinal")) or not 0 <= cursor["start_ordinal"] <= cursor["ordinal"] <= cursor["end_ordinal"]:
            raise CheckpointError("invalid committed cursor bounds")
        intervals.append((cursor["start_ordinal"], cursor["end_ordinal"]))
    intervals.sort()
    if intervals[0][0] != 0 or any(a[1] != b[0] for a, b in zip(intervals, intervals[1:])):
        raise CheckpointError("rank cursor partitions overlap or leave gaps")
    return manifests


def publish_training_set(root: str | Path, *, set_id: str, transaction_ids: tuple[str, ...],
                         config_identity: str, architecture_identity: str, ranks: tuple[int, ...],
                         guarantee: AtomicPublicationGuarantee) -> Path:
    """Publish JSON admission for a complete set of already committed rank shards.

    Caller freezes writers and supplies verified same-directory atomic-replace
    semantics. Crash durability of directory entries is not asserted here.
    """
    if not isinstance(guarantee, AtomicPublicationGuarantee) or not isinstance(set_id, str) or not _TRANSACTION.fullmatch(set_id):
        raise CheckpointError("explicit publication guarantee and safe set identity required")
    if not ranks or any(type(rank) is not int or rank < 0 for rank in ranks) or len(set(ranks)) != len(ranks):
        raise CheckpointError("explicit unique process ranks required")
    manifests = _training_set_members(root, transaction_ids, config_identity=config_identity,
                                      architecture_identity=architecture_identity, ranks=ranks)
    directory = Path(root) / ("set-" + set_id)
    directory.mkdir(exist_ok=False)
    body = {"schema_version": 1, "format": "safetensors", "config_identity": config_identity,
            "architecture_identity": architecture_identity, "ranks": list(ranks),
            "transactions": [{"id": m.transaction_id, "sha256": m.sha256} for m in manifests]}
    temporary = directory / ".COMMITTED.tmp"
    _write_new(temporary, json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))
    os.replace(temporary, directory / "COMMITTED.json")
    return directory
