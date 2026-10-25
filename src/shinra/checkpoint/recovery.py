"""Fail-closed loading and identity-checked restoration of checkpoint state."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .manifest import CheckpointError, CheckpointManifest
from .serialization import TensorSpec, load_tensor_group, module_state_dict, tensor_inventory
from .transactions import recover_checkpoint


def load_checkpoint(root: str | Path, transaction_id: str, *,
                    config_identity: str, architecture_identity: str,
                    expected_inventories: Mapping[str, Mapping[str, TensorSpec]],
                    device: str) -> tuple[CheckpointManifest, dict[str, dict[str, Any]]]:
    """Load every manifested tensor group after whole-transaction integrity checks."""
    manifest = recover_checkpoint(
        root, transaction_id,
        config_identity=config_identity,
        architecture_identity=architecture_identity,
    )
    if set(expected_inventories) != set(manifest.groups):
        raise CheckpointError("caller inventory must cover every checkpoint tensor group exactly")
    for required in manifest.required_groups:
        if required not in expected_inventories:
            raise CheckpointError(f"required tensor group inventory is missing: {required}")
    transaction_path = Path(root) / transaction_id
    loaded: dict[str, dict[str, Any]] = {}
    for name in sorted(manifest.groups):
        loaded[name] = load_tensor_group(
            transaction_path, manifest.groups[name], expected_inventories[name], device=device)
    return manifest, loaded


def restore_module_state(module: Any, tensors: Mapping[str, Any]) -> None:
    """Apply a complete, exact model state inventory; never permit partial loading."""
    current = module_state_dict(module)
    expected = tensor_inventory(current)
    actual = tensor_inventory(tensors)
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        raise CheckpointError(f"model state inventory mismatch; missing={missing}, unexpected={unexpected}")
    for name, spec in expected.items():
        if actual[name] != spec:
            raise CheckpointError(f"model state shape/dtype mismatch: {name}")
    if not callable(getattr(module, "load_state_dict", None)):
        raise CheckpointError("module must provide load_state_dict()")
    module.load_state_dict(dict(tensors), strict=True)
