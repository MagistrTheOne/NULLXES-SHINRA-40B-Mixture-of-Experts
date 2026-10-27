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


def load_model_checkpoint(model, config, root: str | Path, transaction_id: str, *, device: str):
    """Apply model tensors only after committed transaction and fixed D02 admission."""
    from shinra.core.initialization import parameter_specs

    if model.tensor_execution is not None:
        raise CheckpointError("TP shard inventory admission is not yet integrated; no implicit repartitioning")

    manifest = recover_checkpoint(root, transaction_id, config_identity=config.identity,
                                  architecture_identity=config.architecture_identity)
    expected = {spec.name: TensorSpec(spec.shape, {"bfloat16": "BF16", "float32": "F32"}[spec.dtype])
                for spec in parameter_specs(config)}
    tensors = load_tensor_group(Path(root)/transaction_id, manifest.groups["model"], expected, device=device)
    # Invalidate an old identity before any potentially partial mutation.
    model.checkpoint_identity = None
    restore_module_state(model, tensors)
    model.checkpoint_identity = manifest.sha256
    return manifest


def restore_training_set(engine, root: str | Path, set_id: str, *, rng_devices: tuple[int, ...]):
    """Restore one admitted DP/EP rank from a complete committed checkpoint set.

    Existing engine objects supply identities and ownership. All members must
    invoke this operation in the same order. Any error requires group restart;
    tensor mutation is not rolled back or represented as atomically reversible.
    """
    import torch
    import torch.distributed as dist
    from shinra.core.initialization import parameter_specs
    from shinra.training.precision import capture_rng, restore_rng
    from shinra.training.scheduler import StepScheduler
    from shinra.data.streaming import StreamCursor, PackedStream
    from .transactions import _TRANSACTION, _read_json, _training_set_members
    from .serialization import capture_optimizer_state, restore_optimizer_state

    if engine.state.phase not in ("COMMITTED", "RECOVERY_REQUIRED"):
        raise CheckpointError("restore requires idle or failed engine")
    engine.state.phase = "RECOVERY_REQUIRED"
    if not isinstance(set_id, str) or not _TRANSACTION.fullmatch(set_id):
        raise CheckpointError("invalid checkpoint set ID")
    marker = Path(root) / ("set-" + set_id) / "COMMITTED.json"
    if marker.is_symlink() or marker.parent.is_symlink() or not marker.is_file():
        raise CheckpointError("complete training set marker is missing or unsafe")
    body = _read_json(marker)
    if not isinstance(body, dict) or set(body) != {"schema_version", "format", "config_identity", "architecture_identity", "ranks", "transactions"}:
        raise CheckpointError("invalid training set schema")
    if type(body["schema_version"]) is not int or body["schema_version"] != 1 or body["format"] != "safetensors":
        raise CheckpointError("unsupported training set version/format")
    if body["config_identity"] != engine.config.identity or body["architecture_identity"] != engine.config.architecture_identity or body["ranks"] != list(engine.group.ranks):
        raise CheckpointError("training set configuration/ownership mismatch")
    entries = body["transactions"]
    if not isinstance(entries, list) or any(not isinstance(e, dict) or set(e) != {"id", "sha256"} for e in entries):
        raise CheckpointError("invalid transaction inventory")
    manifests = _training_set_members(root, tuple(e["id"] for e in entries), config_identity=engine.config.identity,
                                      architecture_identity=engine.config.architecture_identity, ranks=engine.group.ranks)
    if any(m.sha256 != e["sha256"] for m, e in zip(manifests, entries)):
        raise CheckpointError("rank manifest changed after set publication")
    engine.group.agree(body, device=engine.device)
    manifest = manifests[engine.group.rank]
    meta = manifest.metadata
    execution = engine.model.expert_execution
    expected_owner = {"mode": "full_replica_dp" if execution is None else "expert_parallel",
                      "ranks": list(engine.group.ranks), "rank": engine.group.rank,
                      "expert_identity": None if execution is None else execution.ownership.identity}
    if meta["ownership"] != expected_owner:
        raise CheckpointError("expert ownership identity mismatch")
    if meta["tokenizer_identity"] != engine.config.get("tokenizer.artifact_identity") or meta["corpus_manifest_identity"] != engine.config.get("data.manifest_identity"):
        raise CheckpointError("tokenizer/corpus identity mismatch")
    if meta["precision_policy"] != {"model": "bf16", "router": "float32", "optimizer": "fp32_master"} or meta.get("rng_devices") != list(rng_devices):
        raise CheckpointError("precision or RNG device policy mismatch")
    progress = meta["training_progress"]
    if not isinstance(progress, dict) or set(progress) != {"successful_step", "consumed_tokens", "phase"} or progress["phase"] != "COMMITTED":
        raise CheckpointError("invalid committed training progress")
    step = progress["successful_step"]
    if type(step) is not int or step != manifest.successful_step or type(progress["consumed_tokens"]) is not int or not 0 <= progress["consumed_tokens"] <= engine.config.get("training.token_budget"):
        raise CheckpointError("invalid training progress values")
    scheduler = StepScheduler(engine.config, completed_step=step)
    if meta["scheduler_state"] != scheduler.metadata():
        raise CheckpointError("scheduler progress/algorithm mismatch")
    cursor = StreamCursor(**meta["data_cursor"])
    current = engine.stream.committed
    if (cursor.start_ordinal, cursor.end_ordinal) != (current.start_ordinal, current.end_ordinal):
        raise CheckpointError("data partition changed across resume")
    if max(m.metadata["data_cursor"]["end_ordinal"] for m in manifests) != len(engine.stream.reader.manifest.documents):
        raise CheckpointError("checkpoint partitions do not cover approved corpus")
    # Reconstruct and verify the exact real-data prefix before model mutation.
    restored_stream = PackedStream(engine.stream.reader, engine.stream.packer, cursor)
    expected_model = {}
    for spec in parameter_specs(engine.config):
        if execution is not None and ".moe.experts." in spec.name:
            parts = spec.name.split(".")
            if execution.ownership.owners[int(parts[1])][int(parts[4])] != engine.group.rank:
                continue
        expected_model[spec.name] = TensorSpec(spec.shape, {"bfloat16": "BF16", "float32": "F32"}[spec.dtype])
    if set(expected_model) != set(engine.optimizer.masters):
        raise CheckpointError("live logical parameter inventory differs from checkpoint contract")
    expected = {"model": expected_model,
                "master": {n: TensorSpec(s.shape, "F32") for n,s in expected_model.items()},
                "router": {f"layers.{i}.{suffix}": TensorSpec(shape, dtype) for i in range(48)
                           for suffix, shape, dtype in (("bias", (128,), "F32"), ("completed_step", (), "I64"))},
                "rng": tensor_inventory(capture_rng(rng_devices))}
    saved_optimizer = meta["optimizer"]
    live_optimizer = capture_optimizer_state(engine.optimizer.optimizer, engine.optimizer.masters).metadata()
    expected_lr = engine.config.get("training.learning_rate") if step == 0 else StepScheduler(engine.config, completed_step=step-1).next_lr()
    expected_groups = live_optimizer["parameter_groups"]
    for group in expected_groups:
        group["options"]["lr"] = expected_lr
    if saved_optimizer["optimizer_type"] != live_optimizer["optimizer_type"] or saved_optimizer["parameter_groups"] != expected_groups:
        raise CheckpointError("optimizer type, logical groups or coefficients changed")
    states = saved_optimizer["parameter_state"]
    if set(states) != (set(expected_model) if step else set()):
        raise CheckpointError("AdamW state is incomplete for committed optimizer step")
    expected["optimizer"] = {}
    for name, entries_state in states.items():
        if set(entries_state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise CheckpointError("unsupported or incomplete AdamW state")
        for key, entry in entries_state.items():
            if set(entry) != {"kind", "tensor_name", "device_policy"} or entry["kind"] != "tensor":
                raise CheckpointError("AdamW tensor metadata required")
            policy = "cpu" if key == "step" or engine.device.type == "cpu" else "parameter"
            if entry["device_policy"] != policy:
                raise CheckpointError("AdamW tensor placement differs from admitted adapter")
            expected["optimizer"][entry["tensor_name"]] = TensorSpec(() if key == "step" else expected_model[name].shape, "F32")
    loaded = {name: load_tensor_group(Path(root)/manifest.transaction_id, manifest.groups[name], inventory, device="cpu")
              for name, inventory in expected.items()}
    for name, tensor in loaded["master"].items():
        if not bool(torch.isfinite(tensor).all()) or not torch.equal(tensor.to(loaded["model"][name].dtype), loaded["model"][name]):
            raise CheckpointError("model/master values diverge or are nonfinite")
    for name, entries_state in states.items():
        for key, entry in entries_state.items():
            value = loaded["optimizer"][entry["tensor_name"]]
            if not bool(torch.isfinite(value).all()) or (key == "step" and value.item() != step) or (key == "exp_avg_sq" and bool((value < 0).any())):
                raise CheckpointError("invalid AdamW numerical state")
    balancing = meta["balancing"]
    if not isinstance(balancing, list) or len(balancing) != 48:
        raise CheckpointError("incomplete balancing metadata")
    for i, balance in enumerate(engine.balances):
        if balancing[i] != {"algorithm": "bounded_sign", "learning_rate": balance.rate, "bound": balance.bound, "completed_step": step}:
            raise CheckpointError("balancing policy/step mismatch")
        bias = loaded["router"][f"layers.{i}.bias"]
        if loaded["router"][f"layers.{i}.completed_step"].item() != step or not bool(torch.isfinite(bias).all()) or bool((bias.abs() > balance.bound).any()):
            raise CheckpointError("invalid router state")
    # Compare replicated values without overwriting a divergent checkpoint.
    replicated = [(name, value) for name, value in loaded["master"].items() if name not in engine.owned_names]
    replicated += list(loaded["router"].items())
    replicated += [(entry["tensor_name"], loaded["optimizer"][entry["tensor_name"]])
                   for name, entries_state in states.items() if name not in engine.owned_names
                   for entry in entries_state.values()]
    for name, value in sorted(replicated):
        local = value.to(engine.device)
        reference = local.clone()
        dist.broadcast(reference, src=engine.group.ranks[0], group=engine.group.process_group)
        equal = engine.group.integer_sum(torch.tensor(int(torch.equal(local, reference)), dtype=torch.int64, device=engine.device))
        if equal.item() != engine.group.size:
            raise CheckpointError("replicated checkpoint values differ: " + name)
    engine.model.checkpoint_identity = None
    with torch.no_grad():
        for name, master in engine.optimizer.masters.items():
            master.copy_(loaded["master"][name].to(master.device))
    restore_module_state(engine.model, loaded["model"])
    for group in engine.optimizer.optimizer.param_groups:
        group["lr"] = expected_lr
    restore_optimizer_state(engine.optimizer.optimizer, engine.optimizer.masters, saved_optimizer, loaded["optimizer"])
    with torch.no_grad():
        for i, balance in enumerate(engine.balances):
            balance.bias.copy_(loaded["router"][f"layers.{i}.bias"].to(balance.bias.device))
            balance.completed_step.fill_(step)
            balance.next_bias, balance.attempt = None, None
            balance.events.clear()
            balance.pending = [0] * 128
            balance.phase = "COMMITTED"
    engine.optimizer.clear()
    restore_rng(loaded["rng"], rng_devices)
    engine.group.integer_sum(torch.ones((), dtype=torch.int64, device=engine.device))
    engine.stream, engine.scheduler = restored_stream, scheduler
    engine.state.successful_step, engine.state.consumed_tokens = step, progress["consumed_tokens"]
    engine.state.attempt, engine.state.phase = None, "COMMITTED"
    engine.model.checkpoint_identity = manifest.sha256
    return manifest
