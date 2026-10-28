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

    if any(getattr(model, name) is not None for name in
           ("parallel_execution", "tensor_execution", "expert_execution", "context_execution")):
        raise CheckpointError("single-transaction model loader requires resident tensors; use restore_training_set for mixed shards")

    manifest = recover_checkpoint(root, transaction_id, config_identity=config.identity,
                                  architecture_identity=config.architecture_identity)
    expected = {spec.name: TensorSpec(spec.shape, {"bfloat16": "BF16", "float32": "F32"}[spec.dtype])
                for spec in parameter_specs(config)}
    tensors = load_tensor_group(Path(root)/transaction_id, manifest.groups["model"], expected, device=device)
    # Invalidate an old identity before any potentially partial mutation.
    model.checkpoint_identity = None
    model.checkpoint_load_state = "RECOVERY_REQUIRED"
    restore_module_state(model, tensors)
    if hasattr(model, "inference_balancing_bias"):
        del model.inference_balancing_bias
    model.checkpoint_identity = manifest.sha256
    model.checkpoint_load_state = "COMMITTED"
    return manifest


def load_resident_from_training_set(model, config, root: str | Path, set_id: str, *,
                                    source_configuration, expected_set_identity: str,
                                    approval_reference: str):
    """Explicit mixed-checkpoint -> resident inference reconstruction.

    The caller pins the canonical JSON set digest and supplies BOTH source and
    destination configuration snapshots. Only operational fields may differ;
    architecture identity includes initialization, RoPE, tokenizer and routing.
    No distributed groups are created and no forward is executed. All tensors
    are verified/reconstructed on CPU before any resident parameter mutation.
    Storage must be immutable throughout this operation. No bounded-memory or
    atomic in-memory recovery claim is made.
    """
    import hashlib
    import json
    import torch
    from shinra.core.initialization import parameter_specs
    from shinra.core.validation import validate_architecture
    from shinra.distributed.ownership import ParallelLayout
    from .transactions import read_training_set
    from .manifest import _digest

    validate_architecture(config)
    validate_architecture(source_configuration)
    if not isinstance(approval_reference, str) or not approval_reference.strip():
        raise CheckpointError("explicit reconstruction approval reference required")
    _digest(expected_set_identity, "expected set identity")
    if any(getattr(model, name) is not None for name in
           ("parallel_execution", "tensor_execution", "expert_execution", "context_execution")):
        raise CheckpointError("reconstruction target must be the complete resident canonical model")
    if model.configuration_identity != config.identity or config.architecture_identity != source_configuration.architecture_identity:
        raise CheckpointError("source/target logical architecture or resident configuration differs")
    body, manifests = read_training_set(root, set_id, configuration=source_configuration)
    identity = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                         allow_nan=False).encode("utf-8")).hexdigest()
    if identity != expected_set_identity:
        raise CheckpointError("checkpoint set identity differs from explicit reconstruction request")
    ownership = manifests[0].metadata["ownership"]
    if ownership["mode"] != "mixed":
        raise CheckpointError("BLOCKED: this reconstruction entrypoint requires an explicit mixed layout")
    layout = ParallelLayout(**ownership["layout"])
    expected = {s.name: TensorSpec(s.shape, "F32" if s.dtype == "float32" else "BF16")
                for s in parameter_specs(config)}
    if tensor_inventory(module_state_dict(model)) != expected:
        raise CheckpointError("resident model inventory differs from canonical logical tensors")
    assembled, slices = {}, {}
    router_reference = None
    router_inventory = {f"layers.{i}.{suffix}": TensorSpec(shape, dtype) for i in range(48)
                        for suffix,shape,dtype in (("bias", (128,), "F32"), ("completed_step", (), "I64"))}
    with torch.no_grad():
        for rank, manifest in zip(layout.ranks, manifests):
            inventory = layout.inventory(source_configuration, rank)
            local_expected = {n: TensorSpec(tuple(v["local_shape"]), "F32" if v["dtype"] == "float32" else "BF16")
                              for n,v in inventory.items()}
            tensors = load_tensor_group(Path(root)/manifest.transaction_id, manifest.groups["model"],
                                        local_expected, device="cpu")
            for name, value in tensors.items():
                if not bool(torch.isfinite(value).all()):
                    raise CheckpointError("nonfinite checkpoint model tensor: " + name)
                spec = inventory[name]
                axis, start, stop = spec["axis"], spec["start"], spec["stop"]
                key = (axis, start, stop)
                if name not in assembled:
                    assembled[name] = torch.empty(expected[name].shape, dtype=value.dtype, device="cpu")
                    slices[name] = set()
                target = assembled[name] if axis is None else assembled[name].narrow(axis, start, stop-start)
                if key in slices[name]:
                    # These are declared replicas, never another logical slice.
                    if not torch.equal(target, value):
                        raise CheckpointError("divergent checkpoint model replicas: " + name)
                else:
                    for old_axis, left, right in slices[name]:
                        if axis != old_axis or axis is None or max(start, left) < min(stop, right):
                            raise CheckpointError("overlapping logical parameter slices: " + name)
                    target.copy_(value)
                    slices[name].add(key)
            router = load_tensor_group(Path(root)/manifest.transaction_id, manifest.groups["router"],
                                       router_inventory, device="cpu")
            for layer in range(48):
                bias = router[f"layers.{layer}.bias"]
                if (not bool(torch.isfinite(bias).all()) or
                        bool((bias.abs() > config.get("router.balancing.bound")).any()) or
                        router[f"layers.{layer}.completed_step"].item() != manifest.successful_step):
                    raise CheckpointError("invalid committed inference balancing state")
            if router_reference is None:
                router_reference = router
            elif any(not torch.equal(router_reference[n], v) for n,v in router.items()):
                raise CheckpointError("divergent checkpoint router replicas")
            del tensors
        if set(assembled) != set(expected):
            raise CheckpointError("missing logical parameters or experts")
        for name, owned in slices.items():
            if owned == {(None, None, None)}:
                continue
            ordered = sorted(owned, key=lambda s: s[1])
            axis = ordered[0][0]
            if (ordered[0][1] != 0 or ordered[-1][2] != expected[name].shape[axis] or
                    any(a[2] != b[1] for a,b in zip(ordered, ordered[1:]))):
                raise CheckpointError("incomplete global parameter reconstruction: " + name)
        # Allocate/copy auxiliary state before resident model mutation.
        bias = torch.stack([router_reference[f"layers.{i}.bias"] for i in range(48)]).to(model.embeddings.weight.device)
        frozen_bias = bias.clone()
        model.checkpoint_identity = None
        model.checkpoint_load_state = "RECOVERY_REQUIRED"
        restore_module_state(model, assembled)
        model.inference_balancing_bias = frozen_bias
        model.checkpoint_identity = identity
        model.checkpoint_load_state = "COMMITTED"
    return {"checkpoint_identity": identity, "successful_step": manifests[0].successful_step,
            "balancing_bias": bias, "source_configuration_identity": source_configuration.identity,
            "resident_configuration_identity": config.identity, "approval_reference": approval_reference}


def restore_training_set(engine, root: str | Path, set_id: str, *, rng_devices: tuple[int, ...]):
    """Restore one admitted DP/EP/TP/CP rank from a complete committed checkpoint set.

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
    from .transactions import read_training_set
    from .serialization import capture_optimizer_state, restore_optimizer_state

    if engine.state.phase not in ("COMMITTED", "RECOVERY_REQUIRED"):
        raise CheckpointError("restore requires idle or failed engine")
    engine.state.phase = "RECOVERY_REQUIRED"
    body, manifests = read_training_set(root, set_id, configuration=engine.config, ranks=engine.group.ranks)
    engine.group.agree(body, device=engine.device)
    manifest = manifests[engine.group.rank]
    meta = manifest.metadata
    execution = engine.model.expert_execution
    expected_owner = {"mode": "full_replica_dp" if execution is None else "expert_parallel",
                      "ranks": list(engine.group.ranks), "rank": engine.group.rank,
                      "expert_identity": None if execution is None else execution.ownership.identity}
    if engine.parallel is not None:
        expected_owner = engine.parallel.ownership_metadata()
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
    if engine.parallel is not None:
        expected_model = {name: TensorSpec(tuple(spec["local_shape"]), {"float32": "F32", "bfloat16": "BF16"}[spec["dtype"]])
                          for name, spec in engine.parameter_layout.items()}
        if meta.get("parameter_layout") != engine.parameter_layout:
            raise CheckpointError("TP slice inventory differs; implicit repartitioning forbidden")
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
    # Verify every actual replica group before mutation, including TP-replicated
    # norm/router values. Tensor shards are never compared to other TP slices.
    def compare(value, group, label):
        local = value.to(engine.device)
        reference = local.clone()
        dist.broadcast(reference, src=group.ranks[0], group=group.process_group)
        equal = group.integer_sum(torch.tensor(int(torch.equal(local, reference)), dtype=torch.int64, device=engine.device))
        if equal.item() != group.size:
            raise CheckpointError("replicated checkpoint values differ: " + label)
    for name, value in sorted(loaded["master"].items()):
        if engine.parallel is None:
            replica_groups = [] if name in engine.owned_names else [engine.group]
        else:
            replica_groups = [engine.parallel.replica_group(name)]
            if engine.parameter_layout[name]["axis"] is None:
                replica_groups.append(engine.parallel.groups["tensor"])
        for group in replica_groups:
            compare(value, group, name)
            for entry in states.get(name, {}).values():
                compare(loaded["optimizer"][entry["tensor_name"]], group, entry["tensor_name"])
    for name, value in sorted(loaded["router"].items()):
        compare(value, engine.group, name)
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
