"""Explicit full-replica DP optimizer transactions; no implicit local mode.

TP/EP/CP kernels are separate primitives and are not silently substituted for
this full-replica orchestration. Physical topology is always caller-owned.
"""
from contextlib import contextmanager
import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

from shinra.core.validation import require_operation
from shinra.model.modeling import ModelOutput
from shinra.distributed.communication import Group
from shinra.distributed.parallelism import synchronize_gradients
from .objective import causal_loss_sum
from .precision import validate_model_precision


def training_forward(model, batch, bias, *, recomputation: bool):
    device = model.embeddings.weight.device
    ids, positions, segments, uids = (torch.tensor(values, dtype=torch.int64, device=device)
                                     for values in (batch.token_ids, batch.positions, batch.segment_ids, batch.token_uids))
    if not recomputation:
        return model(ids, positions, segments, uids, bias), ids
    hidden = model.embeddings(ids)
    selections, loads = [], []

    def make_run(block, snapshot):
        saved = {}
        state = {"replay": False}

        @contextmanager
        def original():
            state["replay"] = False
            yield

        @contextmanager
        def replay():
            state["replay"] = True
            try:
                yield
            finally:
                state["replay"] = False

        def run(value):
            output, assignments, counts = block(value, positions, segments, uids, snapshot,
                                                 replay_ids=saved.get("assignments") if state["replay"] else None)
            if not state["replay"]:
                saved["assignments"] = assignments.detach().clone()
                saved["counts"] = counts.detach().clone()
            return output

        return run, lambda: (original(), replay()), saved

    for index, block in enumerate(model.layers):
        run, contexts, saved = make_run(block, bias[index])
        hidden = checkpoint(run, hidden, use_reentrant=False, context_fn=contexts, preserve_rng_state=True)
        selections.append(saved["assignments"])
        loads.append(saved["counts"])
    return ModelOutput(model.lm_head(model.final_norm(hidden)), tuple(selections), tuple(loads)), ids


class TrainingEngine:
    def __init__(self, config, model, optimizer, scheduler, balances, stream, state, group: Group, telemetry):
        require_operation(config, "train")
        validate_model_precision(model)
        if config.get("training.gradient_clip_norm_type") != "l2":
            raise ValueError("explicit L2 clipping policy required")
        if len(balances) != 48 or scheduler.completed_step != state.successful_step:
            raise ValueError("training component progress/inventory mismatch")
        self.config, self.model, self.optimizer, self.scheduler = config, model, optimizer, scheduler
        self.balances, self.stream, self.state, self.group, self.telemetry = balances, stream, state, group, telemetry
        self.device = model.embeddings.weight.device
        group.validate_device(model.embeddings.weight)
        group.agree({"configuration": config.identity,
                     "manifest": stream.reader.manifest.identity,
                     "step": state.successful_step,
                     "parameters": [(name, list(value.shape), str(value.dtype))
                                    for name, value in sorted(optimizer.masters.items())]}, device=self.device)
        if state.successful_step != 0 or state.consumed_tokens != 0 or optimizer.optimizer.state:
            raise RuntimeError("BLOCKED: coordinated training resume admission is not implemented")
        if model.configuration_identity != config.identity:
            raise ValueError("model and training configuration identities differ")
        if set(optimizer.parameters) != set(dict(model.named_parameters())) or any(
                optimizer.parameters[name] is not value for name, value in model.named_parameters()):
            raise ValueError("optimizer must own this model's canonical parameters")
        # Fresh-run replicas use the group's first logical member as source.
        # This assigns no physical infrastructure and never assumes cross-device
        # RNG produces identical weights. Resume must use a separate admission.
        for balance in balances:
            if balance.checkpoint_metadata()["completed_step"] != 0 or bool((balance.bias != 0).any()):
                raise ValueError("fresh training requires committed zero balancing state")
        with torch.no_grad():
            for name in sorted(optimizer.masters):
                master = optimizer.masters[name]
                dist.broadcast(master, src=group.ranks[0], group=group.process_group)
                optimizer.parameters[name].copy_(master.to(optimizer.parameters[name].dtype))
        # Validate disjoint, complete manifest partitions without assigning them.
        bounds = torch.tensor([stream.committed.start_ordinal, stream.committed.end_ordinal], dtype=torch.int64, device=self.device)
        partitions = [torch.empty_like(bounds) for _ in group.ranks]
        dist.all_gather(partitions, bounds, group=group.process_group)
        intervals = sorted(tuple(p.cpu().tolist()) for p in partitions)
        if intervals[0][0] != 0 or intervals[-1][1] != len(stream.reader.manifest.documents) or any(a[1] != b[0] for a,b in zip(intervals,intervals[1:])):
            raise ValueError("DP corpus partitions must be disjoint and cover manifest")

    def step(self, attempt: str) -> dict:
        self.state.begin(attempt)
        mutated = False
        try:
            self.stream.begin(attempt)
            self.optimizer.clear()
            snapshots = torch.stack([b.begin(attempt, completed_step=self.state.successful_step) for b in self.balances])
            self.model.train()
            local_tokens, local_targets, numerator = 0, 0, 0.0
            for microbatch in range(self.config.get("training.accumulation_steps")):
                batch = self.stream.next_batch()
                if batch is None:
                    break
                if len(batch.token_ids) > self.config.get("training.microbatch_tokens"):
                    raise ValueError("packed batch exceeds explicit microbatch token budget")
                output, ids = training_forward(self.model, batch, snapshots,
                                               recomputation=self.config.get("training.recomputation"))
                mask = torch.tensor(batch.loss_mask, dtype=torch.bool, device=self.device)
                loss, targets = causal_loss_sum(output.logits, ids, mask)
                loss.backward()
                self.optimizer.accumulate()
                for layer, balance in enumerate(self.balances):
                    balance.record(attempt, f"{attempt}:{microbatch}:{layer}", output.expert_counts[layer], recomputation=False)
                local_tokens += len(batch.token_ids)
                local_targets += targets
                numerator += float(loss.detach().item())
            totals = self.group.integer_sum(torch.tensor([local_tokens, local_targets], dtype=torch.int64, device=self.device))
            tokens, targets = totals.cpu().tolist()
            if targets <= 0 or self.state.consumed_tokens + tokens > self.config.get("training.token_budget"):
                raise RuntimeError("no global targets or token budget exceeded; no update")
            for balance in self.balances:
                counts = self.group.integer_sum(torch.tensor(balance.local_counts(attempt), dtype=torch.int64, device=self.device))
                balance.prepare(attempt, synchronized_counts=tuple(counts.cpu().tolist()), unique_tokens=tokens)
            gradients = synchronize_gradients(self.optimizer.masters, self.optimizer.accumulated, self.group)
            grad_norm = self.optimizer.install_gradients(gradients, global_tokens=targets,
                                                         max_norm=self.config.get("training.max_grad_norm"))
            rate = self.scheduler.next_lr()
            self.state.prepare()
            for balance in self.balances:
                balance.start_optimizer_mutation(attempt)
            self.state.start_mutation()
            mutated = True
            self.optimizer.step(rate)
            # All members must complete tensor mutation before metadata advances.
            self.group.integer_sum(torch.ones((), dtype=torch.int64, device=self.device))
            step = self.state.successful_step + 1
            for balance in self.balances:
                balance.commit(attempt, completed_step=step)
            self.scheduler.commit(step)
            self.stream.commit(attempt)
            self.state.commit(tokens)
            summed_loss = self.group.sum(torch.tensor(numerator, dtype=torch.float64, device=self.device)).item()
            event = {"successful_step": step, "tokens": tokens, "targets": targets,
                     "loss": summed_loss/targets, "gradient_norm": float(grad_norm.item()),
                     "learning_rate": rate, "duplicate_documents": self.stream.duplicates}
            self.telemetry.emit("optimizer_commit", event)
            return event
        except Exception:
            if mutated:
                self.state.phase = "RECOVERY_REQUIRED"
            else:
                self.optimizer.clear()
                for balance in self.balances:
                    if balance.attempt == attempt and balance.phase in ("OPEN", "PREPARED"):
                        balance.abort(attempt)
                if self.stream.attempt == attempt:
                    self.stream.abort(attempt)
                if self.state.phase in ("OPEN", "PREPARED"):
                    self.state.abort()
            # A failed rank may leave peers inside a collective. External launcher
            # must terminate the group; never attempt a guessed rescue collective.
            raise
