# Order 008 — publication and distributed continuation

26.10.2026. Static source changes only. Source Complete: NOT ACHIEVED. Training Ready: NO-GO.

## Publication confirmed

Order 007 was published to origin/main as 355e37fae2d461b858ae1e9566e666edb0fdaf35. Git reported fast-forward from 3f3c8b4; ls-remote matched HEAD and the working tree was clean immediately after that push. The commit changed 48 files and its tree contains 59 Python files, not 59 newly created files.

## EP integration

ExpertExecution binds the canonical router/dispatch/combine path to explicit ExpertOwnership and a real multi-member Group. Model construction verifies ownership against the single configuration's manifest. ModuleDict keys remain numeric logical expert IDs: layers.L.moe.experts.E.{gate,up,down}.weight. Owners allocate only their experts; the union remains 128 experts per layer. Top-8 IDs and differentiable combination weights remain unchanged. All-to-all transports assignment inputs and returns outputs; explicit autograd reverses both exchanges. Duplicate global token/expert pairs fail.

Training supports EP with disjoint whole-document sources on the same logical group. Shared gradients are summed; owner-only expert gradients already include all source contributions and are not summed twice. Global norm counts replicated tensors once and each owned tensor once, including members owning no experts. Router counts are recorded once at original forward, summed over unique-token sources, and updated only after optimizer mutation acknowledgement. Recomputation disables early stopping so ranks replay complete collective sequences.

Uneven EP source exhaustion explicitly aborts the transaction and restores committed cursors. No tail samples are silently dropped; an empty-shard scheduling strategy is not implemented. Rank exceptions can leave peers inside collectives; launcher termination/restart remains required.

## CP integration boundary

ContextExecution is connected through model/decoder/attention. Global metadata validates unique UIDs, contiguous document positions and total context range. QK-Norm and RoPE happen before KV exchange; gathered keys are not rotated again. The existing exact attention backward combines KV gradients through real all-reduce before returning local slices. This implementation gathers KV and makes no ring-attention or memory-efficiency claim.

Context-sharded training remains explicitly blocked: shifted targets across shard boundaries, streaming cursor ownership and mixed-group orchestration are not integrated. Single-sequence inference rejects distributed models instead of invoking mismatched collectives.

## Complete-set checkpoint and resume source

write_training_snapshot retains Safetensors for model, FP32 masters, optimizer, router and RNG. EP model/master inventories include only locally owned experts plus replicated parameters. Ownership mode, rank list and ownership identity are JSON metadata.

publish_training_set accepts one already committed transaction per logical rank. It verifies hashes through existing recovery, matching configuration/architecture/progress/scheduler/balancing/source revision, required tensor groups and nonoverlapping cursor intervals. A separate atomic JSON marker admits the entire set only after those checks. This is metadata, not a custom tensor container. All tensor payloads remain Safetensors.

restore_training_set operates on an idle or failed engine. It validates the complete set before applying the local state, checks fixed logical shapes/dtypes and exact optimizer groups, step/moment state, scheduler, tokenizer/corpus, ownership, router bounds and explicit RNG devices. The real corpus prefix is replayed to reconstruct deduplication and cursor state. Model values must equal dtype-converted FP32 masters. Replicated masters, optimizer states and router values are compared across ranks before mutation. Only after state restoration and group acknowledgement does the engine become COMMITTED.

Any restore error leaves engine state RECOVERY_REQUIRED. Partial tensor mutation is not rolled back or represented as atomic recovery. Storage publication requires AtomicPublicationGuarantee supplied by the caller; directory-entry crash durability remains UNKNOWN. No save, publication, restore, RNG operation or collective was executed during development. The current constructor starts from idle zero-step components; resume is a separate explicit operation, not arbitrary injection of successful_step.

## Remaining implementation work

- TP model integration and physical tensor-shard serialization: only linear partition primitives exist. Partition-invariant initialization and combined EP/TP groups remain unresolved source work.
- CP training loss/stream integration and mixed EP/TP/CP ownership.
- EP uneven-source scheduling without dummy data or loss of tail documents.
- Runtime numerical/gradient/collective/recompute and checkpoint verification in a separately authorized environment.
- Approved initialization, tokenizer registry/artifact, corpus, operational settings, actual GM compatibility and storage contract remain absent. No infrastructure chosen.

## Static evidence / restrictions

AST parsed all 59 Python files without importing SHINRA; internal import modules resolve and no static import cycles were found. Source review covered collective ordering, logical names, owner gradient handling, global clipping and checkpoint producer/consumer contracts. git diff --check is clean. These checks are not runtime validation. One YAML retained; no architecture dimensions or null operational values changed; no new dependencies.

No model initialization, forward/backward, training, inference, tests, synthetic data, downloads, installation, deployment or GPU planning.

## Changed source files

- `src/shinra/checkpoint/recovery.py`
- `src/shinra/checkpoint/transactions.py`
- `src/shinra/distributed/context_parallel.py`
- `src/shinra/distributed/expert_parallel.py`
- `src/shinra/distributed/ownership.py`
- `src/shinra/distributed/parallelism.py`
- `src/shinra/inference/generation.py`
- `src/shinra/model/attention.py`
- `src/shinra/model/decoder.py`
- `src/shinra/model/modeling.py`
- `src/shinra/moe/layer.py`
- `src/shinra/training/engine.py`
- `src/shinra/training/optimizer.py`
