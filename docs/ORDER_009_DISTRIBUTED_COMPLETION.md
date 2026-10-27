# Order 009 — distributed source continuation

27.10.2026. Base: 1cd693ea7531bbb0e98f443affb1c2bb268dc58a.

**SOURCE COMPLETE: NOT ACHIEVED. TRAINING READY: NO-GO.**

## IMPLEMENTED

TensorExecution is now connected to canonical model construction, embeddings, every decoder attention/MoE block and untied LM head. There is no alternate model or Dense FFN.

Input embedding weights are partitioned on hidden columns. All other non-router matrices are partitioned on output rows. Q/K/V shard boundaries preserve complete global heads and the four-query-head-per-KV relationship: admitted TP membership divides eight KV heads (2,4,8). Attention output, expert gate/up/down and full-vocabulary LM output use real feature all-gathers. Full activations remain replicated; this is not an activation-memory-efficiency claim. Router weights and all RMSNorm/QK-Norm weights are explicitly replicated.

Feature-gather backward selects the owned feature slice without summing replicated losses. Before sharded linear operations, an explicit autograd copy sums input gradients across TP contributions. Embedding backward selects its owned hidden slice. Names remain canonical; TensorExecution.inventory records global/local shape, axis and slice boundaries without creating model tensors.

Initialization preserves the existing per-logical-name seed, normal FP32 sampling, residual scaling and storage cast. One source member generates the complete logical tensor and broadcasts it; each member retains its defined slice. No independent shard RNG streams are guessed. This deliberately materializes a full temporary logical tensor: memory efficiency is not claimed. Fixed-stack/source-generator reproducibility is the scope; cross-device bitwise equivalence remains unverified. No seed, scale or other operational null was filled.

Added ContextBatch and partition_context_batch: prediction-position ownership is explicit. A shard's final prediction may own the real next token from the following shard. Existing whole-document loss masks exclude cross-document and separator targets. No fabricated last-token label or padding target is inserted. context_causal_loss_sum handles only the selected real targets and returns a summed numerator/count for later global normalization.

## INCOMPLETE / BLOCKED

TP is integrated into canonical model source, but replica-aware training coordination, optimizer/global clipping ownership and TP checkpoint inventories are not integrated. TrainingEngine explicitly rejects TP models. Existing model checkpoint loader rejects implicit TP repartitioning. Single-sequence generation rejects distributed models.

Mixed TP/EP/CP construction is rejected until group membership/ownership validation exists. CP target-partition and objective algorithms do not yet constitute CP training: streaming transaction ownership, recomputation/collective ordering and global normalization still need coordinator integration.

Uneven EP exhaustion still aborts the transaction; empty-source participation and zero-size backend admission remain unfinished. Mixed DP/EP/TP/CP checkpoint publication and restore remain unfinished. Existing Order 008 DP/EP resume is preserved and does not pretend to support TP shards.

These are remaining implementation tasks, not evidence that the user has withheld source authorization. No new GO is required to continue them. Unknown runtime/backend behavior is separately unverified.

## ARCHITECTURE COMPLIANCE

Logical total 39,694,338,560 / conventional active 4,304,898,560 unchanged. All 48 blocks retain 128 routed experts, Top-8, zero shared experts, width800, GQA32/8/128, vocab131072 and context262144. Tensor slices do not add logical parameters. One canonical YAML unchanged; no new dependencies or pretrained assets.

## STATICALLY REVIEWED

AST parsed all 59 Python modules without importing SHINRA. No unresolved internal import modules or import cycles detected. git diff --check clean. Reviewed constructor propagation, replicated-vs-sharded weight roles, autograd adjoints and real CP boundary-label ownership. These are source checks, not numerical certification.

## NOT EXECUTED

No initialization, forward/backward, training, inference, checkpoint round-trip, tests, datasets, synthetic inputs, dependency installation, downloads, deployment or GPU planning. No performance, VRAM, runtime-success or training-readiness claim.

## FILES CHANGED

- `src/shinra/checkpoint/recovery.py`
- `src/shinra/core/initialization.py`
- `src/shinra/data/packing.py`
- `src/shinra/distributed/context_parallel.py`
- `src/shinra/distributed/parallelism.py`
- `src/shinra/inference/generation.py`
- `src/shinra/model/attention.py`
- `src/shinra/model/decoder.py`
- `src/shinra/model/embeddings.py`
- `src/shinra/model/lm_head.py`
- `src/shinra/model/modeling.py`
- `src/shinra/model/normalization.py`
- `src/shinra/moe/experts.py`
- `src/shinra/moe/layer.py`
- `src/shinra/moe/router.py`
- `src/shinra/training/engine.py`
- `src/shinra/training/objective.py`

Documentation updated: this report, IMPLEMENTATION_STATUS.md, MODULE_CONTRACTS.md, D08_PROJECT_STRUCTURE.md, D07_PRETRAINING.md and ARCHITECTURE_CHANGELOG.md. Historical reports retained.

## NEXT

Implement explicit mixed-group ownership and TP-aware optimizer/reduction/checkpoint inventories, then connect ContextBatch/causal loss to collective-safe stream transactions. Complete uneven-source scheduling with backend admission and no dummy data. Finish full mixed-state recovery and repeat static integration review before considering Source Complete.
