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


## Mixed integration update — 27.10.2026

This section supersedes the INCOMPLETE/NEXT statements in the earlier increment above. Source paths for the Order 009 distributed-training requirements are now implemented and statically reviewed. Numerical correctness, deadlock freedom on the actual backend, runtime compatibility and recovery durability remain UNVERIFIED. This is not Training Ready or a runtime certification.

### IMPLEMENTED: logical ownership and model integration

ParallelLayout admits a complete, unique Cartesian coordinate inventory (DP, EP, CP, TP), explicit global ranks, a [48,128] expert-owner table and approval reference. No physical topology is selected. The caller supplies seven actual process-group roles, including explicit singleton groups: world; tensor (vary TP); expert (vary EP); context (vary CP); shared_parameters (vary DP/EP/CP); expert_parameters (vary DP/CP); source_replicas (vary CP/TP). Membership and order must match the approved layout. No implicit world_size=1 fallback exists.

MixedExecution connects these groups to the canonical model and TrainingEngine. TP remains weight-sharded with full replicated activations/logits. Input embeddings own hidden columns; non-router matrices own output rows. Router and norms are explicitly replicated. TP divides eight KV heads. Logical names and parameter counts are unchanged. Initialization still samples the complete per-name logical stream at the TP source before distributing slices; no independent shard-seed approximation is used.

### IMPLEMENTED: CP targets and uneven sources

Each (DP,EP) source owns a disjoint manifest interval. Its CP/TP replicas read the same verified source and agree on batch content, availability and pending cursor. CP cuts use offsets floor(N*c/C); zero-length real partitions are permitted. Prediction position i owns the actual target i+1, including across a cut, only where the packed document mask permits it. BOS/EOS/separator semantics remain those of DocumentPacker.

Only TP coordinate zero contributes token counts, target counts, summed loss and router load to world statistics. CP positions and EP/DP sources are distinct. L = sum(real eligible cross-entropies)/N_targets. Gradient accumulation sums numerators before division by the global target count; it never averages microbatch means.

When one EP source is exhausted, its rank retains a zero-length real-work graph and continues all required communication, including processing other sources' expert assignments. No document, loss target or fake expert is inserted. All-source exhaustion exits before optimizer mutation. A final partial accumulation with real targets may commit; a zero-target or over-budget transaction fails without advancing committed cursors. Duplicate-only suffixes are replayable after abort, not silently committed as training tokens.

EP transport exchanges fixed-length count vectors, then processes nonempty P2P edges in lexicographic source/destination order using isend/irecv and Work.wait. Backward reverses the send/receive counts. No zero-count payload support is presumed. CP uses padded transport buffers trimmed to actual lengths; padding never becomes a model token. All-empty CP gathers and all-empty TP feature gathers skip payload collectives consistently within their group. CP remains gathered-KV exact attention, not ring attention.

### IMPLEMENTED: gradients and state transitions

Shared parameter gradients SUM over DP/EP/CP replicas at fixed TP coordinate. Expert gradients SUM over DP/CP at fixed expert owner and TP coordinate; inverse EP transport already supplies contributions from EP sources. TP copy backward sums input derivatives; feature-gather backward selects its owned slice rather than summing duplicate losses.

For global clipping, each actual tensor shard contributes once: replica-group leader, and for unsharded norm/router tensors TP coordinate zero only. norm = sqrt(world_sum(sum_selected_shards((gradient/N_targets)^2))). The common clipping factor applies to every stored replica. FP32 masters and router precision are retained.

Router snapshots remain frozen through forward/recomputation. Load events are recorded only for original forward. All required gradients/statistics complete before optimizer mutation. Completion acknowledgment precedes scheduler/cursor/balancing commit. Any post-mutation failure leaves RECOVERY_REQUIRED. A rank exception may strand peers in a collective: the external launcher must terminate/restart the group. No in-process rescue, atomic rollback or automatic retry is claimed.

### IMPLEMENTED: Safetensors distributed resume

Snapshots now carry exact logical/local shape, dtype, axis and start/stop slice inventory plus full approved layout identity. Model, FP32 master, optimizer, router and RNG tensors remain Safetensors; metadata and indexes remain JSON. publish_training_set requires the actual Configuration for mixed ownership. Complete rank coverage and canonical per-rank inventories reject missing/duplicate shards or implicit repartitioning. CP/TP source cursors must agree.

restore_training_set checks the whole set, identities, scheduler/optimizer policies, cursor partitions, shapes/dtypes, model/master agreement, and replicated master/optimizer/router values before applying state. Restoration rebuilds the verified real-data prefix for deduplication. A mutation failure remains RECOVERY_REQUIRED. Publication still requires an explicit AtomicPublicationGuarantee; storage crash durability is UNKNOWN. The three earlier checkpoint P1 fixes are preserved.

### Interface boundary and remaining decisions

Single-sequence inference remains the resident-model path. It explicitly rejects distributed execution; the single-transaction model loader does not silently consolidate TP shards. Mixed training resume uses restore_training_set with the same approved layout. Distributed generation and cross-layout export are not claimed by this update.

Execution remains BLOCKED on approved initialization seed/scales, tokenizer registry/artifact and corpus, operational training settings, logical group manifest, actual GM/backend compatibility and verified storage contract. Existing unresolved chat grammar/output-policy decisions and document-splitting policy remain explicit. No null values were filled. No GPU, server or infrastructure was selected.

### FILES CHANGED in this increment

- src/shinra/checkpoint/recovery.py
- src/shinra/checkpoint/transactions.py
- src/shinra/data/packing.py
- src/shinra/distributed/communication.py
- src/shinra/distributed/context_parallel.py
- src/shinra/distributed/ownership.py
- src/shinra/distributed/parallelism.py
- src/shinra/model/modeling.py
- src/shinra/training/engine.py
- src/shinra/training/optimizer.py

Documentation: this report, IMPLEMENTATION_STATUS.md, MODULE_CONTRACTS.md, D07_PRETRAINING.md, D08_PROJECT_STRUCTURE.md, D09_DEPENDENCIES.md, ARCHITECTURE_CHANGELOG.md. No new Python module, dependency or YAML.

### STATICALLY REVIEWED / NOT EXECUTED

Source-only AST/import-graph review, changed-interface comparison and Git whitespace review. Logical parameter formula: 2*131072*2560 + 48*(2*2560*4096 + 2*2560*1024 + 128*3*2560*800 + 2560*128 + 2*2560 + 40*128) + 2560 = 39,694,338,560. Substituting 8 active experts for 128 gives 4,304,898,560 under the conventional active-count definition. No parameter tensor was materialized.

No model initialization, forward/backward, training, inference, checkpoint round-trip, tests, synthetic inputs, corpus/model downloads, installation, deployment or GPU planning. Static review cannot establish distributed numerical equivalence or absence of runtime deadlocks.
