# Order 010 — final source integration review

29.10.2026. Authority: NULLXES / Founder. Baseline: `ed34d8671f520d0ab1c946eb878d501356e35474`.

**SOURCE COMPLETE: ACHIEVED for the mandatory Order 010 source scope.**
**RUNTIME UNVERIFIED. TRAINING READY: NO-GO.**

This is source-level acceptance, not numerical certification, a training result or a storage durability claim. The initial 17 staged files were preserved and included in this continuation. Orders 007–009 remain historical records; this document supersedes their unfinished-work lists for the interfaces described below.

## IMPLEMENTED

The canonical modules now connect Tokenizer → Data/Packing → Model/MoE → mixed DP/EP/TP/CP → Training → Safetensors → Resume → resident inference loading. There is no alternate model, dependency addition, new YAML or runnable test suite. Existing TP initialization and expert/attention mathematics are retained.

## P1/P2 REMEDIATION

`core.validation.training_sequence_budget` defines `training.microbatch_tokens` as the explicit positive token limit for ONE complete logical source sequence, before CP. It must not exceed the fixed architectural context. There is no rank-dependent reinterpretation or guessed default.

`DocumentPacker.capacity` now uses this budget; `context_capacity` remains the architecture limit and the stride for stable token UIDs. `PackedStream` already packs against `packer.capacity`, including separators. `pack` checks the combined length before constructing batch lists. Training admission checks packer/config agreement and the complete batch before partitioning; a CP batch carries the same global length for the engine's secondary check.

An oversized whole document is rejected with its source/document identity, ordinal, required length and budget. D06 does not approve a concrete segmented-continuation policy, so none is invented. The committed cursor remains unchanged; the engine aborts the transaction and restores the prefix. No truncation, skipped retained document, fabricated label or automatic budget increase occurs. Actual document tokenization remains whole-fragment BPE, preserving merges across byte chunks.

`ShinraForCausalLM.validate_inputs` is the common boundary for ordinary forward and the original recomputation forward. It checks shape/dtype/device, vocabulary/position ranges, nonnegative unique UIDs, context limits, finite nontrainable FP32 balancing snapshots, segment continuity, replay IDs, mixed group membership, TP input agreement and CP global range ownership. CP lengths must match the declared floor-divided global ranges.

Training calls this boundary exactly once outside checkpoint replay closures. Block replay uses captured original Top-8 IDs and frozen bias; load events are recorded only after the original forward/backward microbatch. Full block recomputation remains explicit with early stopping disabled. TP routing ID/weight agreement protects the per-expert communication schedule during original execution and replay.

## DISTRIBUTED STATUS

`ParallelLayout` describes a complete Cartesian logical inventory `(DP, EP, CP, TP)` with explicit ranks and `[48,128]` owner table. `MixedExecution` requires caller-created real groups, including singleton groups. No process count, device or physical topology is selected here.

| Role | Varying coordinates | Responsibility |
|---|---|---|
| world | DP/EP/CP/TP | Step agreement, unique counts, global norm |
| tensor | TP | Tensor slices and replicated activations |
| expert | EP | Assigned-token transport to expert owners |
| context | CP | Global metadata and gathered KV |
| shared_parameters | DP/EP/CP | Sum shared-parameter gradients at fixed TP |
| expert_parameters | DP/CP | Sum owned-expert gradients at fixed EP/TP |
| source_replicas | CP/TP | Same source content, availability and cursor |

Membership/order/backend must agree with the approved layout. A caller must create groups in a consistent global order, use a serial collective schedule and provide timeout/supervision. Failure of a rank can strand peers; the external launcher must terminate/restart the entire failed group. No rescue collective or in-process rollback is promised.

TP partitions embeddings on hidden columns and other non-router matrices on output rows; router/norms are explicitly replicated. TP degrees divide eight KV heads. Logical parameter names remain unchanged. Full activations/logits are replicated; this implementation does not claim activation-memory efficiency.

The source adjoints were reviewed: output-feature gather returns the owning gradient slice; copy-before-sharded-linear sums input derivatives over TP. Row-partition linear sums partial outputs and returns the replicated logical upstream gradient to each partition. Embedding gather backward selects the stored hidden columns. Q/K/V/O, expert gate/up/down and LM head retain full logical operations. Norm/router copies are not summed as independent TP losses.

CP owns prediction positions, including the real next-token label across a shard boundary. QK-Norm and RoPE precede KV exchange; cached/gathered keys are not rotated twice. The causal mask uses global document positions and segments. KV-gather backward sums contributions from all query partitions before returning the owned key/value slice. This is gathered-KV exact blockwise attention, not ring attention.

Uneven EP sources retain zero-length real-work graphs when exhausted. They still receive/process assigned work from other sources and participate in inverse transport/backward. P2P edges with nonzero payloads execute in lexicographic source/destination order; fixed-size count exchange and a completion control collective include every member. No dummy token, fake expert or loss target is inserted. All-source exhaustion ends before optimizer mutation. Duplicate-only suffixes remain replayable after abort.

Only TP coordinate zero contributes each source/context partition to world token counts, target counts, loss numerator and router loads. The objective is the sum of eligible full-vocabulary cross-entropies divided by the number of unique eligible targets. Accumulation sums numerators, not microbatch means. Expert gradients do not receive a second EP SUM after inverse transport.

Global clipping counts each tensor shard once: replica-group leader, and TP coordinate zero for unsharded router/norm tensors. FP32 synchronized gradients are checked for finiteness before mutation. FP32 masters remain local to the actual owned tensors. Uniform decay/zero-gradient behavior is explicit in the existing AdamW adapter and requires configuration approval.

## SAFETENSORS STATUS

`write_training_snapshot`, `publish_training_set`, `read_training_set` and `restore_training_set` use the existing checkpoint engine. Tensor payloads are exclusively Safetensors; JSON carries identities, layouts, scheduler/cursors, options and publication markers.

The rank set must cover the exact rank inventory at a single committed optimizer step. Model/master shapes and dtypes derive from canonical logical names and ownership. Optimizer groups must cover those names exactly; every nonzero step requires scalar FP32 step plus FP32 first/second moments for every owned parameter. Router and RNG inventories are checked. Scheduler, balancing policy, optimizer coefficients and corpus/tokenizer identities must match the source configuration. CP/TP replicas must carry identical cursors. `corpus_documents` closes the interval coverage check at the last document rather than only checking internal gaps.

Complete-set publication now requires the exact source Configuration for all admitted layouts. Older snapshots missing the strengthened metadata are rejected; no implicit migration is implemented. No actual checkpoint existed or was migrated during this work.

Resume verifies file hashes, layouts, source identities, progress, optimizer policy/tensors, model/master agreement, finite/bounded router state and all declared replicated master/optimizer/router values before applying local state. It reconstructs deduplication from the verified real prefix. RNG and scheduler/cursors are restored, then the group acknowledges completion. Any restoration failure leaves `RECOVERY_REQUIRED`; partial mutations are not rolled back or called atomic.

Writers must freeze state during snapshot/publication, and readers require immutable checkpoint storage throughout admission/loading. Publication still requires `AtomicPublicationGuarantee` for same-directory atomic replacement. Directory-entry crash durability, object-store publication and a concrete launcher/storage backend remain unapproved. A per-rank COMMITTED transaction alone is not a resumable distributed set.

## DISTRIBUTED-TO-INFERENCE CONTRACT

`checkpoint.recovery.load_resident_from_training_set` explicitly reconstructs an approved mixed set into the canonical resident model. The caller supplies source and resident Configuration objects, a reconstruction approval reference and `expected_set_identity` (SHA-256 of canonical sorted compact JSON for the set marker). These are in-memory snapshots of the same configuration schema, not additional repository YAML files.

Source configuration must match the checkpoint exactly; source/destination architecture identities must match. That identity includes model/initialization, attention, RoPE, routing and tokenizer identity. Only operational differences are admitted by this explicitly invoked conversion. Training resume still requires the original exact configuration/layout.

Every parameter slice is reconstructed at its declared axis/offset; declared replicas must have identical values. All experts and full tensor intervals must be present, with no overlaps or gaps. Router biases/steps are checked across ranks. Reconstruction and validation finish on CPU before resident model mutation. This deliberately materializes the logical model; bounded memory and performance are not claimed.

On mutation failure, model identity remains invalid and load state is `RECOVERY_REQUIRED`. Success returns the committed router bias and a set-level checkpoint identity; generation rejects a different balancing snapshot. No pretrained substitution, implicit repartitioning, distributed generation, training-topology migration or new tensor format is introduced. Legacy non-mixed sets are explicitly outside this conversion entrypoint.

## ARCHITECTURE COMPLIANCE

D01/D02: 48 decoder blocks, hidden2560, GQA32/8/128, 128 routed experts/block, Top-8, shared0, SwiGLU width800, vocabulary131072, untied embeddings and context target262144 remain fixed. Logical totals remain **39,694,338,560 / 4,304,898,560**. No Dense FFN or alternate architecture is present.

D03/D04: FP32 router/selected softmax/combination, stable Top-8, original weights, dropless same-expert processing, per-head QK-Norm before fixed RoPE, document causal masking and exact attention formulas are retained. BF16 storage and FP32 masters remain explicit. Partition initialization broadcasts the complete logical-name sample stream before slicing; no independent shard RNG initialization is introduced.

D05/D06: native byte BPE and exact text/byte semantics retained; no trained artifact, corpus or segmentation policy invented. D07–D10: explicit optimizer/ownership/storage approval boundaries, single YAML and unknown operational values retained. Chat header grammar, FIM policy and forbidden-output masking remain separate unresolved decisions, outside the plain causal-pretraining source completion scope.

## DATASET EXCLUSION

No datasets, corpus packs, archives, training shards, learned tokenizer artifacts, model weights, real checkpoints, synthetic documents or caches were created or added. `src/shinra/data/` remains Python source. `.gitignore` protects common data/payload/cache/credential paths and extensions without excluding source modules or documentation. Corpus location remains UNKNOWN and must be outside the repository under an approved contract.

## STATICALLY REVIEWED

AST parsed all 59 Python modules without importing SHINRA, Torch or Safetensors. Explicit internal import targets resolve; the source import graph has no detected cycles. Reviewed input/cursor ownership, CP boundary labels, empty-work paths, collective adjoints/order, TP replica counting and checkpoint producer/consumer inventories. D02 was recalculated using integer arithmetic without tensor allocation. Canonical YAML content and dependency files are unchanged.

Git whitespace checks and file inventory/credential-pattern inspection are part of publication admission. Such inspection is not a guarantee that arbitrary secrets are detectable. No generated test cases were used.

API evidence: official [PyTorch 2.9 distributed documentation](https://docs.pytorch.org/docs/2.9/distributed.html) was inspected for group ordering and waiting for async operations before crossing groups; [checkpoint documentation](https://docs.pytorch.org/docs/2.9/checkpoint.html) for non-reentrant contexts and full replay. The current `stable` redirect target was unavailable to the browsing tool. These references do not identify or certify the installed GM runtime; no version was imported and no dependency changed.

## BLOCKED / UNKNOWN

Training remains blocked on approved initialization values, numerical policy approvals, tokenizer registry/artifact, real external corpus, optimizer/scheduler/batch settings, logical group manifest, actual managed runtime compatibility, storage contract, supervision and execution authorization. Long-context learning is unmeasured. Segmented document continuation, general training resharding, distributed generation and object-store publication are not claimed.

Runtime numerical equivalence, collective progress/deadlock behavior, recomputation, actual restore and resident reconstruction remain unverified. Static source acceptance cannot certify those properties on an unspecified runtime.

## NOT EXECUTED

No model initialization, forward/backward, training, inference, tests, synthetic inputs, checkpoint save/load/round-trip, tokenizer training, data/model downloads, dependency installation, benchmarks or GPU planning.

## GIT PUBLICATION

One final commit is authorized with message `feat(shinra): complete distributed training integration and source remediation`. This report is part of that commit and does not predict its own SHA or claim a push before Git confirmation. The final delivery records the actual local/remote SHA, branch and worktree status. Complete changed-file inventory is the commit's Git diff against the baseline above.
