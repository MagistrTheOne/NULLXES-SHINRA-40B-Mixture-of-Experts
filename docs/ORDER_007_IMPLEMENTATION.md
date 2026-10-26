# Engineering Order 007 — source implementation report

26.10.2026. Base commit 3f3c8b4. No new commit/push performed for Order 007.

## IMPLEMENTED SOURCE

Tokenizer: native byte alphabet, stable rank/leftmost BPE, exact in-memory pair-frequency training, strict UTF-8, reversible bytes, explicitly approved special registry, canonical JSON artifacts with hashes/provenance and complete-fragment streaming. No tokenizer artifact trained or emitted.

Data: approved manifests, licensing/provenance, real document byte hashes, explicit language evidence, byte-verified exact deduplication, whole-document packing, document-aware positions/loss masks and reproducible partitioned cursors. No corpus connected.

Training: full vocabulary loss sums, globally normalized target counts, explicit FP32-master AdamW candidate, scheduler, gradient accumulation/clipping, fixed-routing non-reentrant recomputation, global expert counts and commit-only state advancement. Fresh-run DP replicas agree on configuration/inventory then synchronize masters from first logical group member. Failures after optimizer mutation require whole-state recovery.

Distributed: caller-owned process groups, real all-reduce/all-gather/all-to-all with paired backward, expert ownership/transport, tensor-partition linear primitives and gathered-KV context attention. These primitives do not constitute integrated EP/TP/CP model execution.

Inference: identity-checked single-sequence prefill/decode, post-RoPE KV cache, transactional append, explicit sampling, stop/context handling. No serving implementation or inference execution.

Checkpoint: prior three P1 corrections retained. Producer stages actual model, optimizer, FP32 masters, router and RNG tensors through Safetensors; cursor/scheduler/identities remain JSON. Model-only recovery uses canonical parameter inventory. No custom binary format introduced.

## INCOMPLETE / BLOCKED

- Coordinated training checkpoint set publication and all-state restoration are not implemented. TrainingEngine rejects nonzero initial progress or populated optimizer state. Generic tensor restoration is not training resume. No successful recovery claim.
- EP/TP/CP are independent source primitives; decoder assembly remains full resident model. No sharded model coordinator or integrated distributed optimizer recovery.
- Caller must freeze state during snapshots. Per-rank publication cannot establish a complete global checkpoint set. Storage atomicity/durability is UNKNOWN.
- No approved corpus, trained vocabulary or operational training settings. Structural IDs require explicit approval. Chat header grammar is unresolved; typed structural/FIM sequences do not imply complete chat formatting.
- BPE training repeatedly counts pairs in memory; dedup stores retained bytes; cursor reconstruction replays prefix documents. Scalability has not been measured and is not claimed.
- Whole documents beyond context fail explicitly. Packer currently fills architectural context; a smaller configured microbatch budget can reject the pack rather than truncate it. Split/curriculum policy requires a decision.
- CP primitive gathers KV on each member; it is not a ring-attention implementation or a memory-budget claim.
- Remote rank failure requires launcher-level group termination; no speculative recovery collective. Installed GM API/backend compatibility UNKNOWN.
- RNG capture covers explicit Torch CPU/CUDA generators, not arbitrary external library RNG.

## ARCHITECTURE COMPLIANCE

48 routed-MoE blocks, 128 experts, Top-8, zero shared experts, width800 and Q/K/V/O dimensions unchanged. D02 counts remain 39,694,338,560 total / 4,304,898,560 conventional active. FP32 master tensors are optimizer state, not extra model parameters. Single YAML retained. Added optimizer_precision/parameter_group_policy/gradient_clip_norm_type as null; no previous null replaced. Safetensors remains the tensor format.

## STATICALLY REVIEWED

AST parsed 59 Python files without importing SHINRA. Static import-module resolution: 0 missing internal modules; 0 detected cycles. External non-stdlib roots: safetensors, torch, yaml. These checks do not prove runtime correctness. Reviewed new training transaction, checkpoint producer, canonical decoder calls and inference cache identity interfaces.

## FILES CREATED / MODIFIED

Paths below are relative to the repository root, from the current diff against HEAD. New tokenizer/data/training/distributed/inference packages are created; existing config/core/model/checkpoint/docs files are modified. This report is also newly created.

- `Readme.md`
- `configs/shinra_40b_moe.yaml`
- `docs/ARCHITECTURE_CHANGELOG.md`
- `docs/D07_PRETRAINING.md`
- `docs/D08_PROJECT_STRUCTURE.md`
- `docs/D09_DEPENDENCIES.md`
- `docs/D10_DECISIONS.md`
- `docs/IMPLEMENTATION_STATUS.md`
- `docs/MODULE_CONTRACTS.md`
- `src/shinra/checkpoint/recovery.py`
- `src/shinra/checkpoint/transactions.py`
- `src/shinra/core/validation.py`
- `src/shinra/data/__init__.py`
- `src/shinra/data/deduplication.py`
- `src/shinra/data/ingestion.py`
- `src/shinra/data/language.py`
- `src/shinra/data/packing.py`
- `src/shinra/data/provenance.py`
- `src/shinra/data/registry.py`
- `src/shinra/data/streaming.py`
- `src/shinra/distributed/__init__.py`
- `src/shinra/distributed/communication.py`
- `src/shinra/distributed/context_parallel.py`
- `src/shinra/distributed/expert_parallel.py`
- `src/shinra/distributed/ownership.py`
- `src/shinra/distributed/parallelism.py`
- `src/shinra/inference/__init__.py`
- `src/shinra/inference/decoding.py`
- `src/shinra/inference/generation.py`
- `src/shinra/inference/kv_cache.py`
- `src/shinra/inference/sampling.py`
- `src/shinra/model/modeling.py`
- `src/shinra/tokenizer/__init__.py`
- `src/shinra/tokenizer/byte_bpe.py`
- `src/shinra/tokenizer/merges.py`
- `src/shinra/tokenizer/serialization.py`
- `src/shinra/tokenizer/special_tokens.py`
- `src/shinra/tokenizer/streaming.py`
- `src/shinra/tokenizer/vocabulary.py`
- `src/shinra/training/__init__.py`
- `src/shinra/training/engine.py`
- `src/shinra/training/objective.py`
- `src/shinra/training/optimizer.py`
- `src/shinra/training/precision.py`
- `src/shinra/training/scheduler.py`
- `src/shinra/training/state.py`
- `src/shinra/training/telemetry.py`
- `docs/ORDER_007_IMPLEMENTATION.md`

## NOT EXECUTED

No model initialization, forward/backward, training, inference, tokenizer training, checkpoint save/load or round-trip, tests, project algorithm execution, dependency installation, downloads, deployment or GPU planning. Only source edits/read-only review/AST processing were performed. Runtime correctness, numerical gradients, performance and publication durability remain UNVERIFIED.

## CONTINUATION

Implement coordinated checkpoint-set admission/restoration before enabling resumed TrainingEngine construction. Then integrate explicit EP/TP/CP ownership and execution into canonical model assembly without replacing local MoE semantics. Reconcile remaining chat/split/operational decisions through existing null gates. Do not represent Order 007 as fully complete.
