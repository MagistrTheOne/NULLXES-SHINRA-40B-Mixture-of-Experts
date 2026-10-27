# Module contracts and implementation status

25.10.2026. Inventory reflects current source after Engineering Order 005 static audit. A listed contract is not evidence of runtime correctness.

## Existing source

| Module | Responsibility | Status |
|---|---|---|
| `core/configuration.py` | Canonical YAML decoding, immutable config values and identities | Implemented; static review only |
| `core/validation.py` | Closed schema, fixed architecture and operation prerequisites | Implemented; static review only |
| `core/initialization.py` | Explicit recipe, logical parameter inventory and parameter allocation | Implemented; BF16 sampling precision corrected; never executed |
| `model/embeddings.py` | Input token embeddings | Implemented |
| `model/normalization.py` | RMSNorm and independent per-head Q/K RMSNorm | Implemented |
| `model/rope.py` | Fixed adjacent-pair RoPE | Implemented; operation requires explicit approval and theta |
| `model/attention.py` | GQA, document-aware causal mask, blockwise attention and explicit backward | Implemented; backend/chunk settings and runtime correctness unverified |
| `model/decoder.py` | Pre-norm attention and routed-MoE residual block | Implemented |
| `model/modeling.py` | 48-block model assembly and full-sequence forward surface | Implemented; not initialized or executed |
| `model/lm_head.py` | Untied full-vocabulary output projection | Implemented |
| `moe/router.py` | FP32 router, Top-8 selection, selected-logit softmax | Implemented |
| `moe/experts.py` | SwiGLU expert computation with explicit chunking | Implemented |
| `moe/dispatch.py` | Stable local assignment ordering | Implemented; distributed primitive exists separately; local MoE is not EP-integrated |
| `moe/combine.py` | FP32 weighted combine in original token order | Implemented |
| `moe/balancing.py` | Load accounting and prepared/committed bias state | Implemented primitive; requires an explicit initial bias; distributed synchronization/optimizer orchestration external |
| `moe/layer.py` | Local router/dispatch/expert/combine assembly | Implemented; no distributed expert execution |
| `checkpoint/manifest.py` | Versioned strict JSON checkpoint metadata and tensor inventory records | Implemented; static source only |
| `checkpoint/serialization.py` | Safetensors shard I/O, expected inventory checks and stable-name optimizer snapshots | Implemented; runtime dependency not installed |
| `checkpoint/transactions.py` | Staged complete-state writes, hashes and explicit atomic-publication capability | Implemented; storage durability not claimed |
| `checkpoint/recovery.py` | Full group load, exact identity/inventory validation and strict module restoration | Implemented; no runtime restore performed |
| package `__init__.py` files | Package markers | Present; no eager model construction |

## Order 007 source contracts — 26.10.2026

| Modules | Contract and boundary |
|---|---|
| tokenizer/vocabulary, merges, special_tokens | Immutable admitted byte payloads, creation-ranked merge DAG, explicitly approved structural IDs |
| tokenizer/byte_bpe, streaming | Exact UTF-8/bytes, no normalization, rank/leftmost merge semantics, complete-fragment buffering independent of I/O chunk size; exact in-memory training |
| tokenizer/serialization | Canonical JSON, SHA-256 identity, strict provenance/merge/statistics admission; no artifact generated |
| data/registry, provenance, ingestion, language | Approved manifest only, strict source/license/language evidence, relative paths and whole-document byte hash admission |
| data/deduplication, packing, streaming | Byte-equal dedup after hash lookup, BOS/EOS and document boundaries, shifted loss mask, stable ordinal token IDs, transactional cursor and deterministic replay |
| training/objective, optimizer, scheduler, precision | Summed full-vocabulary loss; FP32-master explicitly selected AdamW, global target normalization, explicit schedule and RNG device list |
| training/state, engine, telemetry | Fresh full-replica DP transactions, fixed routing snapshots during recomputation, global load statistics, commit-only cursor/scheduler/step; failures after mutation require recovery |
| distributed/communication, ownership, expert_parallel | Caller-owned groups, metadata consensus, explicit owner map, real variable split all-to-all and inverse backward |
| distributed/parallelism, context_parallel | TP linear primitives with explicit autograd, full-replica gradient sum, gathered-KV CP attention; not integrated model sharding |
| inference/kv_cache, decoding, sampling, generation | Single-sequence cache transaction, post-RoPE keys, absolute positions, explicit sampling/stop controls and identities |
| checkpoint/transactions, recovery | Existing generic serializer retained; producer stages model/master/optimizer/router/RNG plus JSON; model load checks canonical inventory. Coordinated training restoration remains absent |

Snapshot caller must freeze mutation for the full write. Per-rank staged checkpoint is not a committed multi-rank checkpoint set. Initialization and all execution remain prohibited.

## Required invariants

Only the single `configs/shinra_40b_moe.yaml` is canonical. Unknown fields and missing schema fields must fail validation; null is UNKNOWN and cannot be converted into an execution default. Source modules must preserve the fixed 48-layer MoE architecture and must not introduce Dense fallback, stubs, mocks, or synthetic execution paths.

All checkpoint tensor payloads must use the official Safetensors format. Metadata may use declarative UTF-8 JSON; no pickle or custom tensor/binary checkpoint format is permitted. Checkpoint source exists, but coordinated training restore, approved storage backend, durability, GM compatibility and runtime recovery remain BLOCKED/unverified.

## Corrected checkpoint contracts — 25.10.2026

Index metadata.total_size = sum(TensorRecord.nbytes), excluding headers. FileRecord.size = physical file bytes. Transaction training validator receives Mapping[str, frozenset[str]] from raw tensor keys on write and TensorGroup.records on read. Tensor optimizer entries contain exactly kind/tensor_name/device_policy; placement is cpu or parameter. Unknown fields and unsupported dtype variants fail closed. No checkpoint execution verified; see [current audit](AUDIT_REMEDIATION_2026-10-25.md).

## Order 008 continuation — 26.10.2026

ExpertExecution now connects ownership/transport to canonical MoELayer and TrainingEngine. ModuleDict preserves logical expert names; only owned experts are materialized. DP/EP synchronization excludes owner expert gradients from repeated SUM. Global clipping counts replicas once. ContextExecution connects global positions/segments to model attention; CP training remains blocked. publish_training_set and restore_training_set provide explicit complete-set DP/EP checkpoint admission/restoration. Failure leaves RECOVERY_REQUIRED, not a rollback claim. See ORDER_008_PROGRESS.md for exact limitations.

## Order 009 continuation — 27.10.2026

Canonical TP model source and full-logical-stream slice initialization are now implemented, with explicitly replicated router/norms. CP real-target boundary partitioning and summed objective added. TP training/checkpoint integration, full CP training, uneven EP scheduling and mixed-group orchestration remain incomplete and fail closed where exposed. Source Complete is NOT achieved. No model/test/checkpoint execution. Details and changed files: [Order 009 report](ORDER_009_DISTRIBUTED_COMPLETION.md).
