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
| `moe/dispatch.py` | Stable local assignment ordering | Implemented; distributed transport absent |
| `moe/combine.py` | FP32 weighted combine in original token order | Implemented |
| `moe/balancing.py` | Load accounting and prepared/committed bias state | Implemented primitive; requires an explicit initial bias; distributed synchronization/optimizer orchestration external |
| `moe/layer.py` | Local router/dispatch/expert/combine assembly | Implemented; no distributed expert execution |
| `checkpoint/manifest.py` | Versioned strict JSON checkpoint metadata and tensor inventory records | Implemented; static source only |
| `checkpoint/serialization.py` | Safetensors shard I/O, expected inventory checks and stable-name optimizer snapshots | Implemented; runtime dependency not installed |
| `checkpoint/transactions.py` | Staged complete-state writes, hashes and explicit atomic-publication capability | Implemented; storage durability not claimed |
| `checkpoint/recovery.py` | Full group load, exact identity/inventory validation and strict module restoration | Implemented; no runtime restore performed |
| package `__init__.py` files | Package markers | Present; no eager model construction |

## Contracted but absent source

| Package | Required modules | Status |
|---|---|---|
| `tokenizer/` | byte BPE, vocabulary, merges, special tokens, serialization, streaming | Not implemented |
| `data/` | registry, provenance, ingestion, deduplication, language, packing, streaming | Not implemented |
| `training/` | objective, optimizer, scheduler, precision, engine, state, telemetry | Not implemented |
| `distributed/` | ownership, parallelism, expert/context parallelism, communication | Not implemented |
| `inference/` | generation, sampling, KV cache, decoding | Not implemented |

## Required invariants

Only the single `configs/shinra_40b_moe.yaml` is canonical. Unknown fields and missing schema fields must fail validation; null is UNKNOWN and cannot be converted into an execution default. Source modules must preserve the fixed 48-layer MoE architecture and must not introduce Dense fallback, stubs, mocks, or synthetic execution paths.

All checkpoint tensor payloads must use the official Safetensors format. Metadata may use declarative UTF-8 JSON; no pickle or custom tensor/binary checkpoint format is permitted. Checkpoint source exists, but training-state producers, approved storage backend, durability, GM compatibility and runtime recovery remain BLOCKED/unverified.
