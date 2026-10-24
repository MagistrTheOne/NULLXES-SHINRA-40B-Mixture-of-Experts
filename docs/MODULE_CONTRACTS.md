# Module contracts

24.10.2026. PROPOSED interfaces будущих модулей, не реализованные APIs. Формы и инварианты D01–D07 обязательны для зависимой реализации после approval.

| Module under src/shinra | Input → output; responsibility |
|---|---|
| core/configuration.py | Единственный YAML → typed immutable config; duplicate/unknown key rejection |
| core/validation.py | Config + operation → findings или validated plan; без mutation/defaults |
| core/initialization.py | Explicit random recipe + logical tensors → initialized tensors; без foreign weights |
| model/embeddings.py | IDs → hidden; range checks, untied input weights |
| model/normalization.py | Hidden или per-head Q/K → RMS-normalized; FP32 stats |
| model/rope.py | Normalized Q/K + positions → rotated Q/K; identity/version |
| model/attention.py | Hidden + boundaries/positions/cache protocol → hidden; exact GQA contract |
| model/decoder.py | Hidden → attention residual → MoE residual; no Dense branch |
| model/modeling.py | IDs + explicit state → 48-block output; assembly без I/O |
| model/lm_head.py | Hidden → full-vocabulary logits либо эквивалентные chunks |
| moe/router.py | Hidden + frozen b → selected IDs/weights; no state mutation |
| moe/experts.py | Expert inputs + expert ID → SwiGLU outputs width 800 |
| moe/dispatch.py | Routing plan → stable assignments; chunking dropless |
| moe/combine.py | Assignment outputs → original token order, FP32 sum |
| moe/balancing.py | Unique counts + commit token → bounded b update |
| moe/layer.py | Hidden + explicit routing context → combined MoE output |
| tokenizer/byte_bpe.py | Bytes/text → deterministic IDs; no pretrained assets |
| tokenizer/vocabulary.py | Vocabulary mapping → validated byte/special/learned IDs |
| tokenizer/merges.py | Ordered pair rules → validated rank index; native trainer algorithm if authorized |
| tokenizer/special_tokens.py | Typed structural messages → trusted special IDs |
| tokenizer/serialization.py | Tokenizer records ↔ versioned manifest/bytes; identity checks |
| tokenizer/streaming.py | Complete content fragments → exact encoding; byte output → incremental strict UTF-8 |
| data/registry.py | Source/license approval → immutable source registry |
| data/provenance.py | Raw/transformed identities → provenance records |
| data/ingestion.py | Approved manifest → documents or explicit rejections |
| data/deduplication.py | Document/index → retained/duplicate with provenance |
| data/language.py | Supplied language evidence → metadata; no invented classifier |
| data/packing.py | Segments → PackedBatch with positions/masks/token_uids |
| data/streaming.py | Manifest/cursor → ordered records + pending cursor |
| training/objective.py | Full vocabulary logits/targets/mask → loss sum and count |
| training/optimizer.py | Explicit algorithm/hyperparameters/params → optimizer integration |
| training/scheduler.py | Explicit schedule + committed progress → LR/state |
| training/precision.py | Validated dtype policy → scoped precision, preserving router FP32 |
| training/engine.py | Approved plan/model/data/state/adapters → transactional training orchestration |
| training/state.py | Attempt events → committed/pending state transitions |
| training/telemetry.py | Actual observations → versioned events, no fabricated metrics |
| distributed/ownership.py | Explicit mapping → logical expert/parameter ownership |
| distributed/parallelism.py | Logical partition/group specs → validated execution plan |
| distributed/expert_parallel.py | Assignments + groups → real exchange and backward reverse exchange |
| distributed/context_parallel.py | Sequence shards + boundaries → exact distributed attention contract |
| distributed/communication.py | Explicit collective specs → actual backend operations/failures |
| checkpoint/manifest.py | State inventory → versioned manifest/identity |
| checkpoint/serialization.py | Supported tensors/metadata ↔ selected payload format |
| checkpoint/transactions.py | Shards/acks → manifest/COMMITTED publication last |
| checkpoint/recovery.py | Committed manifest + identities → validated complete restored state |
| inference/generation.py | Request + loaded identity → autoregressive token events/stopping reason |
| inference/sampling.py | Logits + explicit policy/RNG → next token |
| inference/kv_cache.py | Layer K/V appends → transactional cache state |
| inference/decoding.py | Token events → bytes/structured text with strict UTF-8 |

Package __init__.py files только определяют package surface/version без запуска операций. Public interfaces принимают validated configs; mutation разрешена только объявленным владельцам state. Ошибки typed: configuration, identity mismatch, unsupported backend, nonfinite computation, data approval, transaction failure. Нельзя ловить исключение и возвращать фиктивный результат.

UNKNOWN: конкретные class/function names и библиотечные adapters до реализации. BLOCKED: backend-dependent APIs без выбранного и проверенного стека. Ни этот список, ни static review не подтверждают runtime correctness.
