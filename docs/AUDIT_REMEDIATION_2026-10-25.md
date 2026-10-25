# Order 005 — source re-audit and checkpoint remediation

25.10.2026. This is the current audit; the earlier ENGINEERING_ORDER_005_AUDIT report is a historical snapshot.

## AUDIT SUMMARY

Base revision: `3e5828200dac53b5ff22e31d9175c45e73c6ff8c`, branch `main`; worktree clean at entry. Changes below are local, not committed or pushed. No remote refresh was performed; this report makes no new assertion about live GitHub state.

Actual source: 25 Python files in core/model/moe/checkpoint plus package initializers. All existing modules were read and statically reviewed. No tokenizer, data, training, distributed or inference packages exist. Missing responsibilities remain missing, not replaced with stubs.

Architecture preserved: 48 MoE blocks, 128 experts, Top-8, width 800, no shared experts, hidden 2560, GQA 32/8 heads of width 128, untied vocabulary matrices of size 131072. Context target remains 262144. No Dense FFN/fallback or alternative model configuration found.

Three HIGH checkpoint defects were corrected in source. One additional dependency dtype mismatch and stale documentation were corrected. FIXED below means source remediation plus static review only. Publication and training resume remain NO-GO pending authorized runtime verification, storage admission and actual training-state integration.

## FINDINGS

Line references in this table refer to the base revision; current line numbers may move.

| ID | Severity | File/line | Observed defect / violated contract | Correction / status |
|---|---|---|---|---|
| P1-01 | HIGH | checkpoint/serialization.py:203–216; transactions.py:275 | Index total_size summed physical shard bytes including headers, verifier expected tensor payload bytes | FIXED: sum TensorRecord.nbytes; FileRecord.size still stores physical size |
| P1-02 | HIGH | checkpoint/transactions.py:117; serialization.py capture/restore optimizer | Writer emitted device_policy but transaction validator rejected it | FIXED: exact three-field tensor schema, cpu/parameter only; unknown fields still rejected |
| P1-03 | HIGH | checkpoint/transactions.py:130,230 | Write path supplied tensor mappings; manifest verification supplied noniterable TensorGroup objects | FIXED: validator takes Mapping[str,frozenset[str]]; write converts tensor keys, read converts record names |
| P2-01 | MEDIUM | docs/D08_PROJECT_STRUCTURE.md:113–117; D07/D10 | Current-state prose claimed existing checkpoint package absent | FIXED: current source inventory synchronized; historical report explicitly dated |
| P2-02 | MEDIUM | checkpoint/serialization.py:18–20; manifest.py:18–24 | Schema admitted F8_E4M3FNUZ/F8_E5M2FNUZ although pinned v0.6.2 Torch adapter lacks these dtypes | FIXED: reject unsupported variants without dtype conversion; BF16 baseline unchanged |
| G01 | HIGH | missing training/distributed/inference/data/tokenizer packages | No complete resume orchestration, loss/optimizer/scheduler, RNG producer, cursor, collectives, tokenizer or generation | BLOCKED readiness; no complete-system claim. Independent future source work remains authorized |
| G02 | HIGH | checkpoint/recovery.py; transactions.py | Metadata presence and identities are checked, but scheduler/cursor/RNG/ownership semantic consistency needs absent producers/coordinator | BLOCKED end-to-end training resume; generic loader is not a training-state restore engine |
| G03 | MEDIUM | core/configuration.py:architecture_identity; checkpoint/recovery.py | Strict config identity also includes operational fields; architecture identity conservatively includes some operational settings | PROPOSED: future approved whitelist/migration policy. Do not weaken exact identity checks silently |

## Mathematical and source audit

CALCULATED, without importing SHINRA:

P_expert = 3*2560*800 = 6,144,000.

P_nonexpert = 2*131072*2560 + 48*[2*2560*(32+8)*128 + 2560*128 + 2*2560 + (32+8)*128] + 2560 = 1,945,602,560.

P_total = P_nonexpert + 48*128*P_expert = 39,694,338,560.

P_active = P_nonexpert + 48*8*P_expert = 4,304,898,560.

Input lookup accessed count = P_active - 131072*2560 + 2560; arithmetic parameter count excludes input table = P_active - 131072*2560. These are logical parameter definitions, not FLOPs or measurements.

Constructors and parameter_specs agree on 18,867 trainable tensors, names, shapes, residual roles and BF16/FP32 policy. Embedding and head are separately allocated. No extra trainable bias or shared expert is declared. Initialization details and exact candidate recipe are in [INITIALIZATION_RECIPE](INITIALIZATION_RECIPE.md); numeric candidates remain null in YAML.

Attention: head grouping floor(h/4); per-head QK RMSNorm precedes adjacent-pair RoPE; scale 1/sqrt(128); mask combines segment equality and key_position<=query_position. Online softmax/backward algebra reviewed: dS=P*(dO V^T-sum(dO*O)); dQ=dS K/sqrt(H), dK=dS^T Q/sqrt(H), dV=P^T dO; KV gradients accumulate across four query heads. Floating-point agreement and higher-order differentiation are not certified; higher-order gradients are explicitly unsupported. Full-sequence assembly requires contiguous segments starting at zero; no KV cache/decode/CP integration exists.

Routing: FP32 logits and weights, balancing only in selection, stable ties from ascending expert axis, softmax of selected original logits, ascending-ID combine. Dispatch is a permutation with inverse, experts process all assignments in same-expert chunks, no reassign/drop path. Autograd flows through selected logits/weights, gather/linear/SiLU/combine; discrete selection is nondifferentiable. Frozen-bias/version and recomputation event identity are caller obligations; no training coordinator enforces them yet. Empty expert assignments skip expert work, not dummy expert computation.

Balancing: pending counts use Python integer overflow checks before int64 collectives; integer load comparisons, bounded FP32 update, explicit prepare/mutation/commit/abort. Failed mutation requires recovery, not step-number rollback. No collective exists; global count provenance is not validated by the local primitive.

Potential synchronization points: bool(tensor predicate), item(), cpu().tolist(), finite scans, per-head/chunk/expert Python loops. No performance regression or throughput number is inferred without execution. Gather backward reduction ordering is not promised bitwise deterministic across devices.

Configuration: one YAML, closed schema and explicit type checks; duplicate/merge keys, anchors/aliases/tags/directives, multiple documents and nonfinite numeric values rejected by source paths. Immutable mappings and canonical JSON/SHA-256 identities. Fixed architectural values in validation are rejection constraints, not injected defaults. Null remains UNKNOWN and operation gates block dependent work. No parser invocation/import was performed during this audit.

## SAFETENSORS

All model, optimizer/master tensor state (if present), router buffers and RNG tensor groups use official save_file/load_file; metadata is declarative JSON. Aliased storage within a group is rejected; copied contiguous CPU tensors are written per shard. No pickle, torch.save/load or custom binary writer is present. max_shard_bytes groups whole tensors and is not a hard file-size ceiling: one tensor can exceed it and headers add overhead.

The index now describes payload size while file records preserve exact physical size/hash. Global manifest verification converts every group's records into a logical name inventory before optimizer-reference validation. Device placement metadata survives capture/validation/restore. Unsupported optimizer value types still fail explicitly.

Publication is gated by an explicit caller-attested atomic-replace contract. It is not proof of filesystem durability. Callers must freeze committed state while capturing all groups; generic snapshots retain references until shard copies are made. No checkpoint was created, published, loaded or applied. A model restoration failure after mutation requires full recovery; no atomic rollback is claimed. Storage backend, synchronized state producers, GM compatibility and operational resume remain UNKNOWN/BLOCKED.

Primary sources rechecked 25.10.2026: [official Safetensors Torch API](https://huggingface.co/docs/safetensors/api/torch), [pinned v0.6.2 Torch implementation](https://github.com/huggingface/safetensors/blob/v0.6.2/bindings/python/py_src/safetensors/torch.py). The tagged adapter confirms the narrowed FP8 dtype admission. Dependencies were not installed or changed.

## STATICALLY REVIEWED

Read all 25 Python files; compared D01–D10, module contracts, YAML and packaging metadata. AST parsing without imports; AST internal import graph including relative imports: no cycles detected. Searched source for prohibited APIs/placeholders and enumerated YAML/payload files. Recalculated D02 with standalone integer arithmetic. Reviewed the three corrected write/verify/restore data paths. git diff --check used for patch whitespace. These checks do not execute parser, model or checkpoint algorithms.

## NOT EXECUTED

No model initialization, forward/backward, training, inference, checkpoint round-trip, tests, synthetic inputs, downloads, dependency installation, deployment, remote computation or GPU planning. No checkpoint payload exists in the inspected tree. No commit or push.

## FILE INVENTORY

All paths below are relative to `D:/NULLXES/NULLXES SHINRA 40B MoE/`.

### Python — reviewed

- `src/shinra/__init__.py`
- `src/shinra/checkpoint/__init__.py`
- `src/shinra/checkpoint/manifest.py`
- `src/shinra/checkpoint/recovery.py`
- `src/shinra/checkpoint/serialization.py`
- `src/shinra/checkpoint/transactions.py`
- `src/shinra/core/__init__.py`
- `src/shinra/core/configuration.py`
- `src/shinra/core/initialization.py`
- `src/shinra/core/validation.py`
- `src/shinra/model/__init__.py`
- `src/shinra/model/attention.py`
- `src/shinra/model/decoder.py`
- `src/shinra/model/embeddings.py`
- `src/shinra/model/lm_head.py`
- `src/shinra/model/modeling.py`
- `src/shinra/model/normalization.py`
- `src/shinra/model/rope.py`
- `src/shinra/moe/__init__.py`
- `src/shinra/moe/balancing.py`
- `src/shinra/moe/combine.py`
- `src/shinra/moe/dispatch.py`
- `src/shinra/moe/experts.py`
- `src/shinra/moe/layer.py`
- `src/shinra/moe/router.py`

### YAML and dependency manifests

- `configs/shinra_40b_moe.yaml`
- `requirements.txt`
- `pyproject.toml`

### Markdown

- `docs/ARCHITECTURE_CHANGELOG.md`
- `docs/AUDIT_REMEDIATION_2026-10-25.md`
- `docs/D01_ARCHITECTURE.md`
- `docs/D02_PARAMETERS.md`
- `docs/D03_ROUTING.md`
- `docs/D04_ATTENTION_CONTEXT.md`
- `docs/D05_TOKENIZER.md`
- `docs/D06_DATA_PIPELINE.md`
- `docs/D07_PRETRAINING.md`
- `docs/D08_PROJECT_STRUCTURE.md`
- `docs/D09_DEPENDENCIES.md`
- `docs/D10_DECISIONS.md`
- `docs/ENGINEERING_ORDER_005_AUDIT.md`
- `docs/IMPLEMENTATION_STATUS.md`
- `docs/INITIALIZATION_RECIPE.md`
- `docs/MODULE_CONTRACTS.md`
- `docs/README.md`
- `Readme.md`

## FILES CHANGED

- `src/shinra/checkpoint/serialization.py` — P1-01 payload size; reject unsupported FNUZ dtype and remove unused reverse map.
- `src/shinra/checkpoint/transactions.py` — P1-02 exact optimizer schema; P1-03 typed name inventory at both callers.
- `src/shinra/checkpoint/manifest.py` — Match dtype admission to supported serializer.
- `docs/AUDIT_REMEDIATION_2026-10-25.md` — New current report, findings, complete inventory and limitations.
- `docs/INITIALIZATION_RECIPE.md` — New separate explicit candidate, equations, logical tensor inventory and RNG limitations.
- `docs/D01_ARCHITECTURE.md` — Remove obsolete implementation-permission blocker; link candidate.
- `docs/D03_ROUTING.md` — Distinguish existing local primitives from absent training/distributed integration.
- `docs/D07_PRETRAINING.md` — Current checkpoint source and NO-GO boundaries.
- `docs/D08_PROJECT_STRUCTURE.md` — Correct package inventory and actual static import status.
- `docs/D09_DEPENDENCIES.md` — Tagged upstream dtype evidence, no dependency change.
- `docs/D10_DECISIONS.md` — Current checkpoint readiness and initialization proposal links.
- `docs/ENGINEERING_ORDER_005_AUDIT.md` — Mark old snapshot historical and link current report; preserve prior evidence.
- `docs/IMPLEMENTATION_STATUS.md` — Current source fixes and remaining operational blockers.
- `docs/MODULE_CONTRACTS.md` — Exact total_size/device_policy/inventory contracts.
- `docs/ARCHITECTURE_CHANGELOG.md` — Record bounded source remediation without architectural changes.
- `docs/README.md` — Link current audit and initialization candidate.
- `Readme.md` — State checkpoint remediation and continuing NO-GO.

Unchanged: single YAML, requirements, packaging metadata, all model/core/MoE sources and D02 dimensions. No added executable module, alternative profile or test infrastructure. Next implementation work must use these corrected canonical contracts; tokenizer/training decisions listed in D10 remain explicit rather than invented.
