# Engineering Order 005 — Static Source Audit

25.10.2026. Independent static source review and narrowly scoped remediation. This report does not certify runtime behavior.

## AUDIT SUMMARY

- **Repository revision:** branch `main`, commit `f8566125325fb1f6597da1eace853f65abe5164a`, `main` ahead of `origin/main` by one commit at inspection. Both staged and unstaged local changes were present; no commit was created.
- **Actual source state:** 20 Python files exist under `src/shinra/core`, `src/shinra/model` and `src/shinra/moe`. Tokenizer, data, training, distributed, checkpoint and inference source packages are absent.
- **Architecture compliance:** static inventory and D02 algebra agree with 48 MoE blocks, 128 experts/block, Top-8, expert width 800, shared experts 0, hidden 2560, GQA 32/8 with head width 128, vocabulary 131072, untied embeddings, total 39,694,338,560 and conventional active 4,304,898,560 parameters. No Dense FFN/fallback was found.
- **Critical findings:** none established by this static review. Absence of a finding is not proof of runtime correctness.
- **High-priority findings:** initialization precision and operation gating defects were fixed. Safetensors save/load and complete-state recovery remain blocked because those modules do not exist and storage/runtime compatibility is unknown.
- **Remaining unknowns:** approved initialization values, RoPE approval, attention backend/chunk sizes, balancing coefficients, tokenizer IDs/artifact, corpus, optimizer/training plan, distributed topology, Safetensors dependency compatibility, and checkpoint storage durability.

## REPOSITORY INVENTORY

### Git and worktree

The local repository reported `main`, HEAD `f8566125325fb1f6597da1eace853f65abe5164a`, and `main...origin/main [ahead 1]`. Existing staged and unstaged changes were preserved; this audit did not commit or reset them.

### Python source — 20 files

- `src/shinra/__init__.py`
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

### Configuration, manifests, and documentation

- The only YAML is `configs/shinra_40b_moe.yaml`.
- Dependency/build manifests are `requirements.txt` and `pyproject.toml`.
- Markdown: root `Readme.md`; `docs/ARCHITECTURE_CHANGELOG.md`, `docs/D01_ARCHITECTURE.md` through `docs/D10_DECISIONS.md`, `docs/IMPLEMENTATION_STATUS.md`, `docs/MODULE_CONTRACTS.md`, `docs/README.md`, and this audit.
- The canonical YAML already contains `checkpoint.format: safetensors`; it was not modified.
- No checkpoint payload file or test/fixture/mock/smoke directory was found.

### Missing source relative to D08

`tokenizer/`, `data/`, `training/`, `distributed/`, `checkpoint/` and `inference/` are documented target packages, not present implementation. Their work is BLOCKED where it depends on unknown approvals, selected runtime/dependencies or storage durability.

## STATIC FINDINGS

| ID | Severity | File | Observed implementation and violated contract | Required correction | Status |
|---|---|---|---|---|---|
| O5-01 | HIGH | `src/shinra/core/initialization.py` | Random values for BF16 parameters were sampled directly in BF16, conflating sampling and storage dtypes; Engineering Order 005 §6.3 requires an explicit FP32-sampling path. | Sample non-norm tensors in FP32 and explicitly cast to the declared storage dtype. Keep gamma fill at one and the logical-name seed policy. | FIXED |
| O5-02 | HIGH | `src/shinra/core/validation.py` | `attention.sliding_window: null` was accepted by the schema and was not an operation blocker, allowing UNKNOWN policy past model gating. | Permit only explicit `false` as a supported value; retain null as UNKNOWN and block model-dependent operations on it. | FIXED |
| O5-03 | MEDIUM | `src/shinra/moe/balancing.py` | Constructor silently created an all-zero balancing buffer, making a proposed initial state an implicit execution default. | Require an explicit, non-trainable, finite FP32 initial bias on the requested device and within the approved bound; clone it into the owned buffer. | FIXED |
| O5-04 | HIGH | `src/shinra/checkpoint/` (absent) | No Safetensors serializer, tensor inventory validator, manifest, complete-state publication, or recovery implementation exists. The format policy alone does not provide checkpoint functionality. | Keep save/load/resume BLOCKED until a compatible Safetensors version, metadata schema, storage durability and full tensor-state contract are selected and implemented. No alternate binary format or pickle fallback. | BLOCKED |
| O5-05 | MEDIUM | `docs/IMPLEMENTATION_STATUS.md`, `docs/D08_PROJECT_STRUCTURE.md`, `docs/MODULE_CONTRACTS.md`, `docs/D09_DEPENDENCIES.md` | Documents said Python source did not exist or that actual imports were not inspectable, contradicting the current source tree. | Reconcile implementation status and contracts with the actual source inventory; preserve historical changelog entries as historical. | FIXED |
| O5-06 | HIGH | Missing `tokenizer/`, `data/`, `training/`, `distributed/`, `inference/` | D08/D01/D03–D07 describe these responsibilities, but no corresponding implementation is present. This is a readiness gap, not a reason to add stubs or claim completion. | Do not advertise those capabilities; implement only after their required approvals and integration contracts are available. | BLOCKED |

No Dense FFN, Dense fallback, pretrained loading path, `torch.save`/`torch.load`, pickle checkpoint implementation, `NotImplementedError`, `pass` placeholder, or fake collective was found in the inspected source. A missing implementation remains a blocker, not a passing feature.

## ARCHITECTURE AND MODEL SOURCE

### Parameter algebra

Static D02 terms:

- Two untied vocabulary matrices: `2 × 131072 × 2560 = 671,088,640`.
- Attention per layer: `2 × 2560 × (32 + 8) × 128 = 26,214,400`; across 48 layers: `1,258,291,200`.
- Router per layer: `128 × 2560 = 327,680`; across 48: `15,728,640`.
- Expert per expert: `3 × 2560 × 800 = 6,144,000`; all experts: `48 × 128 × 6,144,000 = 37,748,736,000`.
- Norms: 48 layers of `2 × 2560 + (32 + 8) × 128`, plus final `2560`, total `494,080`.
- Sum: `39,694,338,560`. Selected experts contribute `48 × 8 × 6,144,000 = 2,359,296,000`; conventional active total is `4,304,898,560`.

`parameter_specs()` and source constructors describe the embedding/head, attention projections, layer norms, separate Q/K per-head scales, FP32 router matrices and all gate/up/down expert matrices. No attention, expert or router trainable bias is declared. The final head is separately initialized and not tied to embeddings.

### Attention

Static source implements GQA grouping, Q/K per-head normalization before RoPE, fixed adjacent-pair RoPE, causal and segment-aware masking, and a blockwise exact attention path with an explicit first-order backward. It has no sliding-window path. Full 262144-position operation, device behavior, numerical agreement, cache semantics and performance are not runtime-verified. The model assembly is a full-sequence path; decode continuation/KV cache and context-parallel execution are absent.

Several finite/range checks convert tensor predicates to Python booleans, and attention loops over heads and chunks in Python. These are potential device synchronization/performance costs by static inspection only; no runtime penalty is asserted.

### MoE

Static source implements FP32 router projection and selected-logit softmax, stable Top-8 expert ID ordering, local stable dispatch, expert SwiGLU, FP32 combine, and chunked per-expert work without dropping or reassigning assignments. Router forward does not mutate balancing state. Balancing has attempt phases and recomputation event accounting, but distributed count synchronization and the successful optimizer transaction are external obligations; no training engine verifies them.

## INITIALIZATION

### Parameter inventory

The logical inventory covers input embedding `[131072,2560]`, independent output head `[131072,2560]`, per-layer Q/K/V/O projections, two hidden RMSNorms, Q/K per-head scales, router `[128,2560]`, and 128 expert gate/up/down matrices per layer. All 48 blocks are constructed with routed MoE; no shared/Dense FFN parameters are specified.

### Seed and recipe status

Initialization is explicit and not performed during import. Logical parameter names feed SHA-256 seed derivation to individual generators; expert names differ, and input/output names are independent. This gives a deterministic seed mapping for a fixed implementation/runtime contract, not proof of equal random tensors across devices or library versions. Residual scaling applies to parameters marked `residual_projection`.

Engineering Order 005's normal distribution with `std=1/sqrt(2560)` and residual multiplier `1/sqrt(2*48)` is recorded as **PROPOSED**, not applied as a default. Existing YAML initialization fields remain null; initialization remains blocked. Actual parameter allocation and numerical behavior were not executed.

## SAFETENSORS

- **Policy:** all checkpoint tensor payloads must use standard Safetensors; metadata is declarative UTF-8 JSON. Custom tensor containers, pickle and alternative payload formats are prohibited.
- **YAML/parser:** canonical YAML already says `safetensors`; closed Python schema requires the same value. There is no silent format conversion.
- **Tensor categories required by contract:** model tensors, optimizer tensors/master weights where applicable, router balancing buffers, RNG tensor states, and other actual tensor states required for resume.
- **Naming/sharding/load/recovery:** not implemented. No `checkpoint` Python package or checkpoint artifact exists.
- **Dependency:** Safetensors is not in `requirements.txt`. No version was selected or installed. Managed PyTorch compatibility and storage durability are UNKNOWN; therefore adapter/dependency admission and resume remain BLOCKED. Official API documentation describes `save_file`/`load_file`, including contiguous dense save inputs and lack of protection from concurrent tensor mutation; API documentation does not establish a complete SHINRA checkpoint transaction.

## CONFIGURATION AUDIT

`configuration.py` rejects duplicate/merge keys, YAML anchors/aliases, directives/tags and multiple documents, uses a constrained scalar decoder, and freezes configuration mappings. `validation.py` applies a closed nested schema, exact fixed-model constraints, parameter accounting, and operation-specific blockers. Parser and schema preserve null as UNKNOWN rather than synthesizing execution values. Architecture constants in the validator intentionally duplicate the fixed contract as rejection constraints; they are not injected defaults.

The balancing state now requires its initial bias explicitly at construction; the canonical YAML remains untouched as requested. `checkpoint.format` already matched the Safetensors requirement. Balancing coefficients, initialization, backend/chunking and operational fields remain null and block dependent operations.

## STATICALLY REVIEWED

- Read all 20 Python source files and checked D01–D10, module contracts, dependency register, implementation status and both READMEs.
- AST-parsed all Python files without importing project modules or creating a model.
- AST import graph: no cycles found among the existing modules; import roots are `yaml`, `torch`, `shinra` and Python standard library.
- Searched source for prohibited serialization APIs/formats, placeholders, pretrained loading, Dense fallback, token drop/reassignment, and sliding-window implementation.
- Enumerated repository YAML/manifests/Markdown and searched for checkpoint payload extensions and prohibited test directories.
- Checked architecture arithmetic against D02 and compared it with declared logical parameter shapes.
- Did not inspect or fetch model weights, corpora, or checkpoint artifacts.

## BLOCKED / NOT EXECUTED

BLOCKED: Safetensors serializer and resume; tokenizer and artifact; data ingestion/corpus; objective and training engine; optimizer/scheduler; real distributed communication; inference/sampling/KV cache; effective 256K runtime behavior; backend/GM compatibility and durable publication.

NOT EXECUTED: model or tokenizer initialization, forward/backward, training, inference, tests (none created or run), dependency installation, downloads, deployment, remote execution, or GPU planning. No benchmark or model-quality claim is made.

## FILES CHANGED BY THIS AUDIT

- `src/shinra/core/initialization.py` — FP32 sampling then explicit storage cast.
- `src/shinra/core/validation.py` — UNKNOWN sliding-window operation gate; only false supported.
- `src/shinra/moe/balancing.py` — require explicit validated initial bias; no implicit zero buffer.
- `docs/IMPLEMENTATION_STATUS.md`, `docs/MODULE_CONTRACTS.md`, `docs/D01_ARCHITECTURE.md`, `docs/D07_PRETRAINING.md`, `docs/D08_PROJECT_STRUCTURE.md`, `docs/D09_DEPENDENCIES.md`, `docs/D10_DECISIONS.md`, `docs/ARCHITECTURE_CHANGELOG.md` — factual implementation status, findings and Safetensors policy.
- `docs/ENGINEERING_ORDER_005_AUDIT.md` — this report.

`configs/shinra_40b_moe.yaml`, `requirements.txt` and `pyproject.toml` were not modified.

## Engineering Order 006 — Stage A update

This appendix records subsequent checkpoint source work; the audit above remains an as-of-Order-005 snapshot.

### Implemented source

- `src/shinra/checkpoint/manifest.py`: strict versioned JSON schema, duplicate-key rejection, canonical identities, safe filenames, shard and tensor records, exact inventory and Safetensors-only policy.
- `src/shinra/checkpoint/serialization.py`: exact shape/dtype inventories, persistent module state capture, Safetensors save/load, deterministic shards and standard index JSON, copied contiguous CPU tensors, alias rejection, and stable logical-name optimizer state serialization/restoration.
- `src/shinra/checkpoint/transactions.py`: model/training-resume completeness rules, staged transactions, index/shard hashes, manifest identity, fail-closed partial detection and commit marker publication only with explicit atomic-replace capability.
- `src/shinra/checkpoint/recovery.py`: verification before loading every group and strict full module state restoration.
- `src/shinra/checkpoint/__init__.py`: explicit package API; importing it performs no I/O or model construction.

### Dependency

`requirements.txt` pins `safetensors==0.6.2`. PyPI release metadata declares Python >=3.9 and the `torch` extra requires torch>=1.10; the tagged upstream license is Apache-2.0. The official Torch API provides `save_file`/`load_file` and requires dense contiguous save tensors. SHINRA's project floor is Python >=3.10. The managed GM Torch version is unknown, so real runtime compatibility remains unverified. No package was installed.

### Remaining blockers

No checkpoint was created or loaded. Storage durability and object-store publication are not claimed. The filesystem writer requires a caller-attested same-directory atomic-replace guarantee and only fsyncs written files; it does not claim durable directory metadata. Training-state capture still depends on absent training, scheduler, cursor, RNG and distributed producers; training resume is therefore not operationally authorized. The chosen optimizer remains unknown; unsupported scalar/group representations fail closed.

### Files changed in Stage A

- Created: `src/shinra/checkpoint/__init__.py`, `manifest.py`, `serialization.py`, `transactions.py`, `recovery.py`.
- Modified: `requirements.txt`; `docs/D07_PRETRAINING.md`, `docs/D09_DEPENDENCIES.md`, `docs/IMPLEMENTATION_STATUS.md`, `docs/MODULE_CONTRACTS.md`, `docs/ARCHITECTURE_CHANGELOG.md`, `docs/README.md`, `Readme.md`.
- The canonical YAML was not modified.

Static syntax/import-graph inspection is the only permitted verification here; no module import, package installation, checkpoint I/O, test, model execution, download, deployment or GPU planning was performed.
