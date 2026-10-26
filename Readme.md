 <div align="center">

<h3>NULLXES / INTELLIGENCE DIVISION</h3>

# SHINRA

### 40B MIXTURE OF EXPERTS

**A native large language model architecture. Built from first principles.**

**39.69B Total Parameters · 4.30B Active Parameters · 256K Context Target**

[Architecture](#architecture) · [Mixture of Experts](#mixture-of-experts) · [Technical Specifications](#technical-specifications) · [Documentation](#documentation) · [Development Status](#development-status)

---

**ONE INTELLIGENCE. DIFFERENT BODIES.**

</div>

## Overview

**SHINRA 40B MoE** is a large-scale autoregressive language model project developed by **NULLXES**, an artificial intelligence company founded in 2026 and headquartered in Krasnodar.

SHINRA is designed as an independent, decoder-only Transformer built around a native **Mixture-of-Experts (MoE)** architecture.

The project targets approximately **40 billion total parameters**, activating a selected subset of experts for each token while retaining the representational capacity of the full model.

SHINRA is being engineered from random initialization, with its own model architecture, expert routing system, tokenizer specification, training infrastructure and checkpoint contracts.

**This repository documents the architecture and ongoing implementation of SHINRA. It does not currently provide a trained model or published benchmark results.**

## At a Glance

| Specification | SHINRA 40B MoE |
|:---|:---|
| Model family | SHINRA |
| Organization | NULLXES |
| Architecture | Decoder-only Transformer |
| Feed-forward architecture | Mixture of Experts |
| Total parameters | **39,694,338,560** |
| Active parameters per token¹ | **4,304,898,560** |
| Transformer layers | 48 |
| Routed experts per layer | 128 |
| Experts activated per token | 8 |
| Shared experts | 0 |
| Vocabulary | 131,072 tokens |
| Context length target | 262,144 tokens |
| Primary languages | English / Russian |
| Training approach | From scratch |
| Model weights | Not released |
| Project stage | Architecture / Engineering |

¹ Conventional active-parameter accounting includes shared non-expert weights and the complete input/output embedding matrices. Figures are analytical architectural calculations, not runtime measurements.

---

## Architecture

SHINRA follows a decoder-only, pre-normalized Transformer architecture.

All 48 Transformer blocks contain a routed Mixture-of-Experts feed-forward layer. No alternative dense-FFN architecture is provided.

### Model Execution Flow

```text
                        INPUT TOKENS
                             │
                             ▼
                      TOKEN EMBEDDING
                             │
                             ▼
              ┌──────────────────────────┐
              │     TRANSFORMER BLOCK    │
              │                          │
              │      RMSNorm             │
              │         │                │
              │         ▼                │
              │    GQA + QK-Norm         │
              │         │                │
              │        RoPE              │
              │         │                │
              │     ATTENTION            │
              │         │                │
              │     RESIDUAL ADD         │
              │         │                │
              │       RMSNorm            │
              │         │                │
              │         ▼                │
              │      MoE ROUTER          │
              │         │                │
              │     TOP-8 EXPERTS        │
              │         │                │
              │    DISPATCH / COMPUTE    │
              │         │                │
              │        COMBINE           │
              │         │                │
              │     RESIDUAL ADD         │
              └───────────┬──────────────┘
                          │
                      × 48 LAYERS
                          │
                          ▼
                       RMSNorm
                          │
                          ▼
                    OUTPUT PROJECTION
                          │
                          ▼
                   NEXT-TOKEN LOGITS
```

### Architectural Principles

**Native MoE**

Expert routing is a fundamental component of every Transformer block, not an optional optimization.

**Sparse Expert Activation**

Every token is routed to eight distinct experts per layer, selected from a pool of 128.

**Long-Context Design**

The architecture targets a native positional range of 262,144 tokens. Effective long-context capabilities remain subject to training and evaluation.

**Independent Initialization**

Model parameters are initialized from scratch. Existing pretrained language-model weights are not used.

**Explicit Engineering Contracts**

Parameter shapes, router behavior, numerical precision, attention semantics and checkpoint state are specified in versioned technical documents.

---

## Technical Specifications

### Transformer

| Component | Configuration |
|:---|:---|
| Hidden dimension | 2,560 |
| Transformer blocks | 48 |
| Attention architecture | Grouped Query Attention |
| Query attention heads | 32 |
| Key/value attention heads | 8 |
| Attention head dimension | 128 |
| Query projection | 2,560 → 4,096 |
| Key projection | 2,560 → 1,024 |
| Value projection | 2,560 → 1,024 |
| Output projection | 4,096 → 2,560 |
| Attention normalization | Per-head QK-Norm |
| Block normalization | RMSNorm |
| Position representation | Rotary Position Embeddings |
| LM head | Untied |
| Attention / expert biases | Disabled |

### Parameter Distribution

| Component | Parameters |
|:---|---:|
| Input embeddings | 335,544,320 |
| Output projection head | 335,544,320 |
| Attention projections | 1,258,291,200 |
| Routed expert networks | 37,748,736,000 |
| Expert routers | 15,728,640 |
| RMSNorm and QK-Norm | 494,080 |
| **Total** | **39,694,338,560** |

The parameter counts above follow the documented architecture assumptions and do not represent a loaded or executed model.

For the full derivation, see [D02 — Parameter Accounting](docs/D02_parameter_accounting.md).

---

## Mixture of Experts

SHINRA uses **fine-grained, token-level expert routing**.

Each Transformer block maintains 128 independently parameterized SwiGLU experts. For every token, eight experts are selected and their outputs are combined using normalized routing weights.

### Routing Architecture

```text
                     TOKEN HIDDEN STATE
                             │
                             ▼
                      FP32 ROUTER
                             │
                             ▼
                     128 EXPERT SCORES
                             │
                   BALANCING BIAS
                    (selection only)
                             │
                             ▼
                       TOP-8 SELECT
                             │
              ┌──────────────┴──────────────┐
              │                             │
       EXPERT ASSIGNMENT              ROUTING WEIGHTS
              │                       (original logits)
              ▼                             │
      DROPLESS DISPATCH                     │
              │                             │
              ▼                             │
       SELECTED EXPERTS                     │
              │                             │
              └──────────────┬──────────────┘
                             ▼
                     WEIGHTED COMBINE
                             │
                             ▼
                        MoE OUTPUT
```

### Routing Properties

- **Top-8 expert selection** from 128 routed experts.
- **FP32 router logits** for the specified numerical routing path.
- **Deterministic logical tie-breaking** by expert ID.
- **Separate non-trainable balancing bias** affecting selection only.
- **Selected-expert softmax** using original router logits.
- **Dropless routing**, with no silent token removal.
- **Stable dispatch/combine identities** across expert execution.
- **Explicit routing-state ownership** during optimizer transactions.
- **No dense-FFN fallback** under unsupported execution conditions.

The router is designed to preserve expert assignments even when workload distribution is uneven.

These properties define the intended implementation contract; they are not claims of completed runtime validation.

Read the full specification: [D03 — MoE Routing](docs/D03_moe_routing.md).

---

## Long-Context Architecture

SHINRA targets a maximum native context of **262,144 tokens**.

The proposed attention design combines:

- Grouped Query Attention.
- Independent per-head QK-Norm.
- Rotary Position Embeddings.
- Causal document-aware attention boundaries.
- Position-consistent prefill and autoregressive decoding.
- Explicit KV-cache identity and lifecycle contracts.
- Support for long-context training curricula.

The documented RoPE design proposes fixed-frequency rotary embeddings with a base frequency of `1,000,000`, without automatic dynamic context scaling.

**A 256K architectural context target must not be interpreted as a demonstrated 256K effective context.**

Long-context performance will require actual training and independently obtained evaluation results.

Read: [D04 — Attention & Long Context](docs/D04_attention_long_context.md).

---

## Native Tokenizer

SHINRA specifies an independent **byte-level BPE tokenizer** with a vocabulary capacity of **131,072 tokens**.

Its design supports English, Russian, programming languages and structured text without depending on a pretrained tokenizer from another model family.

### Vocabulary Layout

| Token IDs | Purpose |
|:---|:---|
| `0–31` | Special and reserved tokens |
| `32–287` | Complete 256-byte alphabet |
| `288–131071` | Learned BPE vocabulary |
| **Total** | **131,072 slots** |

### Tokenizer Design

- Full byte-level representability.
- Deterministic merge ranking.
- UTF-8-aware text interfaces.
- Whitespace and newline preservation.
- Explicit BOS/EOS semantics.
- Document segmentation tokens.
- Structured conversation token reservations.
- Fill-in-the-middle token reservations.
- Versioned vocabulary and merge serialization.
- Immutable tokenizer identity for model checkpoints.

No pretrained vocabulary or automatically selected public training corpus is included.

Tokenizer training and final vocabulary construction depend on an approved data corpus.

Read: [D05 — Native Tokenizer](docs/D05_native_tokenizer.md).

---

## Pretraining Philosophy

SHINRA is intended to be pretrained from randomly initialized model parameters.

The project does not depend on a pretrained base language model.

### Data Policy

Training-data provenance and quality are first-class engineering requirements.

The project does not automatically adopt general-purpose public pretraining mixtures or substitute synthetic samples for unavailable source material.

A future approved training corpus must provide traceable source metadata, licensing information, document identities and reproducible processing rules.

**The training corpus is not currently selected or released.**

### Training Infrastructure

The planned engineering scope includes:

- Causal language modeling.
- Mixed-precision execution.
- Distributed expert computation.
- Gradient accumulation and recomputation.
- Explicit optimizer and scheduler states.
- Transaction-aware checkpointing.
- Deterministic data-stream metadata.
- Failure handling and recovery semantics.

No particular GPU deployment, hardware topology, throughput or training schedule is claimed by this repository.

---

## Repository Structure

The project separates architecture documentation, model configuration and native source implementation.

```text
NULLXES-SHINRA-40B-Mixture-of-Experts/
│
├── docs/                    # Architecture and engineering specifications
├── configs/                 # Single canonical model YAML
│
├── src/shinra/
│   ├── core/                # Configuration, validation, initialization
│   ├── model/               # Transformer and attention components
│   ├── moe/                 # Routing, experts, dispatch and combine
│   ├── tokenizer/           # Native byte-level BPE
│   ├── data/                # Approved data source interfaces
│   ├── training/            # Pretraining and optimizer state
│   ├── distributed/         # Logical parallelism and communication
│   ├── checkpoint/          # Versioned checkpoints and recovery
│   └── inference/           # Autoregressive generation and KV-cache
│
├── requirements.txt         # Project-specific dependencies
├── pyproject.toml           # Package metadata
└── README.md
```

*The diagram represents the planned project structure. Its presence here does not imply that all components are currently implemented.*

SHINRA follows a **single-configuration policy**, centered on `configs/shinra_40b_moe.yaml`. Alternative Dense model configurations are outside the project's architecture.

---

## Documentation

The following technical documents are available in the repository:

| Document | Description |
|:---|:---|
| [D02 — Parameter Accounting](docs/D02_parameter_accounting.md) | Mathematical derivation of model parameters |
| [D03 — MoE Routing](docs/D03_moe_routing.md) | Expert selection, dispatch, balancing and backward contracts |
| [D04 — Attention & Long Context](docs/D04_attention_long_context.md) | GQA, QK-Norm, RoPE, attention masks and context |
| [D05 — Native Tokenizer](docs/D05_native_tokenizer.md) | Byte-level BPE, token IDs and serialization |
| [Documentation Index](docs/README.md) | Status of architecture decisions and design artifacts |

Documentation distinguishes accepted requirements, calculated properties and proposed decisions. Proposed settings are not automatically production defaults.

---

## Development Status

**Current classification: Architecture & Engineering.**

| Milestone | Status |
|:---|:---|
| Model architecture baseline | Documented |
| Parameter accounting | Calculated |
| MoE routing contract | Documented |
| Long-context attention contract | Documented |
| Native tokenizer contract | Documented |
| Source implementation | In development / static review only; model/checkpoint not runtime verified |
| Safetensors checkpoint source | Implemented; backend durability and training resume integration unverified |
| Tokenizer training | Not reported |
| Model pretraining | Not reported |
| Model evaluation | Not reported |
| Trained model weights | Not released |
| Production deployment | Not reported |

### Research and Engineering Disclaimer

This repository is an evolving engineering project.

Architectural parameter counts are not benchmark results.

Target context length does not establish effective long-context performance.

Documentation of a training or inference subsystem does not imply that it has been executed, validated or deployed.

No accuracy, reasoning, latency, throughput or training-efficiency claims are made without supporting measurements.

---

## Design Commitments

**No dense alternatives.**

SHINRA's feed-forward architecture is routed MoE throughout the model.

**No inherited model weights.**

The project is designed for independent initialization rather than pretrained-weight adaptation.

**No silent architectural degradation.**

Unsupported execution paths must not silently reduce model dimensions, context limits or expert counts.

**No fabricated research results.**

Unknown infrastructure, performance and training outcomes remain explicitly unverified.

**No unapproved corpus substitution.**

Training data is treated as a separately governed engineering asset.

**No undocumented architecture changes.**

Fundamental model decisions are defined through explicit engineering contracts.

---

## Responsible Development

SHINRA is an experimental language-model research and engineering project.

Future model releases should include appropriate documentation of training methodology, known limitations, intended use, safety considerations and evaluation evidence.

No trained model, evaluation suite or deployment is represented as available unless separately published.

---

## Organization

<div align="center">

### NULLXES

**INTELLIGENCE DIVISION**

*Engineering native intelligence systems.*

**Founded 2026 · Krasnodar**

[GitHub Repository](https://github.com/MagistrTheOne/NULLXES-SHINRA-40B-Mixture-of-Experts) · [Technical Documentation](docs/README.md)

---

**NULLXES — ONE INTELLIGENCE. DIFFERENT BODIES.**

**SHINRA / 40B MIXTURE OF EXPERTS**

© 2026 NULLXES

</div>
### Static checkpoint remediation — 25 October 2026

Three checkpoint write/verification defects were corrected in source; no save/load or model execution was performed. Publication and training resume remain NO-GO pending authorized runtime verification and storage/state integration. See [current audit](docs/AUDIT_REMEDIATION_2026-10-25.md) and [initialization candidate](docs/INITIALIZATION_RECIPE.md).

### Source continuation — 26 October 2026

Native tokenizer/data, fresh-run DP training, distributed primitives and single-sequence inference source now exists. Coordinated training resume and EP/TP/CP model assembly remain incomplete. No model, tokenizer training, checkpoint or tests executed. See [current implementation status](docs/IMPLEMENTATION_STATUS.md) and [Order 007 report](docs/ORDER_007_IMPLEMENTATION.md).
