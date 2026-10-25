# Native initialization — candidate and implementation contract

25.10.2026. PROPOSED; not executable defaults. Authority: Engineering Order 005.

## Recipe

For each non-normalization logical tensor, sample Normal(0, sigma) in FP32 using an explicit per-tensor generator, then cast once to its storage dtype. Input embedding and output head are separate parameters with separate logical names. Experts have distinct layer/expert/projection names. No pretrained loading or import-time allocation occurs.

- Base sigma = 1/sqrt(2560) = 0.01976423537605237 (CALCULATED).
- Residual multiplier = 1/sqrt(2*48) = 0.10206207261596577 (CALCULATED).
- Residual sigma = 1/sqrt(2560*96) = 0.002017178826149696 (CALCULATED).
- Residual scaling applies only to attention `o_proj` and expert `down` weights.
- RMSNorm and per-head QK-Norm gamma = 1. No trainable biases.
- Router: FP32 Normal(0, base sigma) candidate, no separate hidden scale. This is a nonzero random distribution, not a promise that every sampled element is nonzero or that this scale is optimal.
- Other parameters: BF16 storage after FP32 sampling.
- Balancing bias: explicit FP32 zero vector candidate, separate from trainable parameters; constructor requires caller-provided state. Fresh completed step is zero; resume must restore saved step and bias together.
- Global seed: UNKNOWN, required explicitly in [single YAML](../configs/shinra_40b_moe.yaml), together with approval_reference, distribution, std and residual_scale. None were filled by this audit.

## Logical inventory

`core/initialization.py::parameter_specs` defines 18,867 trainable tensors:

| Canonical name pattern | Shape | Storage |
|---|---|---|
| embeddings.weight; lm_head.weight (independent) | [131072,2560] each | BF16 |
| layers.L.attention_norm.weight; layers.L.moe_norm.weight | [2560] each | BF16 |
| layers.L.attention.q_proj.weight | [4096,2560] | BF16 |
| layers.L.attention.k_proj.weight; v_proj.weight | [1024,2560] each | BF16 |
| layers.L.attention.o_proj.weight | [2560,4096] | BF16 |
| layers.L.attention.q_norm.weight | [32,128] | BF16 |
| layers.L.attention.k_norm.weight | [8,128] | BF16 |
| layers.L.moe.router.weight | [128,2560] | FP32 |
| layers.L.moe.experts.E.gate.weight; up.weight | [800,2560] each | BF16 |
| layers.L.moe.experts.E.down.weight | [2560,800] | BF16 |
| final_norm.weight | [2560] | BF16 |

L=0..47; E=0..127. Count = 3 + 48*(9 + 128*3). Trainable total = 39,694,338,560; balancing buffers are excluded. Names/shapes match source constructors by static inspection; no model state_dict was instantiated.

## RNG identity and limitations

Current seed derivation is first eight bytes of SHA-256 over UTF-8 `shinra.init.v1:{seed}:{logical_name}`, interpreted big-endian, modulo 2^63. A generator is seeded separately for each tensor. Distinct names avoid intentional reuse of streams; a truncated hash is not a mathematical guarantee against collisions. No bitwise independence or cross-device reproducibility claim is made.

Reproducibility metadata must include recipe, logical name inventory, source revision, exact Python/PyTorch versions, generator device/algorithm and runtime policy. Those runtime facts are UNKNOWN. Same logical tensor construction repeats its seed deliberately; model assembly constructs each parameter once and does not call a later global reset initializer. Partition-independent sampling is not implemented: resharded initialization requires a separate accepted contract.

## Status

VERIFIED: source contains FP32 normal sampling, explicit storage cast, gamma fill, logical seed derivation and canonical inventory. CALCULATED: dimensions and numbers above. PROPOSED: numeric recipe. BLOCKED: execution and adoption of unknown recipe values; training-state producers/RNG capture integration are absent. No initialization, tensors, checkpoints, training or tests were executed for this document.
