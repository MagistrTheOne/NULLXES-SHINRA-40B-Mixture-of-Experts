"""Native exact blockwise GQA with explicit backward, no backend fallback.

The torch_blockwise_exact backend uses FP32 score/softmax accumulation. It is
not a performance claim or a distributed context-parallel implementation.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd.function import once_differentiable

from shinra.core.configuration import Configuration, ConfigurationError
from shinra.core.initialization import InitializationRecipe, ParameterSpec, initialize_parameter
from shinra.core.validation import require_operation
from .normalization import RMSNorm
from .rope import RotaryEmbedding


def require_ieee_matmul(device: torch.device) -> None:
    if device.type == "cuda" and torch.backends.cuda.matmul.allow_tf32:
        raise RuntimeError("FP32 contract requires TF32-disabled matmul; configure the authorized runtime explicitly")


def _mask(qp, kp, qs, ks):
    return (qs[:, None] == ks[None, :]) & (kp[None, :] <= qp[:, None])


class _BlockwiseGQA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, qpos, kpos, qseg, kseg, qchunk, kchunk):
        # q=[Tq,Hq,D], k/v=[Tk,Hkv,D]; all queries and keys are valid tokens.
        tq, heads, width = q.shape
        ratio = heads // k.shape[1]
        scale = 1.0 / math.sqrt(width)
        out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
        lse = torch.empty((tq, heads), dtype=torch.float32, device=q.device)
        with torch.autocast(device_type=q.device.type, enabled=False):
            for head in range(heads):
                kh = head // ratio
                for start in range(0, tq, qchunk):
                    stop = min(start + qchunk, tq)
                    queries = q[start:stop, head].float()
                    maximum = torch.full((stop-start,), -torch.inf, dtype=torch.float32, device=q.device)
                    normalizer = torch.zeros_like(maximum)
                    numerator = torch.zeros((stop-start, width), dtype=torch.float32, device=q.device)
                    for left in range(0, k.shape[0], kchunk):
                        right = min(left+kchunk, k.shape[0])
                        allowed = _mask(qpos[start:stop], kpos[left:right], qseg[start:stop], kseg[left:right])
                        scores = (queries @ k[left:right, kh].float().T) * scale
                        scores = scores.masked_fill(~allowed, -torch.inf)
                        updated = torch.maximum(maximum, scores.max(dim=-1).values)
                        safe_max = torch.where(torch.isfinite(updated), updated, torch.zeros_like(updated))
                        alpha = torch.exp(maximum-safe_max)
                        probabilities = torch.exp(scores-safe_max[:, None])
                        numerator = numerator*alpha[:, None] + probabilities @ v[left:right, kh].float()
                        normalizer = normalizer*alpha + probabilities.sum(dim=-1)
                        maximum = updated
                    if bool((normalizer <= 0).any()):
                        raise ValueError("attention query has no allowed key")
                    out[start:stop, head] = numerator/normalizer[:, None]
                    lse[start:stop, head] = maximum + normalizer.log()
        if not bool(torch.isfinite(out).all()) or not bool(torch.isfinite(lse).all()):
            raise FloatingPointError("nonfinite attention result")
        ctx.save_for_backward(q, k, v, qpos, kpos, qseg, kseg, out, lse)
        ctx.chunks = (qchunk, kchunk)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        q, k, v, qpos, kpos, qseg, kseg, out, lse = ctx.saved_tensors
        qchunk, kchunk = ctx.chunks
        require_ieee_matmul(q.device)
        if not bool(torch.isfinite(grad_out).all()):
            raise FloatingPointError("nonfinite attention output gradient")
        dq = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
        dk = torch.zeros(k.shape, dtype=torch.float32, device=k.device)
        dv = torch.zeros(v.shape, dtype=torch.float32, device=v.device)
        ratio = q.shape[1] // k.shape[1]
        scale = 1.0 / math.sqrt(q.shape[-1])
        with torch.autocast(device_type=q.device.type, enabled=False):
            for head in range(q.shape[1]):
                kh = head // ratio
                for start in range(0, q.shape[0], qchunk):
                    stop = min(start+qchunk, q.shape[0])
                    queries = q[start:stop, head].float()
                    upstream = grad_out[start:stop, head].float()
                    delta = (upstream*out[start:stop, head]).sum(dim=-1, keepdim=True)
                    for left in range(0, k.shape[0], kchunk):
                        right = min(left+kchunk, k.shape[0])
                        keys, values = k[left:right, kh].float(), v[left:right, kh].float()
                        allowed = _mask(qpos[start:stop], kpos[left:right], qseg[start:stop], kseg[left:right])
                        scores = ((queries @ keys.T)*scale).masked_fill(~allowed, -torch.inf)
                        p = torch.exp(scores-lse[start:stop, head, None])
                        ds = p*((upstream @ values.T)-delta)
                        dq[start:stop, head] += (ds @ keys)*scale
                        dk[left:right, kh] += (ds.T @ queries)*scale
                        dv[left:right, kh] += p.T @ upstream
        if not all(bool(torch.isfinite(g).all()) for g in (dq, dk, dv)):
            raise FloatingPointError("nonfinite attention gradients")
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None, None, None, None, None


class Attention(nn.Module):
    def __init__(self, config: Configuration, recipe: InitializationRecipe, *, layer_id: int, device: str, context_execution=None):
        super().__init__()
        self.context_execution = context_execution
        require_operation(config, "model")
        if config.get("attention.backend") != "torch_blockwise_exact":
            raise ConfigurationError("attention.backend", "explicit torch_blockwise_exact selection required; no fallback")
        if config.get("attention.dropout") != 0:
            raise ConfigurationError("attention.dropout", "this exact backend requires explicit zero dropout")
        m = config.get("model")
        if type(layer_id) is not int or not 0 <= layer_id < m["num_hidden_layers"]:
            raise ValueError("invalid layer_id")
        self.hidden, self.heads, self.kv_heads, self.width = m["hidden_size"], m["num_attention_heads"], m["num_key_value_heads"], m["head_dim"]
        self.qchunk, self.kchunk = config.get("attention.query_chunk_tokens"), config.get("attention.key_chunk_tokens")
        prefix = f"layers.{layer_id}.attention"
        for name, shape, role in (("q_proj", (self.heads*self.width, self.hidden), "projection"),
                                  ("k_proj", (self.kv_heads*self.width, self.hidden), "projection"),
                                  ("v_proj", (self.kv_heads*self.width, self.hidden), "projection"),
                                  ("o_proj", (self.hidden, self.heads*self.width), "residual_projection")):
            # ParameterDict gives the canonical '<projection>.weight' key.
            setattr(self, name, nn.ParameterDict({"weight": initialize_parameter(
                ParameterSpec(f"{prefix}.{name}.weight", shape, "bfloat16", role), recipe, device=device)}))
        self.q_norm = RMSNorm(config, recipe, name=f"{prefix}.q_norm.weight", device=device, head_kind="query")
        self.k_norm = RMSNorm(config, recipe, name=f"{prefix}.k_norm.weight", device=device, head_kind="key")
        self.rope = RotaryEmbedding(config)

    def project(self, hidden, positions):
        if hidden.ndim != 2 or hidden.shape[-1] != self.hidden:
            raise ValueError("attention expects compact valid hidden [T,2560]")
        if hidden.dtype != self.q_proj["weight"].dtype or hidden.device != self.q_proj["weight"].device:
            raise ValueError("attention hidden dtype/device mismatch")
        q = F.linear(hidden, self.q_proj["weight"]).reshape(-1, self.heads, self.width)
        k = F.linear(hidden, self.k_proj["weight"]).reshape(-1, self.kv_heads, self.width)
        v = F.linear(hidden, self.v_proj["weight"]).reshape(-1, self.kv_heads, self.width)
        return self.rope(self.q_norm(q), positions), self.rope(self.k_norm(k), positions), v

    def attend(self, q, k, v, qpos, kpos, qseg, kseg):
        require_ieee_matmul(q.device)
        if q.shape[1:] != (self.heads,self.width) or k.shape[1:] != (self.kv_heads,self.width) or k.shape != v.shape:
            raise ValueError("attention Q/K/V shape mismatch")
        for tensor, length in ((qpos,q.shape[0]), (qseg,q.shape[0]), (kpos,k.shape[0]), (kseg,k.shape[0])):
            if tensor.dtype != torch.int64 or tensor.shape != (length,) or tensor.device != q.device:
                raise ValueError("attention metadata must be int64 [T] on Q device")
        if k.device != q.device or v.device != q.device or k.dtype != q.dtype or v.dtype != q.dtype:
            raise ValueError("attention Q/K/V dtype/device mismatch")
        if not all(bool(torch.isfinite(t).all()) for t in (q,k,v)):
            raise FloatingPointError("nonfinite attention Q/K/V")
        output = _BlockwiseGQA.apply(q,k,v,qpos,kpos,qseg,kseg,self.qchunk,self.kchunk)
        result = F.linear(output.to(q.dtype).reshape(q.shape[0], self.heads*self.width), self.o_proj["weight"])
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError("nonfinite attention output projection")
        return result

    def forward(self, hidden, positions, segments):
        q,k,v = self.project(hidden, positions)
        if self.context_execution is not None:
            return self.context_execution.attend(self, q, k, v, positions, segments)
        return self.attend(q,k,v,positions,positions,segments,segments)
