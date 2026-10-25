"""Transactional load accounting; global synchronization is an external obligation."""
from __future__ import annotations

import torch
from torch import nn

from shinra.core.configuration import Configuration, ConfigurationError
from shinra.core.validation import validate_architecture


class BalancingState(nn.Module):
    def __init__(self, config: Configuration, *, device: str, initial_bias: torch.Tensor):
        super().__init__()
        validate_architecture(config)
        for key in ("approval_reference", "learning_rate", "bound"):
            if config.get(f"router.balancing.{key}") is None:
                raise ConfigurationError(f"router.balancing.{key}", "explicit approved value required")
        self.rate = config.get("router.balancing.learning_rate")
        self.bound = config.get("router.balancing.bound")
        expected_shape = (config.get("model.num_routed_experts"),)
        if (initial_bias.shape != expected_shape or initial_bias.dtype != torch.float32
                or initial_bias.device != torch.device(device) or initial_bias.requires_grad):
            raise ValueError("initial balancing bias must be non-trainable FP32 on the configured device")
        if not bool(torch.isfinite(initial_bias).all()) or bool((initial_bias.abs() > self.bound).any()):
            raise ValueError("initial balancing bias must be finite and within the configured bound")
        self.register_buffer("bias", initial_bias.detach().clone())
        self.register_buffer("completed_step", torch.zeros((), dtype=torch.int64, device=device))
        self.phase = "COMMITTED"
        self.attempt: str | None = None
        self.events: set[str] = set()
        self.pending = [0] * config.get("model.num_routed_experts")
        self.next_bias: torch.Tensor | None = None

    def begin(self, attempt: str, *, completed_step: int) -> torch.Tensor:
        if self.phase != "COMMITTED" or not isinstance(attempt, str) or not attempt:
            raise RuntimeError("begin requires committed state and nonempty attempt identity")
        if type(completed_step) is not int or completed_step != int(self.completed_step.item()):
            raise RuntimeError("balancing step identity mismatch")
        if self.bias.dtype != torch.float32 or not bool(torch.isfinite(self.bias).all()):
            raise RuntimeError("invalid balancing buffer")
        if bool((self.bias.abs() > self.bound).any()):
            raise RuntimeError("balancing buffer exceeds configured bound")
        self.attempt = attempt
        self.phase = "OPEN"
        self.events.clear()
        self.pending = [0] * self.bias.numel()
        return self.bias.detach().clone()

    def record(self, attempt: str, event: str, counts: torch.Tensor, *, recomputation: bool) -> None:
        if self.phase != "OPEN" or attempt != self.attempt:
            raise RuntimeError("load event outside active attempt")
        if not isinstance(event, str) or not event:
            raise ValueError("load event requires stable identity")
        if recomputation:
            if event not in self.events:
                raise RuntimeError("recomputation has no original load event")
            return
        if event in self.events:
            raise RuntimeError("duplicate original-forward load event")
        if counts.dtype != torch.int64 or counts.shape != self.bias.shape:
            raise ValueError("load counts must be int64 [128]")
        values = counts.detach().cpu().tolist()
        if any(n < 0 for n in values) or sum(values) % 8:
            raise ValueError("invalid Top-8 assignment counts")
        updated = [a + b for a, b in zip(self.pending, values)]
        if sum(updated) > 2**63 - 1:
            raise OverflowError("load counts exceed int64 reduction range")
        self.pending = updated
        self.events.add(event)

    def local_counts(self, attempt: str) -> tuple[int, ...]:
        if self.phase != "OPEN" or attempt != self.attempt:
            raise RuntimeError("counts requested outside active attempt")
        return tuple(self.pending)

    def prepare(self, attempt: str, *, synchronized_counts: tuple[int, ...], unique_tokens: int) -> None:
        """Receive already synchronized unique-contributor counts from the engine.

        Does not perform a collective or assert that one occurred. Engine must
        validate membership and deduplicate tensor-parallel replicas beforehand.
        """
        if self.phase != "OPEN" or attempt != self.attempt:
            raise RuntimeError("prepare outside active attempt")
        if type(unique_tokens) is not int or unique_tokens <= 0 or 8 * unique_tokens > 2**63 - 1:
            raise ValueError("global token count must be positive and fit int64 assignments")
        if len(synchronized_counts) != self.bias.numel() or any(type(n) is not int or n < 0 for n in synchronized_counts):
            raise ValueError("invalid synchronized expert counts")
        if sum(synchronized_counts) != 8 * unique_tokens:
            raise ValueError("synchronized counts do not sum to 8*N")
        if any(g < l for g, l in zip(synchronized_counts, self.pending)):
            raise ValueError("synchronized counts omit local contributions")
        # Python integer comparisons avoid overflow of 128*n in int64.
        target = 8 * unique_tokens
        direction = [1 if 128*n < target else -1 if 128*n > target else 0 for n in synchronized_counts]
        with torch.no_grad():
            delta = torch.tensor(direction, dtype=torch.float32, device=self.bias.device)
            self.next_bias = (self.bias + self.rate * delta).clamp(-self.bound, self.bound)
        if not bool(torch.isfinite(self.next_bias).all()):
            self.next_bias = None
            raise FloatingPointError("nonfinite prepared balancing update")
        self.phase = "PREPARED"

    def start_optimizer_mutation(self, attempt: str) -> None:
        if self.phase != "PREPARED" or attempt != self.attempt:
            raise RuntimeError("optimizer mutation requires prepared attempt")
        self.phase = "COMMITTING"

    def commit(self, attempt: str, *, completed_step: int) -> None:
        if self.phase != "COMMITTING" or attempt != self.attempt or self.next_bias is None:
            raise RuntimeError("commit requires successful coordinated optimizer mutation")
        previous = int(self.completed_step.item())
        if type(completed_step) is not int or completed_step != previous + 1 or completed_step >= 2**63:
            raise ValueError("completed step must advance exactly once")
        # Any exception in this region leaves COMMITTING: recovery is mandatory.
        with torch.no_grad():
            self.bias.copy_(self.next_bias)
            self.completed_step.fill_(completed_step)
        self.next_bias = None
        self.events.clear()
        self.pending = [0] * self.bias.numel()
        self.attempt = None
        self.phase = "COMMITTED"

    def abort(self, attempt: str) -> None:
        if attempt != self.attempt or self.phase not in ("OPEN", "PREPARED", "COMMITTING"):
            raise RuntimeError("no matching active attempt")
        if self.phase == "COMMITTING":
            self.phase = "RECOVERY_REQUIRED"
            raise RuntimeError("partial optimizer mutation: restore a committed checkpoint")
        self.next_bias = None
        self.events.clear()
        self.pending = [0] * self.bias.numel()
        self.attempt = None
        self.phase = "COMMITTED"

    def checkpoint_metadata(self) -> dict:
        if self.phase != "COMMITTED":
            raise RuntimeError("checkpoint publication requires committed balancing state")
        return {"algorithm": "bounded_sign", "learning_rate": self.rate, "bound": self.bound,
                "completed_step": int(self.completed_step.item())}
