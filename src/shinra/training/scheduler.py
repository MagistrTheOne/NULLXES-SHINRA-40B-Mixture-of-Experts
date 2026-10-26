"""Explicit constant or warmup-cosine learning rates, indexed by committed steps."""
import math
from shinra.core.configuration import Configuration, ConfigurationError


class StepScheduler:
    def __init__(self, config: Configuration, *, completed_step: int):
        self.kind = config.get("training.scheduler")
        self.base_lr = config.get("training.learning_rate")
        self.warmup = config.get("training.warmup_steps")
        self.total = config.get("training.total_steps")
        if self.kind not in ("constant", "linear_warmup_cosine"):
            raise ConfigurationError("training.scheduler", "explicit supported algorithm required")
        if any(x is None for x in (self.base_lr, self.warmup, self.total)):
            raise ConfigurationError("training.scheduler", "schedule values are UNKNOWN")
        if self.kind == "constant" and self.warmup != 0:
            raise ValueError("constant schedule requires explicitly zero warmup")
        if type(completed_step) is not int or not 0 <= completed_step <= self.total:
            raise ValueError("scheduler progress outside schedule")
        self.completed_step = completed_step

    def next_lr(self) -> float:
        step = self.completed_step
        if step >= self.total:
            raise RuntimeError("schedule exhausted")
        if self.kind == "constant":
            return self.base_lr
        if step < self.warmup:
            return self.base_lr * (step + 1) / self.warmup
        decay_steps = self.total - self.warmup
        progress = (step - self.warmup) / max(decay_steps - 1, 1)
        return self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))

    def commit(self, successful_step: int):
        if successful_step != self.completed_step + 1 or successful_step > self.total:
            raise RuntimeError("scheduler must advance exactly once")
        self.completed_step = successful_step

    def metadata(self) -> dict:
        return {"kind": self.kind, "base_lr": self.base_lr, "warmup": self.warmup,
                "total": self.total, "completed_step": self.completed_step}
