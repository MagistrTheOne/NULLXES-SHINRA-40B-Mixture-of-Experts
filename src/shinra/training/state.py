"""Logical optimizer transaction; partial mutations force durable recovery."""
from dataclasses import dataclass


@dataclass
class TrainingState:
    successful_step: int
    consumed_tokens: int
    phase: str = "COMMITTED"
    attempt: str | None = None

    def __post_init__(self):
        if any(type(x) is not int or x < 0 for x in (self.successful_step, self.consumed_tokens)):
            raise ValueError("training progress must be explicit nonnegative integers")
        if self.phase != "COMMITTED" or self.attempt is not None:
            raise ValueError("construct only a complete committed training state")

    def begin(self, attempt: str):
        if self.phase != "COMMITTED" or not isinstance(attempt, str) or not attempt:
            raise RuntimeError("cannot open training attempt")
        self.attempt, self.phase = attempt, "OPEN"

    def prepare(self):
        if self.phase != "OPEN":
            raise RuntimeError("prepare requires OPEN")
        self.phase = "PREPARED"

    def start_mutation(self):
        if self.phase != "PREPARED":
            raise RuntimeError("mutation requires PREPARED")
        self.phase = "COMMITTING"

    def commit(self, tokens: int):
        if self.phase != "COMMITTING" or type(tokens) is not int or tokens <= 0:
            raise RuntimeError("invalid successful commit")
        self.successful_step += 1
        self.consumed_tokens += tokens
        self.phase, self.attempt = "COMMITTED", None

    def abort(self):
        if self.phase == "COMMITTING":
            self.phase = "RECOVERY_REQUIRED"
        elif self.phase in ("OPEN", "PREPARED"):
            self.phase, self.attempt = "COMMITTED", None
        else:
            raise RuntimeError("no abortable attempt")

    def metadata(self):
        if self.phase != "COMMITTED":
            raise RuntimeError("cannot checkpoint uncommitted training state")
        return {"successful_step": self.successful_step, "consumed_tokens": self.consumed_tokens,
                "phase": self.phase}
