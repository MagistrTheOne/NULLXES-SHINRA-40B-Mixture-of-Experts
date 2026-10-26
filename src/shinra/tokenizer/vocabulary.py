"""Immutable native byte payloads; partial training state is never a full artifact."""
from dataclasses import dataclass
from .special_tokens import SpecialRegistry


@dataclass(frozen=True)
class Vocabulary:
    registry: SpecialRegistry
    payloads: tuple[bytes, ...]
    capacity: int
    byte_start: int
    max_token_bytes: int

    def __post_init__(self):
        if type(self.capacity) is not int or self.capacity != 131072 or type(self.byte_start) is not int or self.byte_start != 32:
            raise ValueError("vocabulary must preserve SHINRA capacity and byte allocation")
        if type(self.max_token_bytes) is not int or self.max_token_bytes < 1:
            raise ValueError("explicit positive token byte limit required")
        if not isinstance(self.registry, SpecialRegistry):
            raise ValueError("approved registry required")
        object.__setattr__(self, "payloads", tuple(self.payloads))
        if not 256 <= len(self.payloads) <= self.capacity - self.byte_start:
            raise ValueError("invalid ordinary vocabulary length")
        if self.payloads[:256] != tuple(bytes([b]) for b in range(256)):
            raise ValueError("complete ordered byte alphabet required")
        if any(type(p) is not bytes or not 1 <= len(p) <= self.max_token_bytes for p in self.payloads):
            raise ValueError("invalid token byte payload")
        if len(set(self.payloads)) != len(self.payloads):
            raise ValueError("duplicate ordinary payload")

    @property
    def complete(self) -> bool:
        return self.byte_start + len(self.payloads) == self.capacity

    def payload(self, token_id: int) -> bytes:
        if type(token_id) is not int or not self.byte_start <= token_id < self.byte_start + len(self.payloads):
            raise ValueError("unknown or structural token has no byte payload")
        return self.payloads[token_id - self.byte_start]
