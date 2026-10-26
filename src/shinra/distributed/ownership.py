"""Logical expert ownership independent of physical hardware assignment."""
from dataclasses import dataclass
import hashlib
import json


@dataclass(frozen=True)
class ExpertOwnership:
    owners: tuple[tuple[int, ...], ...]  # [48,128], group-local owner indices
    members: int
    approval_reference: str

    def __post_init__(self):
        if type(self.members) is not int or self.members <= 0 or not self.approval_reference:
            raise ValueError("explicit logical membership and approval required")
        if len(self.owners) != 48 or any(len(row) != 128 for row in self.owners):
            raise ValueError("ownership must cover exactly 48*128 experts")
        if any(type(owner) is not int or not 0 <= owner < self.members for row in self.owners for owner in row):
            raise ValueError("invalid logical expert owner")

    @property
    def identity(self):
        payload = json.dumps({"owners": self.owners, "members": self.members,
                              "approval_reference": self.approval_reference}, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def local_experts(self, layer: int, rank: int) -> tuple[int, ...]:
        if not 0 <= layer < 48 or not 0 <= rank < self.members:
            raise ValueError("ownership index out of range")
        return tuple(i for i, owner in enumerate(self.owners[layer]) if owner == rank)
