"""Externally approved structural registry; ordinary text never creates specials."""
from dataclasses import dataclass
from types import MappingProxyType
from collections.abc import Mapping

from shinra.core.configuration import Configuration
from shinra.core.validation import SPECIAL_NAMES, require_operation


@dataclass(frozen=True)
class SpecialRegistry:
    ids: Mapping[str, int]
    approval_reference: str

    def __post_init__(self):
        if not isinstance(self.approval_reference, str) or not self.approval_reference.strip():
            raise ValueError("special registry requires explicit approval")
        if set(self.ids) != set(SPECIAL_NAMES):
            raise ValueError("special registry names differ from D05 contract")
        if any(type(self.ids[name]) is not int or self.ids[name] != i for i, name in enumerate(SPECIAL_NAMES)):
            raise ValueError("registry must explicitly adopt D05 candidate IDs; no remapping")
        object.__setattr__(self, "ids", MappingProxyType(dict(self.ids)))

    @classmethod
    def from_configuration(cls, config: Configuration):
        require_operation(config, "tokenizer")
        return cls(config.get("tokenizer.special_token_ids"), config.get("tokenizer.approval_reference"))

    def token(self, name: str) -> int:
        if name not in self.ids or name.startswith("reserved_"):
            raise ValueError("unknown or unassigned structural token")
        return self.ids[name]

    def name(self, token_id: int) -> str:
        if type(token_id) is not int or not 0 <= token_id < len(SPECIAL_NAMES):
            raise ValueError("not a structural token ID")
        name = SPECIAL_NAMES[token_id]
        if name.startswith("reserved_"):
            raise ValueError("unassigned structural token")
        return name


@dataclass(frozen=True)
class Special:
    name: str
