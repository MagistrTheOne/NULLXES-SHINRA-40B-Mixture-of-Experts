"""Evidence-backed language metadata; no inferred classifier or fabricated label."""
from dataclasses import dataclass


@dataclass(frozen=True)
class LanguageEvidence:
    label: str
    method: str
    version: str
    provenance: str

    def __post_init__(self):
        if self.label not in ("ru", "en", "mixed", "other", "unknown"):
            raise ValueError("unsupported language label")
        if any(not isinstance(x, str) or not x.strip() for x in (self.method, self.version, self.provenance)):
            raise ValueError("language evidence requires method/version/provenance")


@dataclass(frozen=True)
class LanguagePolicy:
    identity: str
    allowed_labels: tuple[str, ...]
    approval_reference: str

    def __post_init__(self):
        if not self.identity or not self.approval_reference:
            raise ValueError("explicit language policy identity and approval required")
        if not self.allowed_labels or len(set(self.allowed_labels)) != len(self.allowed_labels):
            raise ValueError("language admission requires nonempty unique labels")
        if any(x not in ("ru", "en", "mixed", "other", "unknown") for x in self.allowed_labels):
            raise ValueError("invalid language admission label")

    def admit(self, evidence: LanguageEvidence) -> None:
        if evidence.label not in self.allowed_labels:
            raise ValueError(f"document language {evidence.label} not admitted by {self.identity}")
