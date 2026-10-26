"""Immutable source and document lineage with exact content identities."""
from dataclasses import dataclass
from .language import LanguageEvidence


def digest(value: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("expected lowercase SHA-256")
    return value


def required_text(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("nonempty text metadata required")
    return value


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    revision: str
    approval_reference: str
    license_identifier: str
    license_text_digest: str
    permitted_use: str
    owner: str

    def __post_init__(self):
        for key in ("source_id", "revision", "approval_reference", "license_identifier", "permitted_use", "owner"):
            required_text(getattr(self, key))
        digest(self.license_text_digest)


@dataclass(frozen=True)
class DocumentRecord:
    document_id: str
    source_id: str
    source_revision: str
    relative_path: str
    raw_sha256: str
    byte_length: int
    language: LanguageEvidence
    transformation_history: tuple[str, ...]

    def __post_init__(self):
        for key in ("document_id", "source_id", "source_revision", "relative_path"):
            required_text(getattr(self, key))
        digest(self.raw_sha256)
        if type(self.byte_length) is not int or self.byte_length < 0:
            raise ValueError("document byte length must be a nonnegative integer")
        if not isinstance(self.language, LanguageEvidence):
            raise ValueError("language evidence required")
        if not isinstance(self.transformation_history, tuple):
            raise ValueError("transformation history must be immutable")
        for item in self.transformation_history:
            required_text(item)


@dataclass(frozen=True)
class Document:
    record: DocumentRecord
    payload: bytes
    manifest_identity: str
