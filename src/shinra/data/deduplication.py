"""Exact byte deduplication with explicit duplicate provenance results."""
from dataclasses import dataclass
from .provenance import Document, DocumentRecord


@dataclass(frozen=True)
class DeduplicationResult:
    document: Document
    retained: bool
    original: DocumentRecord


class ExactDeduplicator:
    def __init__(self):
        self.buckets: dict[str, list[Document]] = {}

    def admit(self, document: Document) -> DeduplicationResult:
        bucket = self.buckets.setdefault(document.record.raw_sha256, [])
        for original in bucket:
            if original.payload == document.payload:
                return DeduplicationResult(document, False, original.record)
        bucket.append(document)
        return DeduplicationResult(document, True, document.record)
