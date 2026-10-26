"""Manifest-order streaming with explicit committed and pending cursors."""
from dataclasses import dataclass
from .ingestion import DocumentReader
from .deduplication import ExactDeduplicator
from .packing import DocumentPacker, PackedBatch


@dataclass(frozen=True)
class StreamCursor:
    manifest_identity: str
    tokenizer_identity: str
    ordinal: int
    order: str
    start_ordinal: int
    end_ordinal: int

    def metadata(self) -> dict:
        return {"manifest_identity": self.manifest_identity, "tokenizer_identity": self.tokenizer_identity,
                "ordinal": self.ordinal, "order": self.order,
                "start_ordinal": self.start_ordinal, "end_ordinal": self.end_ordinal}


class PackedStream:
    """One explicit manifest-order corpus pass, with exact dedup prefix replay.

    No shuffle/epoch default. A future shuffled stream needs its own accepted
    cursor contract. Restoring rebuilds dedup state from verified source bytes;
    it does not silently skip unread data or persist an incomplete index.
    """
    def __init__(self, reader: DocumentReader, packer: DocumentPacker, cursor: StreamCursor):
        if reader.manifest.identity != packer.manifest_identity:
            raise ValueError("reader/packer manifest mismatch")
        self.reader, self.packer = reader, packer
        self.committed = cursor
        self.attempt = None
        self.duplicates: list[dict] = []
        self._seek(cursor)

    def _seek(self, cursor: StreamCursor):
        if not isinstance(cursor, StreamCursor) or cursor.manifest_identity != self.reader.manifest.identity or cursor.tokenizer_identity != self.packer.tokenizer_identity:
            raise ValueError("stream cursor identity mismatch")
        if (cursor.order != "manifest_order" or any(type(x) is not int for x in (cursor.ordinal, cursor.start_ordinal, cursor.end_ordinal))
                or not 0 <= cursor.start_ordinal <= cursor.ordinal <= cursor.end_ordinal <= len(self.reader.manifest.documents)):
            raise ValueError("invalid cursor/order")
        dedup = ExactDeduplicator()
        for ordinal in range(cursor.ordinal):
            dedup.admit(self.reader.read(ordinal))
        self.dedup, self.ordinal = dedup, cursor.ordinal

    def begin(self, attempt: str):
        if self.attempt is not None or not isinstance(attempt, str) or not attempt:
            raise RuntimeError("stream requires unique nonempty attempt")
        self.attempt = attempt
        self.duplicates = []

    def pending_cursor(self) -> StreamCursor:
        return StreamCursor(self.reader.manifest.identity, self.packer.tokenizer_identity,
                            self.ordinal, "manifest_order", self.committed.start_ordinal, self.committed.end_ordinal)

    def next_batch(self) -> PackedBatch | None:
        if self.attempt is None:
            raise RuntimeError("reading requires active transaction")
        segments, length = [], 0
        while self.ordinal < self.committed.end_ordinal:
            ordinal = self.ordinal
            document = self.reader.read(ordinal)
            decision = self.dedup.admit(document)
            if not decision.retained:
                self.duplicates.append({"source": document.record.source_id, "document": document.record.document_id,
                                        "original_source": decision.original.source_id,
                                        "original_document": decision.original.document_id})
                self.ordinal += 1
                continue
            segment = self.packer.segment(document, ordinal)
            required = len(segment.tokens) + int(bool(segments))
            if length + required > self.packer.capacity:
                # Undo the peek's dedup admission by restoring the exact prefix.
                self._seek(self.pending_cursor())
                break
            segments.append(segment)
            length += required
            self.ordinal += 1
        return self.packer.pack(tuple(segments)) if segments else None

    def commit(self, attempt: str):
        if self.attempt != attempt:
            raise RuntimeError("stream commit attempt mismatch")
        self.committed = self.pending_cursor()
        self.attempt = None

    def abort(self, attempt: str):
        if self.attempt != attempt:
            raise RuntimeError("stream abort attempt mismatch")
        self._seek(self.committed)
        self.attempt = None
        self.duplicates = []
