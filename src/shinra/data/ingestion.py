"""Explicit local UTF-8 document adapter, hash/length verified before admission."""
import hashlib
from pathlib import Path
from .registry import CorpusManifest
from .provenance import Document


class DocumentReader:
    def __init__(self, manifest: CorpusManifest, *, root: str | Path, io_chunk_bytes: int):
        if not isinstance(manifest, CorpusManifest):
            raise ValueError("approved manifest required")
        if type(io_chunk_bytes) is not int or io_chunk_bytes <= 0:
            raise ValueError("explicit positive I/O chunk size required")
        self.manifest = manifest
        self.root = Path(root).resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("document root must be an existing directory")
        self.io_chunk_bytes = io_chunk_bytes

    def read(self, ordinal: int) -> Document:
        if type(ordinal) is not int or not 0 <= ordinal < len(self.manifest.documents):
            raise ValueError("document ordinal outside manifest")
        record = self.manifest.documents[ordinal]
        relative = Path(record.relative_path)
        if relative.is_absolute() or relative.drive or ".." in relative.parts:
            raise ValueError("document path must be relative to approved root")
        path = (self.root / relative).resolve(strict=True)
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError("document path escapes root or is not a file")
        chunks, size, hasher = [], 0, hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(self.io_chunk_bytes):
                size += len(chunk)
                if size > record.byte_length:
                    raise ValueError("document exceeds declared byte length")
                hasher.update(chunk)
                chunks.append(chunk)
        if size != record.byte_length or hasher.hexdigest() != record.raw_sha256:
            raise ValueError(f"document integrity mismatch: {record.document_id}")
        payload = b"".join(chunks)
        payload.decode("utf-8", errors="strict")
        self.manifest.language_policy.admit(record.language)
        return Document(record, payload, self.manifest.identity)
