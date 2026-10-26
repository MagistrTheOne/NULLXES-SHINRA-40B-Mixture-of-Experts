"""Strict approved manifest admission; no built-in corpus or network adapter."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
from types import MappingProxyType
from collections.abc import Mapping

from shinra.core.configuration import Configuration
from shinra.core.validation import require_operation
from shinra.tokenizer.serialization import read_json_bytes, exact_keys
from .provenance import SourceRecord, DocumentRecord
from .language import LanguageEvidence, LanguagePolicy


@dataclass(frozen=True)
class CorpusManifest:
    identity: str
    approval_reference: str
    sources: Mapping[str, SourceRecord]
    documents: tuple[DocumentRecord, ...]
    language_policy: LanguagePolicy


def load_manifest(config: Configuration) -> CorpusManifest:
    require_operation(config, "ingest")
    payload = Path(config.get("data.manifest_path")).read_bytes()
    identity = hashlib.sha256(payload).hexdigest()
    if identity != config.get("data.manifest_identity"):
        raise ValueError("corpus manifest identity mismatch")
    body = read_json_bytes(payload)
    exact_keys(body, ("schema_version", "approval_reference", "sources", "documents", "language_policy"))
    if type(body["schema_version"]) is not int or body["schema_version"] != 1:
        raise ValueError("unsupported corpus manifest schema")
    if body["approval_reference"] != config.get("data.approval_reference"):
        raise ValueError("corpus approval mismatch")
    exact_keys(body["language_policy"], ("identity", "allowed_labels", "approval_reference"))
    if not isinstance(body["language_policy"]["allowed_labels"], list):
        raise ValueError("language policy labels must be a list")
    policy = LanguagePolicy(body["language_policy"]["identity"], tuple(body["language_policy"]["allowed_labels"]),
                            body["language_policy"]["approval_reference"])
    if policy.identity != config.get("data.language_policy"):
        raise ValueError("language policy identity mismatch")
    if not isinstance(body["sources"], list) or not body["sources"]:
        raise ValueError("explicit source registry required")
    sources = {}
    for entry in body["sources"]:
        exact_keys(entry, SourceRecord.__dataclass_fields__)
        source = SourceRecord(**entry)
        if source.source_id in sources:
            raise ValueError("duplicate source identity")
        sources[source.source_id] = source
    if not isinstance(body["documents"], list) or not body["documents"]:
        raise ValueError("manifest must list real documents explicitly")
    documents, seen = [], set()
    for entry in body["documents"]:
        exact_keys(entry, DocumentRecord.__dataclass_fields__)
        exact_keys(entry["language"], LanguageEvidence.__dataclass_fields__)
        if not isinstance(entry["transformation_history"], list):
            raise ValueError("transformation history must be a list")
        record = DocumentRecord(**{**entry, "language": LanguageEvidence(**entry["language"]),
                                   "transformation_history": tuple(entry["transformation_history"])})
        source = sources.get(record.source_id)
        if source is None or source.revision != record.source_revision:
            raise ValueError("document references unregistered source revision")
        key = (record.source_id, record.document_id)
        if key in seen:
            raise ValueError("duplicate document identity")
        seen.add(key)
        policy.admit(record.language)
        documents.append(record)
    return CorpusManifest(identity, body["approval_reference"], MappingProxyType(sources), tuple(documents), policy)
