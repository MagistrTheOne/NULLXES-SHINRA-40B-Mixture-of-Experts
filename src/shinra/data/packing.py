"""Whole-document compact packing with explicit next-token loss eligibility."""
from dataclasses import dataclass

from shinra.core.configuration import Configuration
from shinra.core.validation import require_operation
from shinra.tokenizer.serialization import TokenizerArtifact
from .provenance import Document


@dataclass(frozen=True)
class Segment:
    document_id: str
    source_id: str
    ordinal: int
    tokens: tuple[int, ...]


@dataclass(frozen=True)
class PackedBatch:
    token_ids: tuple[int, ...]
    positions: tuple[int, ...]
    segment_ids: tuple[int, ...]
    token_uids: tuple[int, ...]
    loss_mask: tuple[bool, ...]  # [T-1], i predicts i+1
    documents: tuple[tuple[str, str], ...]
    tokenizer_identity: str
    manifest_identity: str


class DocumentPacker:
    def __init__(self, config: Configuration, artifact: TokenizerArtifact):
        require_operation(config, "pack")
        if artifact.identity != config.get("tokenizer.artifact_identity") or not artifact.tokenizer.vocabulary.complete:
            raise ValueError("packing requires complete identity-matched native tokenizer")
        self.tokenizer = artifact.tokenizer
        self.tokenizer_identity = artifact.identity
        self.manifest_identity = config.get("data.manifest_identity")
        self.capacity = config.get("model.max_position_embeddings")
        registry = self.tokenizer.vocabulary.registry
        self.bos, self.eos, self.separator = (registry.token(n) for n in ("bos", "eos", "doc_sep"))

    def segment(self, document: Document, ordinal: int) -> Segment:
        if document.manifest_identity != self.manifest_identity:
            raise ValueError("document belongs to another corpus")
        if type(ordinal) is not int or ordinal < 0:
            raise ValueError("explicit nonnegative document ordinal required")
        tokens = (self.bos, *self.tokenizer.encode_bytes(document.payload), self.eos)
        if len(tokens) > self.capacity:
            raise ValueError("document exceeds context including BOS/EOS; approved split policy required")
        return Segment(document.record.document_id, document.record.source_id, ordinal, tokens)

    def pack(self, segments: tuple[Segment, ...]) -> PackedBatch:
        if not segments:
            raise ValueError("cannot produce empty packed batch")
        tokens, positions, identifiers, uids = [], [], [], []
        seen = set()
        for segment in segments:
            if segment.ordinal in seen:
                raise ValueError("duplicate document ordinal in packed batch")
            seen.add(segment.ordinal)
            if tokens:
                tokens.append(self.separator)
                positions.append(positions[-1] + 1)
                identifiers.append(identifiers[-1])
                uids.append(uids[-1] + 1)
            for position, token in enumerate(segment.tokens):
                uid = segment.ordinal * self.capacity + position
                if uid >= 2**63:
                    raise OverflowError("stable token UID exceeds int64")
                tokens.append(token)
                positions.append(position)
                identifiers.append(segment.ordinal)
                uids.append(uid)
        if len(tokens) > self.capacity or max(positions) >= self.capacity:
            raise ValueError("packed context exceeds architectural range")
        loss = tuple(identifiers[i] == identifiers[i+1] and tokens[i+1] != self.separator
                     for i in range(len(tokens)-1))
        return PackedBatch(tuple(tokens), tuple(positions), tuple(identifiers), tuple(uids), loss,
                           tuple((s.source_id, s.document_id) for s in segments),
                           self.tokenizer_identity, self.manifest_identity)
