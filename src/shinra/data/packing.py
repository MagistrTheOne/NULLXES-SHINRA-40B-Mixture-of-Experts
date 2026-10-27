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


@dataclass(frozen=True)
class ContextBatch:
    token_ids: tuple[int, ...]
    positions: tuple[int, ...]
    segment_ids: tuple[int, ...]
    token_uids: tuple[int, ...]
    target_indices: tuple[int, ...]
    target_ids: tuple[int, ...]
    tokenizer_identity: str
    manifest_identity: str
    global_start: int
    global_stop: int
    global_tokens: int


def partition_context_batch(batch: PackedBatch, offsets: tuple[int, ...], rank: int) -> ContextBatch:
    """Assign prediction positions, including the final real target across a cut.

    Offsets are explicit logical boundaries, not inferred hardware topology.
    The existing packed loss mask owns document/EOS/separator semantics. No
    fabricated target is inserted for the last token or a masked boundary.
    """
    size = len(batch.token_ids)
    if (type(rank) is not int or not 0 <= rank < len(offsets)-1 or
            any(type(n) is not int for n in offsets) or offsets[0] != 0 or offsets[-1] != size or
            any(a >= b for a,b in zip(offsets, offsets[1:]))):
        raise ValueError("explicit nonempty contiguous context partitions must cover packed tokens")
    if size > 262144 or any(len(v) != size for v in (batch.positions, batch.segment_ids, batch.token_uids)) or len(batch.loss_mask) != max(size-1, 0):
        raise ValueError("invalid packed context shapes")
    if any(type(v) is not bool for v in batch.loss_mask):
        raise ValueError("loss eligibility must be explicit booleans")
    start, stop = offsets[rank:rank+2]
    selected = [i for i in range(start, min(stop, size-1)) if batch.loss_mask[i]]
    if any(batch.segment_ids[i] != batch.segment_ids[i+1] for i in selected):
        raise ValueError("cross-document loss target is forbidden")
    return ContextBatch(batch.token_ids[start:stop], batch.positions[start:stop],
                        batch.segment_ids[start:stop], batch.token_uids[start:stop],
                        tuple(i-start for i in selected), tuple(batch.token_ids[i+1] for i in selected),
                        batch.tokenizer_identity, batch.manifest_identity, start, stop, size)
