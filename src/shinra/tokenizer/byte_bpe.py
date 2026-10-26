"""Native byte BPE encoding and exact corpus-driven pair-frequency training."""
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from shinra.core.configuration import Configuration
from shinra.core.validation import require_operation
from .special_tokens import SpecialRegistry, Special
from .vocabulary import Vocabulary
from .merges import Merge, MergeTable, replace_pair


class ByteBPE:
    def __init__(self, vocabulary: Vocabulary, merges: tuple[Merge, ...]):
        self.vocabulary = vocabulary
        self.table = MergeTable(vocabulary, merges)

    def encode_bytes(self, value: bytes) -> list[int]:
        if type(value) is not bytes:
            raise TypeError("encode_bytes requires bytes")
        return self.table.apply([self.vocabulary.byte_start + b for b in value])

    def encode_text(self, value: str) -> list[int]:
        if not isinstance(value, str):
            raise TypeError("encode_text requires str")
        return self.encode_bytes(value.encode("utf-8", errors="strict"))

    def decode_bytes(self, ids: Iterable[int]) -> bytes:
        return b"".join(self.vocabulary.payload(token) for token in ids)

    def decode_text(self, ids: Iterable[int]) -> str:
        return self.decode_bytes(ids).decode("utf-8", errors="strict")

    def encode_structured(self, fragments: Iterable[bytes | str | Special]) -> list[int]:
        result = []
        for fragment in fragments:
            if isinstance(fragment, Special):
                result.append(self.vocabulary.registry.token(fragment.name))
            elif isinstance(fragment, str):
                result.extend(self.encode_text(fragment))
            elif type(fragment) is bytes:
                result.extend(self.encode_bytes(fragment))
            else:
                raise TypeError("structured input requires bytes, str or Special")
        return result

    def decode_structured(self, ids: Iterable[int]) -> tuple[bytes | Special, ...]:
        result, content = [], bytearray()
        for token in ids:
            if type(token) is not int:
                raise TypeError("token IDs must be integers")
            if 0 <= token < self.vocabulary.byte_start:
                if content:
                    result.append(bytes(content))
                    content.clear()
                result.append(Special(self.vocabulary.registry.name(token)))
            else:
                content.extend(self.vocabulary.payload(token))
        if content:
            result.append(bytes(content))
        return tuple(result)

    def encode_fim(self, prefix: str, suffix: str, middle: str) -> list[int]:
        return self.encode_structured((Special("bos"), Special("fim_prefix"), prefix,
                                      Special("fim_suffix"), suffix, Special("fim_middle"), middle,
                                      Special("fim_end"), Special("eos")))


@dataclass(frozen=True)
class TrainingResult:
    tokenizer: ByteBPE
    corpus_identity: str
    corpus_approval: str
    fragments: int
    input_bytes: int
    token_frequencies: tuple[tuple[int, int], ...]


def train_bpe(config: Configuration, fragments: Iterable[bytes], *, corpus_identity: str) -> TrainingResult:
    """Exact D05 training on approved, deterministic fragment order.

    Stores weighted sequences in memory and recounts pairs each merge. No
    scalability/throughput claim: insufficient memory propagates as an error,
    never a smaller vocabulary or substituted corpus. Inputs must be finalized
    document/content fragments supplied by the approved ingestion pipeline.
    """
    require_operation(config, "tokenizer_train")
    if corpus_identity != config.get("data.manifest_identity"):
        raise ValueError("training corpus identity mismatch")
    registry = SpecialRegistry.from_configuration(config)
    start = config.get("tokenizer.byte_start_id")
    limit = config.get("tokenizer.max_token_bytes")
    minimum = config.get("tokenizer.min_pair_frequency")
    capacity = config.get("model.vocab_size")
    payloads = [bytes([b]) for b in range(256)]
    payload_set = set(payloads)
    sequences = Counter()
    count, byte_count = 0, 0
    for fragment in fragments:
        if type(fragment) is not bytes:
            raise TypeError("training input must contain byte fragments")
        fragment.decode("utf-8", errors="strict")
        sequences[tuple(start+b for b in fragment)] += 1
        count += 1
        byte_count += len(fragment)
    if count == 0 or byte_count == 0:
        raise ValueError("approved corpus contains no text bytes")
    merges = []
    while start + len(payloads) < capacity:
        pairs = Counter()
        for sequence, frequency in sequences.items():
            for pair in zip(sequence, sequence[1:]):
                pairs[pair] += frequency
        candidates = []
        for pair, frequency in pairs.items():
            if frequency < minimum:
                continue
            payload = payloads[pair[0]-start] + payloads[pair[1]-start]
            if len(payload) <= limit and payload not in payload_set:
                candidates.append((-frequency, pair))
        if not candidates:
            raise ValueError(f"tokenizer admission BLOCKED: only {start+len(payloads)} of {capacity} slots supported by corpus")
        _, pair = min(candidates)
        output = start + len(payloads)
        payload = payloads[pair[0]-start] + payloads[pair[1]-start]
        payloads.append(payload)
        payload_set.add(payload)
        merges.append(Merge(*pair, output))
        updated = Counter()
        for sequence, frequency in sequences.items():
            updated[replace_pair(sequence, pair, output)] += frequency
        sequences = updated
    vocabulary = Vocabulary(registry, tuple(payloads), capacity, start, limit)
    tokenizer = ByteBPE(vocabulary, tuple(merges))
    frequencies = Counter()
    for sequence, multiplicity in sequences.items():
        for token in sequence:
            frequencies[token] += multiplicity
    return TrainingResult(tokenizer, corpus_identity, config.get("data.approval_reference"),
                          count, byte_count, tuple(sorted(frequencies.items())))
