"""Canonical UTF-8 JSON tokenizer bundle; no foreign assets or partial vocabulary."""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

from shinra.core.configuration import Configuration
from shinra.core.validation import require_operation
from .byte_bpe import ByteBPE, TrainingResult
from .special_tokens import SpecialRegistry
from .vocabulary import Vocabulary
from .merges import Merge


def canonical_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8", errors="strict")


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate JSON field {key}")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"nonfinite JSON number {value}")


def read_json_bytes(payload: bytes):
    return json.loads(payload.decode("utf-8", errors="strict"), object_pairs_hook=_pairs, parse_constant=_constant)


def exact_keys(value, keys):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"expected exactly fields {sorted(keys)}")


@dataclass(frozen=True)
class TokenizerArtifact:
    tokenizer: ByteBPE
    identity: str
    corpus_identity: str


def save_artifact(path: str | Path, result: TrainingResult, config: Configuration) -> str:
    require_operation(config, "tokenizer_train")
    tokenizer = result.tokenizer
    vocab = tokenizer.vocabulary
    if not vocab.complete or result.corpus_identity != config.get("data.manifest_identity"):
        raise ValueError("cannot publish incomplete vocabulary or foreign corpus identity")
    if vocab.registry != SpecialRegistry.from_configuration(config):
        raise ValueError("registry differs from approved configuration")
    if vocab.max_token_bytes != config.get("tokenizer.max_token_bytes"):
        raise ValueError("vocabulary byte limit differs from configuration")
    body = {
        "format": "shinra.bytebpe.v1", "vocab_size": vocab.capacity,
        "registry": dict(vocab.registry.ids), "registry_approval": vocab.registry.approval_reference,
        "byte_start": vocab.byte_start, "max_token_bytes": vocab.max_token_bytes,
        "payload_hex": [payload.hex() for payload in vocab.payloads],
        "merges": [[m.left, m.right, m.output] for m in tokenizer.table.merges],
        "provenance": {"corpus_identity": result.corpus_identity, "corpus_approval": result.corpus_approval,
                       "min_pair_frequency": config.get("tokenizer.min_pair_frequency")},
        "statistics": {"fragments": result.fragments, "input_bytes": result.input_bytes,
                       "token_frequencies": [list(pair) for pair in result.token_frequencies]},
    }
    payload = canonical_bytes(body)
    identity = hashlib.sha256(payload).hexdigest()
    # Exclusive creation: a failure leaves an invalid artifact rejected by hash.
    # No automatic publication or config mutation follows this write.
    with Path(path).open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return identity


def load_artifact(path: str | Path, config: Configuration) -> TokenizerArtifact:
    require_operation(config, "tokenizer")
    expected = config.get("tokenizer.artifact_identity")
    if expected is None:
        raise ValueError("tokenizer artifact identity is UNKNOWN")
    payload = Path(path).read_bytes()
    identity = hashlib.sha256(payload).hexdigest()
    if identity != expected:
        raise ValueError("tokenizer artifact identity mismatch")
    body = read_json_bytes(payload)
    exact_keys(body, ("format", "vocab_size", "registry", "registry_approval", "byte_start",
                      "max_token_bytes", "payload_hex", "merges", "provenance", "statistics"))
    if canonical_bytes(body) != payload or body["format"] != "shinra.bytebpe.v1":
        raise ValueError("unsupported or noncanonical tokenizer bundle")
    registry = SpecialRegistry(body["registry"], body["registry_approval"])
    if registry != SpecialRegistry.from_configuration(config):
        raise ValueError("artifact registry is not the approved registry")
    if body["max_token_bytes"] != config.get("tokenizer.max_token_bytes"):
        raise ValueError("artifact token byte limit mismatch")
    if not isinstance(body["payload_hex"], list) or any(not isinstance(x, str) for x in body["payload_hex"]):
        raise ValueError("payload_hex must be a list of hex strings")
    payloads = tuple(bytes.fromhex(x) for x in body["payload_hex"])
    if [x.hex() for x in payloads] != body["payload_hex"]:
        raise ValueError("noncanonical byte payload hex")
    vocab = Vocabulary(registry, payloads, body["vocab_size"], body["byte_start"], body["max_token_bytes"])
    if not vocab.complete:
        raise ValueError("trained vocabulary must fill exactly 131072 slots")
    if not isinstance(body["merges"], list) or any(not isinstance(x, list) or len(x) != 3 for x in body["merges"]):
        raise ValueError("invalid merge serialization")
    tokenizer = ByteBPE(vocab, tuple(Merge(*x) for x in body["merges"]))
    provenance = body["provenance"]
    exact_keys(provenance, ("corpus_identity", "corpus_approval", "min_pair_frequency"))
    corpus = provenance["corpus_identity"]
    if not isinstance(corpus, str) or len(corpus) != 64 or any(c not in "0123456789abcdef" for c in corpus):
        raise ValueError("invalid training corpus identity")
    if not isinstance(provenance["corpus_approval"], str) or not provenance["corpus_approval"].strip():
        raise ValueError("missing training corpus approval")
    if provenance["min_pair_frequency"] != config.get("tokenizer.min_pair_frequency"):
        raise ValueError("training merge threshold mismatch")
    stats = body["statistics"]
    exact_keys(stats, ("fragments", "input_bytes", "token_frequencies"))
    if any(type(stats[k]) is not int or stats[k] <= 0 for k in ("fragments", "input_bytes")):
        raise ValueError("invalid measured corpus counts")
    frequencies = stats["token_frequencies"]
    if not isinstance(frequencies, list):
        raise ValueError("invalid token frequency table")
    previous, decoded_bytes = -1, 0
    for entry in frequencies:
        if not isinstance(entry, list) or len(entry) != 2 or any(type(x) is not int for x in entry):
            raise ValueError("frequency entries require token ID and integer count")
        token, count = entry
        if token <= previous or count <= 0:
            raise ValueError("token frequencies must be ordered, unique and positive")
        decoded_bytes += len(vocab.payload(token)) * count
        previous = token
    if decoded_bytes != stats["input_bytes"]:
        raise ValueError("token frequencies violate byte conservation")
    return TokenizerArtifact(tokenizer, identity, corpus)
