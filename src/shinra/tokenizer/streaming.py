"""Chunk-independent fragment encoding and strict incremental UTF-8 decoding."""
import codecs
from .byte_bpe import ByteBPE


class FragmentEncoder:
    """Buffer until explicit fragment boundary; never merge across documents."""
    def __init__(self, tokenizer: ByteBPE):
        self.tokenizer = tokenizer
        self.buffer = bytearray()

    def feed(self, chunk: bytes) -> None:
        if type(chunk) is not bytes:
            raise TypeError("stream chunks must be bytes")
        self.buffer.extend(chunk)

    def finish(self) -> list[int]:
        payload = bytes(self.buffer)
        payload.decode("utf-8", errors="strict")
        encoded = self.tokenizer.encode_bytes(payload)
        self.buffer.clear()
        return encoded


class TextDecoder:
    def __init__(self, tokenizer: ByteBPE):
        self.tokenizer = tokenizer
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
        self.closed = False

    def feed(self, token_id: int) -> str:
        if self.closed:
            raise RuntimeError("decoder is closed")
        return self.decoder.decode(self.tokenizer.vocabulary.payload(token_id), final=False)

    def finish(self) -> str:
        if self.closed:
            raise RuntimeError("decoder is closed")
        tail = self.decoder.decode(b"", final=True)
        self.closed = True
        return tail
