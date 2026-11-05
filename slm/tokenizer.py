"""BPE do GPT-2 (tiktoken) com tokens de chat e separacao opcional de digitos."""

import re
from concurrent.futures import ThreadPoolExecutor

import tiktoken

ENDOFTEXT = 50256  # separa documentos e abre conversas
USER = 50257
ASSISTANT = 50258
END = 50259
PAD = 50260
VOCAB_SIZE = 50261

_DIGIT = re.compile(r"(\d)")


class Tokenizer:
    def __init__(self, split_digits: bool = False):
        base = tiktoken.get_encoding("gpt2")
        self.split_digits = split_digits
        self.enc = tiktoken.Encoding(
            name="slm_gpt2",
            pat_str=base._pat_str,
            mergeable_ranks=base._mergeable_ranks,
            special_tokens={
                "<|endoftext|>": ENDOFTEXT,
                "<|user|>": USER,
                "<|assistant|>": ASSISTANT,
                "<|end|>": END,
                "<|pad|>": PAD,
            },
        )

    def encode(self, text: str) -> list[int]:
        """Texto nunca vira token especial: `<|end|>` digitado pelo usuario continua sendo texto."""
        if not self.split_digits:
            return self.enc.encode_ordinary(text)
        tokens: list[int] = []
        for piece in _DIGIT.split(text):
            if piece:
                tokens += self.enc.encode_ordinary(piece)
        return tokens

    def encode_batch(self, texts: list[str], num_threads: int = 8) -> list[list[int]]:
        if self.split_digits:
            with ThreadPoolExecutor(num_threads) as pool:
                return list(pool.map(self.encode, texts))
        return self.enc.encode_ordinary_batch(texts, num_threads=num_threads)

    def decode(self, tokens: list[int]) -> str:
        return self.enc.decode(tokens)
