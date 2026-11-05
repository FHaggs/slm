"""Template de chat e empacotamento de conversas em registros de tamanho fixo.

Formato: <|endoftext|> <|user|> texto <|end|> <|assistant|> texto <|end|> ...
A mascara vale 1 nos tokens que o SFT deve aprender: as respostas e o <|end|> que as encerra.
"""

from collections.abc import Iterable

import numpy as np

from slm.tokenizer import ASSISTANT, END, ENDOFTEXT, PAD, USER, Tokenizer

Message = dict[str, str]
Encoded = tuple[np.ndarray, np.ndarray]


def merge_system(messages: list[Message]) -> list[Message]:
    """O template nao tem papel de sistema: o texto dele entra no inicio da primeira fala do usuario."""
    merged: list[Message] = []
    pending_system: list[str] = []
    for message in messages:
        if message["role"] == "system":
            pending_system.append(message["content"])
        elif message["role"] == "user" and pending_system:
            content = "\n\n".join([*pending_system, message["content"]])
            merged.append({"role": "user", "content": content})
            pending_system = []
        else:
            merged.append({"role": message["role"], "content": message["content"]})
    return merged


def _assemble(messages: list[Message], encoded: list[list[int]]) -> Encoded:
    tokens = [ENDOFTEXT]
    mask = [0]
    for message, ids in zip(messages, encoded):
        if message["role"] == "user":
            tokens += [USER, *ids, END]
            mask += [0] * (len(ids) + 2)
        elif message["role"] == "assistant":
            tokens += [ASSISTANT, *ids, END]
            mask += [0, *[1] * (len(ids) + 1)]
        else:
            raise ValueError(f"papel desconhecido: {message['role']}")
    return np.array(tokens, dtype=np.uint16), np.array(mask, dtype=np.uint8)


def render_batch(conversations: list[list[Message]], tokenizer: Tokenizer) -> list[Encoded]:
    merged = [merge_system(messages) for messages in conversations]
    texts = [message["content"] for messages in merged for message in messages]
    encoded = tokenizer.encode_batch(texts)
    rendered = []
    position = 0
    for messages in merged:
        rendered.append(_assemble(messages, encoded[position : position + len(messages)]))
        position += len(messages)
    return rendered


def render(messages: list[Message], tokenizer: Tokenizer) -> Encoded:
    return render_batch([messages], tokenizer)[0]


def render_prompt(messages: list[Message], tokenizer: Tokenizer) -> list[int]:
    """Tokens que o modelo recebe para escrever a proxima resposta do assistente."""
    tokens, _ = render(messages, tokenizer)
    return [*tokens.tolist(), ASSISTANT]


def pack(conversations: Iterable[Encoded], record_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Junta conversas inteiras em registros de `record_len` tokens, completando o fim com <|pad|>.

    Conversas maiores que um registro sao cortadas. Pedacos sem nenhuma resposta sao descartados,
    pois nao teriam o que treinar no SFT.
    """
    records: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    tokens = np.full(record_len, PAD, dtype=np.uint16)
    mask = np.zeros(record_len, dtype=np.uint8)
    used = 0

    def flush() -> None:
        nonlocal tokens, mask, used
        if used:
            records.append(tokens)
            masks.append(mask)
        tokens = np.full(record_len, PAD, dtype=np.uint16)
        mask = np.zeros(record_len, dtype=np.uint8)
        used = 0

    for conv_tokens, conv_mask in conversations:
        conv_tokens, conv_mask = conv_tokens[:record_len], conv_mask[:record_len]
        if not conv_mask.any():
            continue
        if used + len(conv_tokens) > record_len:
            flush()
        tokens[used : used + len(conv_tokens)] = conv_tokens
        mask[used : used + len(conv_tokens)] = conv_mask
        used += len(conv_tokens)
    flush()

    if not records:
        return (
            np.empty((0, record_len), dtype=np.uint16),
            np.empty((0, record_len), dtype=np.uint8),
        )
    return np.stack(records), np.stack(masks)
