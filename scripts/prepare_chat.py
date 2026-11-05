"""Gera os dados de chat do mid-training e do SFT, com ou sem matematica.

A parte de chat usa o SmolTalk sem as configs de matematica. A parte de matematica usa GSM8K treino e
MetaMathQA, sem nenhuma pergunta do GSM8K teste. Com a mesma seed, o chat de um conjunto com matematica
e um prefixo do chat do conjunto sem matematica, entao a unica diferenca entre eles e a matematica.
"""

import argparse
import random
import re
from pathlib import Path

import numpy as np
from datasets import load_dataset

from slm.chat import Encoded, Message, pack, render_batch
from slm.data import write_records
from slm.evaluation import clean_gsm_solution
from slm.tokenizer import PAD, Tokenizer

SMOLTALK = "HuggingFaceTB/smoltalk"
CHAT_CONFIGS = [
    "apigen-80k",
    "everyday-conversations",
    "explore-instruct-rewriting",
    "longalign",
    "openhermes-100k",
    "self-oss-instruct",
    "smol-constraints",
    "smol-magpie-ultra",
    "smol-rewrite",
    "smol-summarize",
    "systemchats-30k",
]
GSM8K_REPEATS = 4
_MATH_RESIDUE = re.compile(r"\\boxed|####")
_NOT_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")


def normalize(text: str) -> str:
    return _NOT_ALPHANUMERIC.sub(" ", text.lower()).strip()


def load_chat(split: str, limit: int | None) -> tuple[list[list[Message]], int]:
    conversations = []
    removed = 0
    for config in CHAT_CONFIGS:
        dataset = load_dataset(SMOLTALK, config, split=split)
        if limit:
            dataset = dataset.select(range(min(limit, len(dataset))))
        for messages in dataset["messages"]:
            if any(_MATH_RESIDUE.search(message["content"]) for message in messages):
                removed += 1
            else:
                conversations.append(messages)
    return conversations, removed


def load_math(limit: int | None) -> tuple[list[list[Message]], int]:
    """Retorna as conversas de matematica e quantas foram removidas por vazamento do GSM8K teste."""
    forbidden = {normalize(row["question"]) for row in load_dataset("openai/gsm8k", "main", split="test")}
    conversations: list[list[Message]] = []
    leaked = 0

    def add(question: str, answer: str, repeats: int = 1, *others: str) -> None:
        nonlocal leaked
        if any(normalize(text) in forbidden for text in (question, *others) if text):
            leaked += 1
            return
        conversation = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        conversations.extend([conversation] * repeats)

    gsm_train = load_dataset("openai/gsm8k", "main", split="train")
    metamath = load_dataset("meta-math/MetaMathQA", split="train")
    if limit:
        gsm_train = gsm_train.select(range(min(limit, len(gsm_train))))
        metamath = metamath.select(range(min(limit, len(metamath))))
    for row in gsm_train:
        add(row["question"], clean_gsm_solution(row["answer"]), GSM8K_REPEATS)
    for row in metamath:
        add(row["query"], row["response"], 1, row["original_question"])
    return conversations, leaked


def take_tokens(
    conversations: list[list[Message]], budget: int, tokenizer: Tokenizer, record_len: int
) -> tuple[list[Encoded], int]:
    """Codifica conversas, em ordem, ate atingir o orcamento de tokens."""
    taken: list[Encoded] = []
    total = 0
    for start in range(0, len(conversations), 4096):
        for encoded in render_batch(conversations[start : start + 4096], tokenizer):
            taken.append(encoded)
            total += min(len(encoded[0]), record_len)
            if total >= budget:
                return taken, total
    return taken, total


def build(
    conversations: list[Encoded], directory: Path, record_len: int, seed: int, **meta
) -> None:
    conversations = list(conversations)
    random.Random(seed).shuffle(conversations)
    tokens, mask = pack(conversations, record_len)
    meta["fill_rate"] = float(np.count_nonzero(tokens != PAD) / max(tokens.size, 1))
    write_records(directory, tokens, mask, **meta)
    print(f"{directory}: {tokens.shape[0]:,} registros, {tokens.size:,} tokens, {meta}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--math-frac", type=float, default=0.0)
    parser.add_argument("--val-tokens", type=int, default=5_000_000)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--limit-per-source", type=int, help="menos dados, para testes rapidos")
    parser.add_argument("--split-digits", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    tokenizer = Tokenizer(split_digits=args.split_digits)
    record_len = args.seq_len + 1
    limit = args.limit_per_source

    chat, removed = load_chat("train", limit)
    random.Random(args.seed).shuffle(chat)
    math_budget = int(args.max_tokens * args.math_frac)
    chat_encoded, chat_tokens = take_tokens(chat, args.max_tokens - math_budget, tokenizer, record_len)

    math_encoded: list[Encoded] = []
    math_tokens = 0
    leaked = 0
    if math_budget:
        math_rows, leaked = load_math(limit)
        random.Random(args.seed + 1).shuffle(math_rows)
        math_encoded, math_tokens = take_tokens(math_rows, math_budget, tokenizer, record_len)

    print(f"chat: {chat_tokens:,} tokens; matematica: {math_tokens:,} tokens", flush=True)
    build(
        chat_encoded + math_encoded,
        args.out / "train",
        record_len,
        args.seed,
        split_digits=args.split_digits,
        chat_tokens=chat_tokens,
        math_tokens=math_tokens,
        removed_math_residue=removed,
        removed_gsm_test_leak=leaked,
    )

    val_chat, _ = load_chat("test", limit)
    random.Random(args.seed).shuffle(val_chat)
    val_encoded, val_tokens = take_tokens(val_chat, args.val_tokens, tokenizer, record_len)
    build(
        val_encoded,
        args.out / "val",
        record_len,
        args.seed,
        split_digits=args.split_digits,
        chat_tokens=val_tokens,
    )


if __name__ == "__main__":
    main()
