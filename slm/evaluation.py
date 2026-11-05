"""Benchmarks: matematica (aritmetica e GSM8K) e multipla escolha por log-likelihood."""

import math
import random
import re
from collections import defaultdict

import torch
import torch.nn.functional as F

from slm.generate import greedy_batch
from slm.model import SLM
from slm.tokenizer import ENDOFTEXT, Tokenizer

_CALCULATOR = re.compile(r"<<[^>]*>>")
_LEADING_NUMBER = re.compile(r"\s*(-?\d[\d,]*(?:\.\d+)?)")
_ANY_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_MARKED_ANSWER = re.compile(r"####\s*(-?\d[\d,]*(?:\.\d+)?)")


def clean_gsm_solution(answer: str) -> str:
    """Remove as anotacoes de calculadora: `48/2 = <<48/2=24>>24` vira `48/2 = 24`."""
    return _CALCULATOR.sub("", answer)


def parse_number(text: str | None) -> float | None:
    if text is None:
        return None
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def leading_number(text: str) -> float | None:
    match = _LEADING_NUMBER.match(text)
    return parse_number(match.group(1)) if match else None


def gsm_answer(text: str) -> float | None:
    """Resposta apos `####`; sem marcador, o ultimo numero do trecho antes da linha em branco."""
    text = text.split("\n\n")[0]
    marked = _MARKED_ANSWER.search(text)
    if marked:
        return parse_number(marked.group(1))
    numbers = _ANY_NUMBER.findall(text)
    return parse_number(numbers[-1]) if numbers else None


def same_number(prediction: float | None, reference: float | None) -> bool:
    if prediction is None or reference is None:
        return False
    return math.isclose(prediction, reference, abs_tol=1e-6)


def arithmetic_items(seed: int = 1234, per_cell: int = 300) -> list[dict]:
    """Soma, subtracao e multiplicacao. Um digito: todas as combinacoes; dois digitos: amostra."""
    rng = random.Random(seed)
    items = []
    operations = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b}
    for symbol, compute in operations.items():
        one_digit = [(a, b) for a in range(1, 10) for b in range(1, 10) if symbol != "-" or a >= b]
        two_digits = [(a, b) for a in range(10, 100) for b in range(10, 100) if symbol != "-" or a >= b]
        for digits, pairs in ((1, one_digit), (2, rng.sample(two_digits, per_cell))):
            for a, b in pairs:
                items.append(
                    {"cell": f"{symbol}_{digits}d", "prompt": f"{a} {symbol} {b} =", "answer": compute(a, b)}
                )
    return items


def _greedy_by_length(
    model: SLM, tokenizer: Tokenizer, prompts: list[list[int]], max_new_tokens: int, batch_size: int
) -> list[str]:
    """Prompts de mesmo tamanho vao juntos, pois nao ha mascara de padding na geracao."""
    device = next(model.parameters()).device
    by_length: dict[int, list[int]] = defaultdict(list)
    for index, prompt in enumerate(prompts):
        by_length[len(prompt)].append(index)
    outputs = [""] * len(prompts)
    for indices in by_length.values():
        for start in range(0, len(indices), batch_size):
            chunk = indices[start : start + batch_size]
            batch = torch.tensor([prompts[i] for i in chunk], device=device)
            for index, tokens in zip(chunk, greedy_batch(model, batch, max_new_tokens).tolist()):
                outputs[index] = tokenizer.decode(tokens)
    return outputs


def run_arithmetic(
    model: SLM, tokenizer: Tokenizer, items: list[dict], batch_size: int = 256, max_new_tokens: int = 8
) -> dict:
    prompts = [[ENDOFTEXT, *tokenizer.encode(item["prompt"])] for item in items]
    outputs = _greedy_by_length(model, tokenizer, prompts, max_new_tokens, batch_size)
    correct = [same_number(leading_number(o), float(i["answer"])) for o, i in zip(outputs, items)]
    cells: dict[str, list[bool]] = defaultdict(list)
    for item, is_correct in zip(items, correct):
        cells[item["cell"]].append(is_correct)
    return {
        "accuracy": sum(correct) / len(correct),
        "cells": {cell: sum(values) / len(values) for cell, values in cells.items()},
        "correct": correct,
    }


def gsm_few_shot_prefix(train_rows: list[dict], shots: int = 3) -> str:
    return "".join(
        f"Question: {row['question']}\nAnswer: {clean_gsm_solution(row['answer'])}\n\n"
        for row in train_rows[:shots]
    )


def run_gsm8k(
    model: SLM,
    tokenizer: Tokenizer,
    test_rows: list[dict],
    prefix: str,
    batch_size: int = 32,
    max_new_tokens: int = 160,
) -> dict:
    prompts = [
        [ENDOFTEXT, *tokenizer.encode(f"{prefix}Question: {row['question']}\nAnswer:")]
        for row in test_rows
    ]
    outputs = _greedy_by_length(model, tokenizer, prompts, max_new_tokens, batch_size)
    predictions = [gsm_answer(output) for output in outputs]
    correct = [
        same_number(prediction, gsm_answer(row["answer"]))
        for prediction, row in zip(predictions, test_rows)
    ]
    return {
        "accuracy": sum(correct) / len(correct),
        "answered": sum(p is not None for p in predictions) / len(predictions),
        "correct": correct,
        "samples": outputs[:5],
    }


def mcnemar_exact(first: list[bool], second: list[bool]) -> dict:
    """Teste pareado: so os itens em que os dois modelos discordam contam."""
    if len(first) != len(second):
        raise ValueError("os resultados precisam cobrir os mesmos itens")
    only_first = sum(a and not b for a, b in zip(first, second))
    only_second = sum(b and not a for a, b in zip(first, second))
    total = only_first + only_second
    tail = sum(math.comb(total, k) for k in range(min(only_first, only_second) + 1))
    p_value = min(1.0, 2 * tail / 2**total) if total else 1.0
    return {"only_first": only_first, "only_second": only_second, "p_value": p_value}


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """IC de 95% para uma proporcao, adequado tambem a acertos raros."""
    if total == 0:
        return 0.0, 1.0
    rate = successes / total
    denominator = 1 + z**2 / total
    center = (rate + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / total + z**2 / (4 * total**2)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


@torch.inference_mode()
def choice_log_likelihoods(
    model: SLM, context: list[int], endings: list[list[int]]
) -> list[tuple[float, float]]:
    """Para cada alternativa, a soma e a media da log-probabilidade de seus tokens dado o contexto."""
    device = next(model.parameters()).device
    limit = model.cfg.max_seq_len + 1
    rows = []
    starts = []
    for ending in endings:
        ending = ending[: limit - 1]
        kept_context = context[-(limit - len(ending)) :]
        rows.append(kept_context + ending)
        starts.append(len(kept_context))
    width = max(len(row) for row in rows)
    tokens = torch.zeros(len(rows), width, dtype=torch.long, device=device)
    for i, row in enumerate(rows):
        tokens[i, : len(row)] = torch.tensor(row, device=device)
    enabled = device.type == "cuda"
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=enabled):
        logits, _ = model(tokens[:, :-1])
    log_probs = F.log_softmax(logits.float(), dim=-1)
    scores = []
    for i, (row, start) in enumerate(zip(rows, starts)):
        positions = torch.arange(start - 1, len(row) - 1, device=device)
        chosen = log_probs[i, positions, tokens[i, positions + 1]]
        scores.append((chosen.sum().item(), chosen.mean().item()))
    return scores


def run_multiple_choice(model: SLM, tokenizer: Tokenizer, items: list[dict]) -> dict:
    """Cada item tem context, endings e label. `accuracy_mean` normaliza pelo tamanho da alternativa."""
    correct_sum = []
    correct_mean = []
    for item in items:
        context = [ENDOFTEXT, *tokenizer.encode(item["context"])]
        endings = [tokenizer.encode(ending) for ending in item["endings"]]
        scores = choice_log_likelihoods(model, context, endings)
        sums = [score[0] for score in scores]
        means = [score[1] for score in scores]
        correct_sum.append(sums.index(max(sums)) == item["label"])
        correct_mean.append(means.index(max(means)) == item["label"])
    return {
        "items": len(items),
        "accuracy": sum(correct_sum) / len(correct_sum),
        "accuracy_mean": sum(correct_mean) / len(correct_mean),
        "correct": correct_sum,
    }


def load_multiple_choice(name: str, limit: int | None = None) -> list[dict]:
    from datasets import load_dataset

    items = []
    if name == "hellaswag":
        for row in load_dataset("allenai/hellaswag", split="validation"):
            items.append(
                {
                    "context": row["ctx"],
                    "endings": [f" {ending}" for ending in row["endings"]],
                    "label": int(row["label"]),
                }
            )
    elif name in ("arc_easy", "arc_challenge"):
        config = {"arc_easy": "ARC-Easy", "arc_challenge": "ARC-Challenge"}[name]
        for row in load_dataset("allenai/ai2_arc", config, split="test"):
            items.append(
                {
                    "context": f"Question: {row['question']}\nAnswer:",
                    "endings": [f" {text}" for text in row["choices"]["text"]],
                    "label": row["choices"]["label"].index(row["answerKey"]),
                }
            )
    else:
        raise ValueError(f"benchmark desconhecido: {name}")
    return items[:limit] if limit else items
