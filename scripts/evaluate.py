"""Etapa 4: avalia um checkpoint em perplexidade, multipla escolha e matematica. Salva um JSON."""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset

from slm.checkpoint import load_model
from slm.data import RecordDataset, evenly_spaced_indices
from slm.evaluation import (
    arithmetic_items,
    gsm_few_shot_prefix,
    load_multiple_choice,
    run_arithmetic,
    run_gsm8k,
    run_multiple_choice,
    wilson_interval,
)


@torch.inference_mode()
def perplexity(model, directory: Path, max_records: int, batch_size: int) -> dict:
    dataset = RecordDataset(directory)
    device = next(model.parameters()).device
    indices = evenly_spaced_indices(len(dataset), max_records)
    loss_sum = 0.0
    count = 0
    for start in range(0, len(indices), batch_size):
        inputs, targets = dataset.batch(indices[start : start + batch_size])
        inputs = torch.from_numpy(inputs).to(device)
        targets = torch.from_numpy(targets).to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, _ = model(inputs)
        loss_sum += F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        ).item()
        count += int((targets != -100).sum())
    loss = loss_sum / count
    return {"tokens": count, "loss": loss, "perplexity": math.exp(loss)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, help="validacao do pre-treino, para a perplexidade")
    parser.add_argument("--val-records", type=int, default=5000)
    parser.add_argument("--benchmarks", nargs="*", default=["hellaswag", "arc_easy"])
    parser.add_argument("--mc-limit", type=int, help="limita os itens de multipla escolha")
    parser.add_argument("--gsm-limit", type=int, help="limita os itens do GSM8K")
    parser.add_argument("--gsm-new-tokens", type=int, default=160)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
    model, tokenizer = load_model(args.checkpoint, device)
    results: dict = {"checkpoint": args.checkpoint, "params": model.num_params()}

    if args.val_dir:
        results["perplexity"] = perplexity(model, args.val_dir, args.val_records, batch_size=16)
        print("perplexidade:", results["perplexity"], flush=True)

    for name in args.benchmarks:
        result = run_multiple_choice(model, tokenizer, load_multiple_choice(name, args.mc_limit))
        results[name] = result
        print(f"{name}: {result['accuracy']:.4f} (media por token: {result['accuracy_mean']:.4f})", flush=True)

    arithmetic = run_arithmetic(model, tokenizer, arithmetic_items())
    results["arithmetic"] = arithmetic
    print(f"aritmetica: {arithmetic['accuracy']:.4f} {arithmetic['cells']}", flush=True)

    train_rows = list(load_dataset("openai/gsm8k", "main", split="train").select(range(3)))
    test = load_dataset("openai/gsm8k", "main", split="test")
    test_rows = list(test.select(range(args.gsm_limit)) if args.gsm_limit else test)
    gsm = run_gsm8k(
        model,
        tokenizer,
        test_rows,
        gsm_few_shot_prefix(train_rows),
        max_new_tokens=args.gsm_new_tokens,
    )
    gsm["ci95"] = wilson_interval(sum(gsm["correct"]), len(gsm["correct"]))
    results["gsm8k"] = gsm
    print(f"gsm8k: {gsm['accuracy']:.4f} IC95 {gsm['ci95']} respondeu {gsm['answered']:.3f}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results))
    print(f"salvo em {args.output}", flush=True)


if __name__ == "__main__":
    main()
