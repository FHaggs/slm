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
    parser.add_argument("--skip-gsm8k", action="store_true", help="sem GSM8K (a geracao nao usa cache: so vale em GPU)")
    # Lotes de avaliacao: os padroes sao os do servidor (24 GB). Em GPU de 8 GB, reduzir (ver PLANO, Fase 5).
    parser.add_argument("--ppl-batch-size", type=int, default=16)
    parser.add_argument("--gsm-batch-size", type=int, default=32)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
    model, tokenizer = load_model(args.checkpoint, device)
    results: dict = {"checkpoint": args.checkpoint, "params": model.num_params()}

    def save() -> None:  # grava a cada etapa: uma queda nao perde o que ja foi medido
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results))

    if args.val_dir:
        results["perplexity"] = perplexity(model, args.val_dir, args.val_records, batch_size=args.ppl_batch_size)
        print("perplexidade:", results["perplexity"], flush=True)
        save()

    for name in args.benchmarks:
        result = run_multiple_choice(model, tokenizer, load_multiple_choice(name, args.mc_limit))
        results[name] = result
        print(f"{name}: {result['accuracy']:.4f} (media por token: {result['accuracy_mean']:.4f})", flush=True)
        save()

    arithmetic = run_arithmetic(model, tokenizer, arithmetic_items())
    results["arithmetic"] = arithmetic
    print(f"aritmetica: {arithmetic['accuracy']:.4f} {arithmetic['cells']}", flush=True)
    save()
    if args.skip_gsm8k:
        print(f"salvo em {args.output} (sem GSM8K)", flush=True)
        return

    train_rows = list(load_dataset("openai/gsm8k", "main", split="train").select(range(3)))
    test = load_dataset("openai/gsm8k", "main", split="test")
    test_rows = list(test.select(range(args.gsm_limit)) if args.gsm_limit else test)
    gsm = run_gsm8k(
        model,
        tokenizer,
        test_rows,
        gsm_few_shot_prefix(train_rows),
        batch_size=args.gsm_batch_size,
        max_new_tokens=args.gsm_new_tokens,
    )
    gsm["ci95"] = wilson_interval(sum(gsm["correct"]), len(gsm["correct"]))
    results["gsm8k"] = gsm
    print(f"gsm8k: {gsm['accuracy']:.4f} IC95 {gsm['ci95']} respondeu {gsm['answered']:.3f}", flush=True)

    save()
    print(f"salvo em {args.output}", flush=True)


if __name__ == "__main__":
    main()
