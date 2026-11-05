"""Mede tokens/s e memoria de GPU por tamanho de modelo e de batch, com tokens aleatorios.

O resultado e um limite superior: nao inclui leitura de dados, avaliacao nem checkpoints.

1 GPU:   python scripts/bench.py --preset final --batch-sizes 32 64 96 128
2 GPUs:  torchrun --standalone --nproc_per_node=2 scripts/bench.py --preset final --batch-sizes 64
"""

import argparse
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from slm.model import SLM, ModelConfig

PRESETS = {
    "final": ModelConfig(d_model=768, n_layer=12, n_head=12, n_kv_head=4, ffn_hidden=2048),
    "plan_b": ModelConfig(d_model=640, n_layer=12, n_head=8, n_kv_head=2, ffn_hidden=1728),
    "pilot": ModelConfig(d_model=384, n_layer=8, n_head=6, n_kv_head=2, ffn_hidden=1024),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=PRESETS, default="final")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[64])
    parser.add_argument("--steps", type=int, default=30, help="steps medidos por batch size")
    parser.add_argument("--warmup-steps", type=int, default=10, help="steps descartados")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/bench.jsonl"))
    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    if world_size > 1:
        dist.init_process_group("nccl")
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")

    cfg = PRESETS[args.preset]
    torch.manual_seed(0)
    model = SLM(cfg).to(device)
    num_params = model.num_params()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1, fused=True)
    net = model if args.no_compile else torch.compile(model)
    if world_size > 1:
        net = DistributedDataParallel(net, device_ids=[device.index])

    if rank == 0:
        print(
            f"torch {torch.__version__} | {torch.cuda.get_device_name(device)} | "
            f"bf16={torch.cuda.is_bf16_supported()} | flash={torch.backends.cuda.flash_sdp_enabled()} | "
            f"preset={args.preset} | params={num_params / 1e6:.1f}M | gpus={world_size}",
            flush=True,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)

    for batch_size in args.batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        record = {
            "preset": args.preset,
            "params": num_params,
            "gpus": world_size,
            "batch_size": batch_size,
            "seq_len": cfg.max_seq_len,
            "compiled": not args.no_compile,
            "model": asdict(cfg),
            "gpu_name": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
        }
        try:
            first_loss = None
            started = None
            for step in range(args.warmup_steps + args.steps):
                data = torch.randint(
                    0, cfg.vocab_size, (batch_size, cfg.max_seq_len + 1), device=device
                )
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, loss = net(data[:, :-1], data[:, 1:])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if first_loss is None:
                    first_loss = loss.item()
                if step == args.warmup_steps - 1:
                    torch.cuda.synchronize(device)
                    started = time.perf_counter()
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            tokens = args.steps * batch_size * cfg.max_seq_len * world_size
            record.update(
                status="ok",
                first_loss=first_loss,
                expected_first_loss=math.log(cfg.vocab_size),
                tokens_per_second=tokens / elapsed,
                seconds_per_step=elapsed / args.steps,
                peak_gpu_gb=torch.cuda.max_memory_allocated(device) / 1e9,
            )
        except torch.OutOfMemoryError:
            optimizer.zero_grad(set_to_none=True)
            record.update(status="oom")

        if rank == 0:
            with args.output.open("a") as file:
                file.write(json.dumps(record) + "\n")
            if record["status"] == "ok":
                print(
                    f"batch={batch_size:4d} | {record['tokens_per_second']:9,.0f} tok/s | "
                    f"{record['seconds_per_step']:.3f} s/step | "
                    f"pico {record['peak_gpu_gb']:.1f} GB | loss inicial {first_loss:.2f}",
                    flush=True,
                )
            else:
                print(f"batch={batch_size:4d} | sem memoria", flush=True)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
