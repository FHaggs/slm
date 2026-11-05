"""Laco de treino comum as tres etapas: DDP, bf16, WSD, grad clip, checkpoints e retomada."""

import json
import math
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel

from slm import checkpoint
from slm.data import EpochSampler, RecordDataset, evenly_spaced_indices
from slm.model import SLM, ModelConfig
from slm.schedule import wsd_lr

EvalHook = Callable[[SLM], dict[str, float]]


@dataclass
class TrainConfig:
    run_dir: str
    data_dir: str
    val_dir: str | None = None
    init_from: str | None = None  # so os pesos: otimizador zerado (mid-training e SFT)
    use_loss_mask: bool = False
    max_tokens: int | None = None
    epochs: float | None = None  # alternativa a max_tokens
    batch_size: int = 24  # por GPU
    peak_lr: float = 3e-4
    warmup_tokens: int = 100_000_000
    decay_frac: float = 0.1
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    log_every: int = 50
    eval_every: int = 1000
    eval_batches: int = 100
    eval_batch_size: int = 16
    ckpt_every: int = 2000
    keep_last: int = 2
    compile: bool = True
    seed: int = 0
    model: dict = field(default_factory=dict)


def load_config(path: str | Path) -> TrainConfig:
    os.environ.setdefault("SLM_SCRATCH", f"/scratch/{os.environ.get('USER', 'user')}/slm")
    raw = yaml.safe_load(Path(path).read_text())
    expanded = {
        key: os.path.expandvars(value) if isinstance(value, str) else value
        for key, value in raw.items()
    }
    return TrainConfig(**expanded)


def _build_optimizer(model: SLM, cfg: TrainConfig, use_cuda: bool) -> torch.optim.AdamW:
    """Weight decay so nas matrizes dos blocos: embeddings e normalizacao ficam de fora."""
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        (no_decay if param.dim() < 2 or name.startswith("tok_emb") else decay).append(param)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.peak_lr,
        betas=(cfg.beta1, cfg.beta2),
        fused=use_cuda,
    )


def _periodic_checkpoints(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("ckpt_*.pt"))


def _truncate_metrics(path: Path, step: int) -> None:
    """Depois de uma queda, descarta o que foi registrado alem do checkpoint usado na retomada."""
    if not path.exists():
        return
    lines = [line for line in path.read_text().splitlines() if json.loads(line)["step"] <= step]
    path.write_text("".join(line + "\n" for line in lines))


@torch.no_grad()
def _validation_loss(
    model: SLM, dataset: RecordDataset, cfg: TrainConfig, device: torch.device, rank: int, world: int
) -> float:
    indices = evenly_spaced_indices(len(dataset), cfg.eval_batches * cfg.eval_batch_size)[rank::world]
    loss_sum = torch.zeros(1, device=device, dtype=torch.float64)
    count = torch.zeros(1, device=device, dtype=torch.float64)
    for start in range(0, len(indices), cfg.eval_batch_size):
        inputs, targets = dataset.batch(indices[start : start + cfg.eval_batch_size])
        inputs = torch.from_numpy(inputs).to(device)
        targets = torch.from_numpy(targets).to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, _ = model(inputs)
        loss_sum += F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        ).double()
        count += (targets != -100).sum()
    if world > 1:
        dist.all_reduce(loss_sum)
        dist.all_reduce(count)
    return (loss_sum / count).item()


def train(
    cfg: TrainConfig,
    max_hours: float | None = None,
    eval_hook: EvalHook | None = None,
    stop_after_steps: int | None = None,
) -> str:
    """Treina ate o fim ou ate o limite de tempo. Retorna "finished" ou "interrupted"."""
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    use_cuda = torch.cuda.is_available()
    if world > 1:
        dist.init_process_group("nccl" if use_cuda else "gloo")
    if use_cuda:
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
        torch.cuda.set_device(device)
        torch.set_float32_matmul_precision("high")
    else:
        device = torch.device("cpu")

    def log(message: str) -> None:
        if rank == 0:
            print(message, flush=True)

    run_dir = Path(cfg.run_dir)
    metrics_path = run_dir / "metrics.jsonl"
    final_path = run_dir / "final.pt"
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
    if world > 1:
        dist.barrier()

    def finish(status: str) -> str:
        if world > 1:
            dist.destroy_process_group()
        return status

    if final_path.exists():
        log(f"{final_path} ja existe: nada a fazer")
        return finish("finished")

    train_data = RecordDataset(cfg.data_dir, use_mask=cfg.use_loss_mask)
    val_data = RecordDataset(cfg.val_dir, use_mask=cfg.use_loss_mask) if cfg.val_dir else None
    seq_len = train_data.seq_len
    split_digits = train_data.meta["split_digits"]
    if cfg.epochs is not None:
        total_tokens = int(cfg.epochs * len(train_data) * seq_len)
    elif cfg.max_tokens is not None:
        total_tokens = cfg.max_tokens
    else:
        raise ValueError("defina max_tokens ou epochs")

    periodic = _periodic_checkpoints(run_dir)
    resume_state = checkpoint.load(periodic[-1], device) if periodic else None
    init_state = None
    if resume_state is not None:
        model_cfg = ModelConfig(**resume_state["model_cfg"])
    elif cfg.init_from:
        init_state = checkpoint.load(cfg.init_from, device)
        model_cfg = ModelConfig(**init_state["model_cfg"])
        if init_state["split_digits"] != split_digits:
            raise ValueError("os dados usam um tokenizer diferente do checkpoint inicial")
    else:
        model_cfg = ModelConfig(**cfg.model)
    if seq_len > model_cfg.max_seq_len:
        raise ValueError(f"dados com seq_len={seq_len} excedem max_seq_len={model_cfg.max_seq_len}")

    torch.manual_seed(cfg.seed)
    model = SLM(model_cfg).to(device)
    optimizer = _build_optimizer(model, cfg, use_cuda)
    step = 0
    records_seen = 0
    if resume_state is not None:
        model.load_state_dict(resume_state["model"])
        optimizer.load_state_dict(resume_state["optimizer"])
        step = resume_state["step"]
        records_seen = resume_state["records_seen"]
        if rank == 0:
            _truncate_metrics(metrics_path, step)
        log(f"retomando de {periodic[-1].name}: step {step}, {records_seen * seq_len:,} tokens")
    else:
        if rank == 0:
            metrics_path.unlink(missing_ok=True)
        if init_state is not None:
            model.load_state_dict(init_state["model"])
            log(f"pesos iniciais de {cfg.init_from}; otimizador zerado")
    del resume_state, init_state

    net = torch.compile(model) if cfg.compile else model
    if world > 1:
        net = DistributedDataParallel(
            net, device_ids=[device.index] if use_cuda else None, broadcast_buffers=False
        )

    log(
        f"params={model.num_params() / 1e6:.1f}M | gpus={world} | batch={cfg.batch_size} | "
        f"seq_len={seq_len} | registros={len(train_data):,} | meta={total_tokens:,} tokens"
    )

    sampler = EpochSampler(len(train_data), cfg.seed)

    def record(row: dict) -> None:
        if rank == 0:
            with metrics_path.open("a") as file:
                file.write(json.dumps({"step": step, "tokens": records_seen * seq_len, **row}) + "\n")

    def evaluate() -> None:
        row: dict[str, float] = {}
        if val_data is not None:
            val_loss = _validation_loss(model, val_data, cfg, device, rank, world)
            row.update(val_loss=val_loss, val_ppl=math.exp(val_loss))
        if eval_hook is not None and rank == 0:
            row.update(eval_hook(model))
        if world > 1:
            dist.barrier()
        if row and rank == 0:
            record(row)
            log(f"step {step:7d} | " + " | ".join(f"{k} {v:.4f}" for k, v in row.items()))

    def save_periodic() -> None:
        if rank != 0:
            return
        checkpoint.save(
            run_dir / f"ckpt_{step:09d}.pt",
            model,
            optimizer=optimizer.state_dict(),
            step=step,
            records_seen=records_seen,
            split_digits=split_digits,
        )
        for old in _periodic_checkpoints(run_dir)[: -cfg.keep_last]:
            old.unlink()

    started = time.perf_counter()
    window_started = started
    window_records = records_seen
    window_loss = torch.zeros(1, device=device)
    window_steps = 0
    if step == 0:
        evaluate()

    while records_seen * seq_len < total_tokens:
        lr = wsd_lr(records_seen * seq_len, total_tokens, cfg.peak_lr, cfg.warmup_tokens, cfg.decay_frac)
        for group in optimizer.param_groups:
            group["lr"] = lr

        indices = sampler.indices(records_seen + rank, cfg.batch_size, stride=world)
        inputs, targets = train_data.batch(indices)
        inputs = torch.from_numpy(inputs).to(device, non_blocking=True)
        targets = torch.from_numpy(targets).to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_cuda):
            _, loss = net(inputs, targets)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        step += 1
        records_seen += cfg.batch_size * world
        window_loss += loss.detach()
        window_steps += 1

        should_stop = stop_after_steps is not None and step >= stop_after_steps
        if step % cfg.log_every == 0:
            if world > 1:
                dist.all_reduce(window_loss)
            mean_loss = window_loss.item() / (window_steps * world)
            if not math.isfinite(mean_loss):
                raise RuntimeError(f"loss nao finita no step {step}: treino abortado")
            now = time.perf_counter()
            tokens_per_second = (records_seen - window_records) * seq_len / (now - window_started)
            record(
                {"loss": mean_loss, "lr": lr, "grad_norm": grad_norm.item(), "tok_per_s": tokens_per_second}
            )
            done = records_seen * seq_len / total_tokens
            log(
                f"step {step:7d} | {100 * done:5.1f}% | loss {mean_loss:.4f} | lr {lr:.2e} | "
                f"grad {grad_norm.item():.2f} | {tokens_per_second:,.0f} tok/s"
            )
            window_started, window_records = now, records_seen
            window_loss.zero_()
            window_steps = 0

            # Todos os processos precisam parar no mesmo step: a decisao vem do rank 0.
            time_is_up = max_hours is not None and now - started > max_hours * 3600
            decision = torch.tensor([time_is_up], device=device, dtype=torch.uint8)
            if world > 1:
                dist.broadcast(decision, 0)
            should_stop = should_stop or bool(decision.item())

        if step % cfg.eval_every == 0:
            evaluate()
        if should_stop:
            save_periodic()
            log(f"interrompido no step {step}; use o mesmo comando para retomar")
            return finish("interrupted")
        if step % cfg.ckpt_every == 0:
            save_periodic()

    if step % cfg.eval_every != 0:
        evaluate()
    if rank == 0:
        checkpoint.save(
            final_path, model, step=step, records_seen=records_seen, split_digits=split_digits
        )
    log(f"fim: {final_path} ({records_seen * seq_len:,} tokens em {step} steps)")
    return finish("finished")


__all__ = ["TrainConfig", "load_config", "train"]
