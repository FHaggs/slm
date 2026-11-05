import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from slm import checkpoint
from slm.trainer import TrainConfig, train

ROOT = Path(__file__).resolve().parent.parent


def make_config(run_dir, data_dir, model, **overrides) -> TrainConfig:
    settings = {
        "run_dir": str(run_dir),
        "data_dir": str(data_dir),
        "val_dir": str(data_dir),
        "max_tokens": 16 * 4 * 40,
        "batch_size": 4,
        "peak_lr": 3e-3,
        "warmup_tokens": 16 * 4 * 2,
        "log_every": 5,
        "eval_every": 10,
        "eval_batches": 2,
        "eval_batch_size": 4,
        "ckpt_every": 10,
        "compile": False,
        "model": model,
    }
    return TrainConfig(**{**settings, **overrides})


def read_metrics(run_dir) -> list[dict]:
    return [json.loads(line) for line in (Path(run_dir) / "metrics.jsonl").read_text().splitlines()]


def test_training_reduces_loss_and_saves_final(tmp_path, tiny_dataset, tiny_model_cfg):
    run_dir = tmp_path / "run"
    assert train(make_config(run_dir, tiny_dataset, tiny_model_cfg)) == "finished"
    assert (run_dir / "final.pt").exists()
    rows = read_metrics(run_dir)
    losses = [row["loss"] for row in rows if "loss" in row]
    assert losses[-1] < losses[0] - 1
    validation = [row["val_loss"] for row in rows if "val_loss" in row]
    assert validation[-1] < validation[0]
    assert rows[-1]["tokens"] >= 16 * 4 * 40


def test_resume_matches_uninterrupted_run(tmp_path, tiny_dataset, tiny_model_cfg):
    """Interromper e retomar precisa chegar exatamente aos mesmos pesos e metricas."""
    straight = tmp_path / "straight"
    train(make_config(straight, tiny_dataset, tiny_model_cfg))

    resumed = tmp_path / "resumed"
    cfg = make_config(resumed, tiny_dataset, tiny_model_cfg)
    assert train(cfg, stop_after_steps=17) == "interrupted"
    assert not (resumed / "final.pt").exists()
    assert (resumed / "ckpt_000000017.pt").exists()
    assert train(cfg) == "finished"

    expected = checkpoint.load(straight / "final.pt")
    actual = checkpoint.load(resumed / "final.pt")
    assert expected["step"] == actual["step"]
    for name, tensor in expected["model"].items():
        assert torch.equal(tensor, actual["model"][name]), name
    # O primeiro registro apos a retomada cobre menos steps (18-20, e nao 16-20), entao fica de fora.
    train_loss = lambda run: {r["step"]: r["loss"] for r in read_metrics(run) if "loss" in r and r["step"] != 20}
    assert train_loss(straight) == pytest.approx(train_loss(resumed))
    val_loss = lambda run: {r["step"]: r["val_loss"] for r in read_metrics(run) if "val_loss" in r}
    assert val_loss(straight) == pytest.approx(val_loss(resumed))
    steps =[(row["step"], "loss" in row) for row in read_metrics(resumed)]
    assert len(steps) == len(set(steps))


def test_keeps_only_last_periodic_checkpoints(tmp_path, tiny_dataset, tiny_model_cfg):
    run_dir = tmp_path / "run"
    train(make_config(run_dir, tiny_dataset, tiny_model_cfg))
    assert [path.name for path in sorted(run_dir.glob("ckpt_*.pt"))] == [
        "ckpt_000000030.pt",
        "ckpt_000000040.pt",
    ]


def test_finished_run_is_not_trained_again(tmp_path, tiny_dataset, tiny_model_cfg):
    run_dir = tmp_path / "run"
    cfg = make_config(run_dir, tiny_dataset, tiny_model_cfg)
    train(cfg)
    before = (run_dir / "final.pt").stat().st_mtime_ns
    assert train(cfg) == "finished"
    assert (run_dir / "final.pt").stat().st_mtime_ns == before


def test_init_from_loads_weights_with_fresh_optimizer(tmp_path, tiny_dataset, tiny_model_cfg):
    first = tmp_path / "first"
    train(make_config(first, tiny_dataset, tiny_model_cfg))
    state = checkpoint.load(first / "final.pt")
    assert "optimizer" not in state

    second = tmp_path / "second"
    cfg = make_config(
        second, tiny_dataset, {}, init_from=str(first / "final.pt"), max_tokens=16 * 4 * 5, eval_every=1000
    )
    train(cfg)
    rows = read_metrics(second)
    trained_loss = [row["val_loss"] for row in read_metrics(first) if "val_loss" in row][-1]
    assert rows[0]["step"] == 0
    assert rows[0]["val_loss"] == pytest.approx(trained_loss, rel=1e-4)


def test_two_process_ddp_and_resume_with_fewer_processes(tmp_path, tiny_dataset, tiny_model_cfg):
    """Exercita o caminho do DDP em CPU, e a retomada em outro numero de processos."""
    run_dir = tmp_path / "ddp"
    config_path = tmp_path / "config.yaml"
    settings = vars(make_config(run_dir, tiny_dataset, tiny_model_cfg))
    config_path.write_text(yaml.safe_dump(settings))
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"}

    subprocess.run(
        [
            sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
            str(ROOT / "scripts/pretrain.py"), "--config", str(config_path), "--max-hours", "0",
        ],
        check=True,
        env=environment,
        timeout=180,
    )
    interrupted = sorted(run_dir.glob("ckpt_*.pt"))
    assert interrupted, "o limite de tempo deveria salvar um checkpoint"
    assert checkpoint.load(interrupted[-1])["step"] == 5
    assert checkpoint.load(interrupted[-1])["records_seen"] == 5 * 4 * 2

    subprocess.run(
        [sys.executable, str(ROOT / "scripts/pretrain.py"), "--config", str(config_path)],
        check=True,
        env=environment,
        timeout=180,
    )
    final = checkpoint.load(run_dir / "final.pt")
    assert final["records_seen"] * 16 >= 16 * 4 * 40
