"""Salvar e carregar checkpoints. A escrita e atomica: uma queda nunca deixa um arquivo pela metade."""

import os
from dataclasses import asdict
from pathlib import Path

import torch

from slm.model import SLM, ModelConfig
from slm.tokenizer import Tokenizer


def save(path: str | Path, model: SLM, **extra) -> None:
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "model_cfg": asdict(model.cfg), **extra}, temporary)
    os.replace(temporary, path)


def load(path: str | Path, device: str | torch.device = "cpu") -> dict:
    return torch.load(path, map_location=device, weights_only=True)


def load_model(path: str | Path, device: str | torch.device = "cpu") -> tuple[SLM, Tokenizer]:
    """Retorna o modelo e o tokenizer com que ele foi treinado."""
    state = load(path, device)
    model = SLM(ModelConfig(**state["model_cfg"])).to(device)
    model.load_state_dict(state["model"])
    return model.eval(), Tokenizer(split_digits=state["split_digits"])
