"""Geracao de texto, sem cache de atencao: cada token novo reprocessa o contexto inteiro."""

from collections.abc import Iterator

import torch
import torch.nn.functional as F

from slm.model import SLM


def _autocast(model: SLM) -> torch.autocast:
    device_type = next(model.parameters()).device.type
    return torch.autocast(device_type, dtype=torch.bfloat16, enabled=device_type == "cuda")


@torch.inference_mode()
def stream(
    model: SLM,
    prompt: list[int],
    max_new_tokens: int,
    temperature: float = 0.7,
    top_k: int | None = 40,
    stop_tokens: frozenset[int] = frozenset(),
    generator: torch.Generator | None = None,
) -> Iterator[int]:
    """Produz um token por vez. Com temperatura 0, escolhe sempre o mais provavel."""
    device = next(model.parameters()).device
    context = torch.tensor([prompt], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        with _autocast(model):
            logits, _ = model(context[:, -model.cfg.max_seq_len :])
        logits = logits[0, -1].float()
        if temperature <= 0:
            token = int(logits.argmax())
        else:
            logits = logits / temperature
            if top_k:
                threshold = logits.topk(min(top_k, logits.size(-1))).values[-1]
                logits = logits.masked_fill(logits < threshold, -float("inf"))
            token = int(torch.multinomial(F.softmax(logits, dim=-1), 1, generator=generator))
        if token in stop_tokens:
            return
        yield token
        context = torch.cat([context, torch.tensor([[token]], device=device)], dim=1)


@torch.inference_mode()
def greedy_batch(model: SLM, prompts: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    """Continua varios prompts de mesmo tamanho e retorna so os tokens novos."""
    context = prompts
    for _ in range(max_new_tokens):
        with _autocast(model):
            logits, _ = model(context[:, -model.cfg.max_seq_len :])
        context = torch.cat([context, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
    return context[:, prompts.size(1) :]
