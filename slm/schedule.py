"""Schedule Warmup-Stable-Decay, medido em tokens para nao depender de batch nem de numero de GPUs."""


def wsd_lr(
    tokens_seen: int, total_tokens: int, peak_lr: float, warmup_tokens: int, decay_frac: float
) -> float:
    decay_start = total_tokens * (1 - decay_frac)
    if tokens_seen < warmup_tokens:
        return peak_lr * (tokens_seen + 1) / warmup_tokens
    if tokens_seen < decay_start:
        return peak_lr
    remaining = (total_tokens - tokens_seen) / (total_tokens - decay_start)
    return peak_lr * max(remaining, 0.0)
