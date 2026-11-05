"""Decoder-only transformer: RoPE, grouped-query attention, RMSNorm, SwiGLU, tied embeddings."""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 50_261  # GPT-2 (50.257) + <|user|>, <|assistant|>, <|end|>, <|pad|>
    d_model: int = 768
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int = 4
    ffn_hidden: int = 2048
    max_seq_len: int = 1024
    rope_base: float = 10_000.0

    def __post_init__(self):
        if self.d_model % self.n_head:
            raise ValueError("d_model deve ser divisivel por n_head")
        if self.n_head % self.n_kv_head:
            raise ValueError("n_head deve ser divisivel por n_kv_head")
        if (self.d_model // self.n_head) % 2:
            raise ValueError("a dimensao por cabeca deve ser par para o RoPE")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        rms_inv = x_float.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt().reciprocal()
        return (x_float * rms_inv).type_as(x) * self.weight


def rope_tables(seq_len: int, head_dim: int, base: float) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = base ** (-torch.arange(0, head_dim, 2).float() / head_dim)
    angles = torch.outer(torch.arange(seq_len).float(), inv_freq)
    angles = torch.cat([angles, angles], dim=-1)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (batch, tempo, cabecas, dim_cabeca); cos e sin: (tempo, dim_cabeca)."""
    first, second = x.chunk(2, dim=-1)
    rotated = torch.cat([-second, first], dim=-1)
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    return x * cos + rotated * sin


class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.w_q = nn.Linear(cfg.d_model, cfg.n_head * self.head_dim, bias=False)
        self.w_k = nn.Linear(cfg.d_model, cfg.n_kv_head * self.head_dim, bias=False)
        self.w_v = nn.Linear(cfg.d_model, cfg.n_kv_head * self.head_dim, bias=False)
        self.w_out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        batch, time, _ = x.shape
        q = self.w_q(x).view(batch, time, self.n_head, self.head_dim)
        k = self.w_k(x).view(batch, time, self.n_kv_head, self.head_dim)
        v = self.w_v(x).view(batch, time, self.n_kv_head, self.head_dim)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        group_size = self.n_head // self.n_kv_head
        k = k.repeat_interleave(group_size, dim=2)
        v = v.repeat_interleave(group_size, dim=2)

        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        )
        return self.w_out(out.transpose(1, 2).reshape(batch, time, -1))


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w_gate = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.w_up = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.w_out = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_out(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm_attn = RMSNorm(cfg.d_model)
        self.attn = GroupedQueryAttention(cfg)
        self.norm_ffn = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_attn(x), cos, sin)
        return x + self.ffn(self.norm_ffn(x))


class SLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm_out = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        cos, sin = rope_tables(cfg.max_seq_len, cfg.d_model // cfg.n_head, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self._init_weights()

    def _init_weights(self) -> None:
        residual_std = 0.02 / math.sqrt(2 * self.cfg.n_layer)
        for name, param in self.named_parameters():
            if name.endswith("w_out.weight"):
                nn.init.normal_(param, std=residual_std)
            elif param.dim() >= 2:
                nn.init.normal_(param, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self, tokens: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        time = tokens.size(1)
        if time > self.cfg.max_seq_len:
            raise ValueError(f"sequencia de {time} tokens excede max_seq_len={self.cfg.max_seq_len}")

        cos, sin = self.rope_cos[:time], self.rope_sin[:time]
        x = self.tok_emb(tokens)
        for block in self.blocks:
            x = block(x, cos, sin)
        logits = self.lm_head(self.norm_out(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
            )
        return logits, loss
