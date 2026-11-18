"""Graficos das curvas de treino (pre-treino, mid-training M e C, SFT) a partir dos metrics.jsonl.

Uso, sem instalar nada no projeto (so le JSON):
    uv run --no-project --with matplotlib python scripts/plot_curves.py \
        --pretrain P/metrics.jsonl --mid-m M/metrics.jsonl --mid-c C/metrics.jsonl --sft S/metrics.jsonl
Cores por modelo (fixas em todas as figuras): pre-treino cinza, M azul, C laranja, SFT verde-agua.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLOR = {"pretrain": "#52514e", "M": "#2a78d6", "C": "#eb6834", "SFT": "#1baf7a"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def read(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9)


def label(ax, x, y, text, color, **kwargs) -> None:
    ax.annotate(text, (x, y), color=color, fontsize=9, fontweight="bold", **kwargs)


def pretrain(rows, out: Path) -> None:
    train = [(r["tokens"] / 1e9, r["loss"]) for r in rows if "loss" in r]
    val = [(r["tokens"] / 1e9, r["val_loss"]) for r in rows if "val_loss" in r and r["step"] > 0]
    lr = [(r["tokens"] / 1e9, r["lr"]) for r in rows if "lr" in r]
    peak = max(v for _, v in lr)
    decay_start = [t for t, v in lr if v >= 0.999 * peak][-1]  # ultimo ponto do patamar do WSD
    end = lr[-1][0]

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.5, 5.6), sharex=True, height_ratios=[3, 1.2], facecolor=SURFACE)
    for ax in (top, bottom):
        style(ax)
        ax.axvspan(decay_start, end, color=GRID, alpha=0.7, lw=0)
    top.plot(*zip(*train), color=COLOR["pretrain"], lw=1.2)
    top.plot(*zip(*val), color=COLOR["M"], lw=2, marker="o", ms=4)
    top.set_ylim(2.6, 5.5)
    top.set_ylabel("Loss", color=MUTED)
    top.set_title("Pré-treino: loss de treino e de validação", loc="left", color=INK, fontsize=12, fontweight="bold")
    at = min(range(len(train)), key=lambda i: abs(train[i][0] - 1.0))
    label(top, 1.0, train[at][1] + 0.08, "Treino", COLOR["pretrain"])
    label(top, 1.0, val[len(val) // 2][1] - 0.1, "Validação", COLOR["M"], va="top")
    top.text(1.95, 4.3, f"Validação final: {val[-1][1]:.2f}\n(perplexidade {2.718281828 ** val[-1][1]:.1f})",
             ha="right", color=INK, fontsize=10)
    top.text((decay_start + end) / 2, 5.3, "decaimento", ha="center", color=MUTED, fontsize=8)
    bottom.plot(*zip(*lr), color=INK, lw=1.5)
    bottom.set_ylabel("Taxa de\naprendizado", color=MUTED, fontsize=9)
    bottom.set_xlabel("Tokens vistos (bilhões)", color=MUTED)
    bottom.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    bottom.set_ylim(0, peak * 1.15)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def mid_arith(rows_m, rows_c, pretrain_acc, out: Path) -> None:
    def curve(rows):
        return [(r["tokens"] / 1e6, 100 * r["arith_acc"]) for r in rows if "arith_acc" in r]

    m, c = curve(rows_m), curve(rows_c)
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
    style(ax)
    ax.axhline(100 * pretrain_acc, color=COLOR["pretrain"], lw=1.2, ls=(0, (4, 3)))
    ax.plot(*zip(*m), color=COLOR["M"], lw=2, marker="o", ms=5)
    ax.plot(*zip(*c), color=COLOR["C"], lw=2, marker="o", ms=5)
    ax.set_ylim(-0.9, 10)
    ax.set_yticks(range(0, 11, 2))
    ax.set_xlabel("Tokens de mid-training (milhões)", color=MUTED)
    ax.set_ylabel("Acerto em aritmética (%)", color=MUTED)
    ax.set_title("Mid-training: aritmética de M (com matemática) e C (só chat)", loc="left", color=INK,
                 fontsize=12, fontweight="bold")
    label(ax, m[1][0] + 5, m[1][1] + 0.5, "M: chat + matemática", COLOR["M"])
    label(ax, c[2][0], c[2][1] + 1.4, "C: só chat", COLOR["C"])
    label(ax, 100, 100 * pretrain_acc - 0.15, f"pré-treino ({100 * pretrain_acc:.1f}%)", COLOR["pretrain"], va="top")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def sft(rows, out: Path) -> None:
    train = [(r["step"], r["loss"]) for r in rows if "loss" in r]
    val = [(r["step"], r["val_loss"]) for r in rows if "val_loss" in r]
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
    style(ax)
    ax.plot(*zip(*train), color=COLOR["pretrain"], lw=1.2)
    ax.plot(*zip(*val), color=COLOR["SFT"], lw=2, marker="o", ms=4)
    ax.set_ylim(1.1, 1.65)
    ax.set_xlabel("Passo de treino", color=MUTED)
    ax.set_ylabel("Loss", color=MUTED)
    ax.set_title("SFT: loss de treino e de validação", loc="left", color=INK, fontsize=12, fontweight="bold")
    label(ax, 1500, 1.32, "Treino", COLOR["pretrain"])
    label(ax, 1500, val[0][1] + 0.04, "Validação", COLOR["SFT"])
    ax.text(val[-1][0], val[-1][1] - 0.06, f"validação: {val[0][1]:.3f} → {val[-1][1]:.3f}",
            ha="right", va="top", color=INK, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    for name in ("pretrain", "mid-m", "mid-c", "sft"):
        parser.add_argument(f"--{name}", required=True, help="metrics.jsonl")
    parser.add_argument("--out", type=Path, default=Path("results/final/figs"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    mid_m, mid_c = read(args.mid_m), read(args.mid_c)
    pretrain(read(args.pretrain), args.out / "pretrain_loss.png")
    mid_arith(mid_m, mid_c, [r["arith_acc"] for r in mid_m if "arith_acc" in r][0], args.out / "mid_aritmetica.png")
    sft(read(args.sft), args.out / "sft_loss.png")
    print("figuras em", args.out)


if __name__ == "__main__":
    main()
