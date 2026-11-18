"""Grafico e tabela dos resultados finais (results/final/{pretrain,mid_c,mid_m,sft}.json).

Uso: uv run --no-project --with matplotlib python scripts/plot_results.py
Barras com intervalo de confianca de 95% (Wilson). O que faltar (por exemplo mid_c) e pulado.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = [("pretrain", "Pré-treino", "#52514e"), ("mid_c", "C", "#eb6834"), ("mid_m", "M", "#2a78d6"), ("sft", "SFT", "#1baf7a")]
TESTS = [("hellaswag", "HellaSwag"), ("arc_easy", "ARC-Easy"), ("arithmetic", "Aritmética"), ("gsm8k", "GSM8K")]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/final"))
    args = parser.parse_args()
    data = {name: json.loads((args.results / f"{name}.json").read_text()) for name, _, _ in MODELS if (args.results / f"{name}.json").exists()}
    models = [m for m in MODELS if m[0] in data]

    fig, axes = plt.subplots(1, 4, figsize=(11, 3.9), facecolor=SURFACE)
    for ax, (key, title) in zip(axes, TESTS):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        top = 0.0
        tops = []
        ax.set_ylim(0, 6 if key == "gsm8k" else 1)
        for i, (name, label, color) in enumerate(models):
            if key not in data[name]:  # etapa ainda nao medida (ex.: GSM8K do C)
                ax.text(i, 0.02 * ax.get_ylim()[1] + 0.001, "a medir", ha="center", rotation=90, va="bottom", fontsize=8, color=MUTED)
                continue
            result = data[name][key]
            p, n = result["accuracy"], len(result["correct"])
            low, high = wilson(p, n)
            ax.bar(i, 100 * p, color=color, width=0.62)
            ax.errorbar(i, 100 * p, yerr=[[100 * (p - low)], [100 * (high - p)]], color=INK, lw=1, capsize=3)
            tops.append((i, 100 * p, 100 * high))
            top = max(top, 100 * high)
        ax.set_ylim(0, top * 1.18)
        for i, value, high in tops:
            ax.text(i, high + 0.02 * top, f"{value:.1f}%", ha="center", va="bottom", fontsize=8.5, color=INK)
        ax.set_title(title, color=INK, fontsize=11, fontweight="bold", loc="left")
        ax.set_xticks(range(len(models)), [m[1] for m in models], fontsize=8.5, color=MUTED)
        ax.tick_params(axis="y", colors=MUTED, labelsize=8.5)
        if key == "hellaswag":
            ax.axhline(25, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=5)
            ax.text(0.98, 0.97, "tracejado: acaso (25%)", transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=MUTED)
            ax.set_ylim(0, 35)
        if key == "arc_easy":
            ax.axhline(25, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=5)
            ax.text(0.98, 0.97, "tracejado: acaso (25%)", transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=MUTED)
    axes[0].set_ylabel("Acerto (%), IC de 95%", color=MUTED)
    fig.tight_layout()
    out = args.results / "figs" / "resultados.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)

    lines = ["| Modelo | Perplexidade | HellaSwag | ARC-Easy | Aritmética | GSM8K | GSM8K: respondeu |", "|---|---|---|---|---|---|---|"]
    for name, label, _ in models:
        d = data[name]
        lines.append(f"| {label} | {d['perplexity']['perplexity']:.2f} | " + " | ".join(f"{100 * d[k]['accuracy']:.1f}%" if k in d else "a medir" for k, _ in TESTS)
                     + (f" | {100 * d['gsm8k']['answered']:.1f}% |" if "gsm8k" in d else " | a medir |"))
    (args.results / "tabela.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("figura:", out)


if __name__ == "__main__":
    main()
