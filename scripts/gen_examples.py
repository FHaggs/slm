"""Gera exemplos de resposta de varios checkpoints para as mesmas perguntas e salva em Markdown.

Uso: uv run python scripts/gen_examples.py --model M=<mid_m/final.pt> --model SFT=<sft/final.pt> --output results/final/exemplos.md
Cada pergunta usa a mesma semente em todos os modelos (temperatura 0,7 e top-k 40 por padrao). Nada e escolhido a mao:
as respostas ruins ficam no arquivo.
"""

import argparse
from pathlib import Path

import torch

from slm.chat import render_prompt
from slm.checkpoint import load_model
from slm.generate import stream
from slm.tokenizer import END, ENDOFTEXT

QUESTIONS = [
    "What is 7 + 5?",
    "What is 12 times 3?",
    "What is 48 - 19?",
    "Tom has 3 apples and buys 4 more. How many apples does he have?",
    "A train travels 60 km per hour for 2 hours. How far does it go?",
    "What is photosynthesis?",
    "What is the capital of France?",
    "Write a short poem about the sea.",
    "Hi! Who are you?",
]


def answer(model, tokenizer, question: str, temperature: float, top_k: int, max_new: int, seed: int) -> str:
    generator = torch.Generator().manual_seed(seed)
    prompt = render_prompt([{"role": "user", "content": question}], tokenizer)
    tokens = list(stream(model, prompt, max_new, temperature=temperature, top_k=top_k,
                         stop_tokens=frozenset({END, ENDOFTEXT}), generator=generator))
    return tokenizer.decode(tokens).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, help="NOME=caminho/final.pt (repetir)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    models = {}
    for spec in args.model:
        name, path = spec.split("=", 1)
        models[name] = load_model(path, "cpu")
    lines = [
        "# Exemplos de geração",
        "",
        f"Temperatura {args.temperature}, top-k {args.top_k}, no máximo {args.max_new_tokens} tokens, semente {args.seed} "
        "(a mesma em todos os modelos). Conversa de um turno, em inglês. Nada foi escolhido a mão: "
        "as respostas ruins também estão aqui. Modelos: " + ", ".join(f"{n} (`{p.split('=', 1)[1]}`)" for n, p in zip(models, args.model)) + ".",
        "",
    ]
    for question in QUESTIONS:
        lines += [f"## {question}", ""]
        for name, (model, tokenizer) in models.items():
            text = answer(model, tokenizer, question, args.temperature, args.top_k, args.max_new_tokens, args.seed)
            lines += [f"**{name}:**", "", *("> " + line for line in (text or "(resposta vazia)").splitlines()), ""]
        print("ok:", question, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print("salvo em", args.output)


if __name__ == "__main__":
    main()
