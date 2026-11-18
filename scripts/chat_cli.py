"""Conversa com o modelo no terminal (sem navegador), para o servidor.

Uso, da raiz do repositorio:
    uv run python scripts/chat_cli.py --checkpoint <final.pt> [--temperature 0.7] [--top-k 40]

Comandos durante a conversa: /reset (nova conversa), /temp 0.5, /topk 20, /max 300, /quit.
O modelo foi treinado em ingles: escreva em ingles.
"""

import argparse

import torch

from slm.chat import render_prompt
from slm.checkpoint import load_model
from slm.generate import stream
from slm.tokenizer import END, ENDOFTEXT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model(args.checkpoint, device)
    print(f"{model.num_params() / 1e6:.1f}M parametros em {device} | {args.checkpoint}")
    print("Comandos: /reset  /temp X  /topk K  /max N  /quit\n")

    temperature, top_k, max_new = args.temperature, args.top_k, args.max_new_tokens
    messages: list[dict[str, str]] = []
    while True:
        try:
            line = input("voce> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line == "/quit":
            break
        if line == "/reset":
            messages = []
            print("(conversa limpa)")
            continue
        if line.startswith(("/temp ", "/topk ", "/max ")):
            command, value = line.split(maxsplit=1)
            try:
                if command == "/temp":
                    temperature = float(value)
                elif command == "/topk":
                    top_k = int(value)
                else:
                    max_new = int(value)
            except ValueError:
                print("valor invalido")
                continue
            print(f"(temperatura={temperature}, top-k={top_k}, max={max_new})")
            continue

        messages.append({"role": "user", "content": line})
        prompt = render_prompt(messages, tokenizer)
        if len(prompt) + max_new > model.cfg.max_seq_len:
            print("(conversa passou do contexto de 1.024 tokens: use /reset)")
            messages.pop()
            continue
        tokens: list[int] = []
        printed = ""
        print("modelo> ", end="", flush=True)
        for token in stream(
            model, prompt, max_new, temperature=temperature, top_k=top_k, stop_tokens=frozenset({END, ENDOFTEXT})
        ):
            tokens.append(token)
            text = tokenizer.decode(tokens)
            if not text.endswith("�"):  # um caractere pode ocupar mais de um token
                print(text[len(printed) :], end="", flush=True)
                printed = text
        print()
        messages.append({"role": "assistant", "content": printed})


if __name__ == "__main__":
    main()
