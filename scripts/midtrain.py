"""Etapa 2: mid-training. Parte dos pesos do pre-treino e treina em chat, com loss em todos os tokens.

A aritmetica e medida a cada avaliacao, gerando a curva de matematica ao longo do treino.
"""

import argparse

from slm.data import RecordDataset
from slm.evaluation import arithmetic_items, run_arithmetic
from slm.tokenizer import Tokenizer
from slm.trainer import load_config, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-hours", type=float)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if not cfg.init_from or cfg.use_loss_mask:
        raise ValueError("o mid-training exige init_from e nao usa mascara")

    items = arithmetic_items()
    tokenizer = Tokenizer(split_digits=RecordDataset(cfg.data_dir).meta["split_digits"])

    def arithmetic_hook(model) -> dict[str, float]:
        result = run_arithmetic(model, tokenizer, items)
        return {"arith_acc": result["accuracy"], **{f"arith_{k}": v for k, v in result["cells"].items()}}

    train(cfg, max_hours=args.max_hours, eval_hook=arithmetic_hook)


if __name__ == "__main__":
    main()
