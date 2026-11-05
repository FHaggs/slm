"""Etapa 1: pre-treino. Previsao do proximo token em texto nao rotulado."""

import argparse

from slm.trainer import load_config, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-hours", type=float)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg.init_from or cfg.use_loss_mask:
        raise ValueError("o pre-treino parte do zero e nao usa mascara")
    train(cfg, max_hours=args.max_hours)


if __name__ == "__main__":
    main()
