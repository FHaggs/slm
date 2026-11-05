"""Etapa 3: supervised fine-tuning. A loss conta apenas os tokens das respostas do assistente."""

import argparse

from slm.trainer import load_config, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-hours", type=float)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if not cfg.init_from or not cfg.use_loss_mask:
        raise ValueError("o SFT exige init_from e use_loss_mask")
    train(cfg, max_hours=args.max_hours)


if __name__ == "__main__":
    main()
