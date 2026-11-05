"""Compara dois JSONs de scripts/evaluate.py com o teste de McNemar, item a item."""

import argparse
import json

from slm.evaluation import mcnemar_exact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first")
    parser.add_argument("second")
    args = parser.parse_args()

    with open(args.first) as file:
        first = json.load(file)
    with open(args.second) as file:
        second = json.load(file)

    print(f"A = {args.first}\nB = {args.second}")
    for name, values in first.items():
        if not isinstance(values, dict) or "correct" not in values or name not in second:
            continue
        test = mcnemar_exact(values["correct"], second[name]["correct"])
        print(
            f"{name:14s} A={values['accuracy']:.4f} B={second[name]['accuracy']:.4f} | "
            f"so A={test['only_first']} so B={test['only_second']} | p={test['p_value']:.4g}"
        )


if __name__ == "__main__":
    main()
