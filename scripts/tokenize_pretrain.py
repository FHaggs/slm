"""Baixa uma amostra do FineWeb-Edu e gera os registros de treino e de validacao do pre-treino.

Os arquivos de treino vem de dumps espalhados no tempo. A validacao vem de um dump que o treino nao
usa, sem URLs presentes no treino. Os documentos sao embaralhados antes de virar registros.
"""

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files

from slm.data import write_records
from slm.tokenizer import ENDOFTEXT, Tokenizer

REPO = "HuggingFaceFW/fineweb-edu"


def select_files(num_train: int) -> tuple[list[str], str]:
    by_dump: dict[str, list[str]] = {}
    for path in list_repo_files(REPO, repo_type="dataset"):
        parts = path.split("/")
        if parts[0] == "data" and path.endswith(".parquet"):
            by_dump.setdefault(parts[1], []).append(path)
    dumps = sorted(by_dump)
    val_dump = dumps[-1]
    train_dumps = dumps[:-1]
    positions = np.linspace(0, len(train_dumps) - 1, num_train).round().astype(int)
    train_files = [sorted(by_dump[train_dumps[i]])[0] for i in positions]
    return train_files, sorted(by_dump[val_dump])[0]


def read_documents(
    path: str, budget: int, tokenizer: Tokenizer, excluded_urls: set[int], workers: int
) -> tuple[list[np.ndarray], list[int]]:
    local = hf_hub_download(REPO, path, repo_type="dataset")
    documents: list[np.ndarray] = []
    url_hashes: list[int] = []
    count = 0
    parquet = pq.ParquetFile(local)
    for batch in parquet.iter_batches(batch_size=4096, columns=["text", "url"]):
        hashes = [hash(url) for url in batch.column("url").to_pylist()]
        texts = batch.column("text").to_pylist()
        kept = [i for i, url_hash in enumerate(hashes) if url_hash not in excluded_urls]
        encoded = tokenizer.encode_batch([texts[i] for i in kept], num_threads=workers)
        for i, ids in zip(kept, encoded):
            documents.append(np.array([ENDOFTEXT, *ids], dtype=np.uint16))
            url_hashes.append(hashes[i])
            count += len(ids) + 1
        if count >= budget:
            break
    print(f"{path}: {len(documents):,} documentos, {count:,} tokens", flush=True)
    return documents, url_hashes


def to_records(documents: list[np.ndarray], record_len: int, max_tokens: int, seed: int) -> np.ndarray:
    order = np.random.default_rng(seed).permutation(len(documents))
    tokens = np.concatenate([documents[i] for i in order])
    num_records = min(len(tokens), max_tokens) // record_len
    return tokens[: num_records * record_len].reshape(num_records, record_len)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--val-tokens", type=int, default=50_000_000)
    parser.add_argument("--num-train-files", type=int, default=12)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--split-digits", action="store_true")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    tokenizer = Tokenizer(split_digits=args.split_digits)
    train_files, val_file = select_files(args.num_train_files)
    budget = int(args.max_tokens * 1.02) // len(train_files) + 1

    documents: list[np.ndarray] = []
    seen_urls: set[int] = set()
    for path in train_files:
        file_documents, hashes = read_documents(path, budget, tokenizer, set(), args.workers)
        documents += file_documents
        seen_urls.update(hashes)
    train = to_records(documents, args.seq_len + 1, args.max_tokens, args.seed)
    write_records(
        args.out / "train",
        train,
        split_digits=args.split_digits,
        files=train_files,
        documents=len(documents),
    )
    print(f"treino: {train.shape[0]:,} registros, {train.size:,} tokens", flush=True)
    del documents, train

    val_documents, _ = read_documents(val_file, args.val_tokens, tokenizer, seen_urls, args.workers)
    val = to_records(val_documents, args.seq_len + 1, args.val_tokens, args.seed)
    write_records(
        args.out / "val",
        val,
        split_digits=args.split_digits,
        files=[val_file],
        documents=len(val_documents),
    )
    print(f"validacao: {val.shape[0]:,} registros, {val.size:,} tokens", flush=True)


if __name__ == "__main__":
    main()
