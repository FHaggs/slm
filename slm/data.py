"""Registros de tamanho fixo em memmap e amostragem deterministica.

Cada conjunto de dados e um diretorio com tokens.bin (uint16), mask.bin opcional (uint8) e meta.json.
Um registro tem seq_len + 1 tokens: as entradas sao [:-1] e os alvos sao [1:].
"""

import json
from pathlib import Path

import numpy as np

from slm.tokenizer import PAD


def write_records(
    directory: str | Path, tokens: np.ndarray, mask: np.ndarray | None = None, **meta
) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if tokens.ndim != 2:
        raise ValueError("tokens deve ter forma (registros, tamanho do registro)")
    tokens.astype(np.uint16).tofile(directory / "tokens.bin")
    if mask is not None:
        mask.astype(np.uint8).tofile(directory / "mask.bin")
    info = {
        "seq_len": tokens.shape[1] - 1,
        "num_records": tokens.shape[0],
        "has_mask": mask is not None,
        **meta,
    }
    (directory / "meta.json").write_text(json.dumps(info, indent=2))


def make_targets(records: np.ndarray, masks: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Retorna entradas e alvos. Alvos em <|pad|> ou fora da mascara viram -100."""
    inputs = records[:, :-1].astype(np.int64)
    targets = records[:, 1:].astype(np.int64)
    targets[targets == PAD] = -100
    if masks is not None:
        targets[masks[:, 1:] == 0] = -100
    return inputs, targets


def evenly_spaced_indices(num_records: int, count: int) -> np.ndarray:
    """Registros vizinhos costumam vir do mesmo documento: uma validacao representativa os espalha."""
    count = min(count, num_records)
    return np.floor(np.arange(count) * num_records / count).astype(np.int64)


class RecordDataset:
    def __init__(self, directory: str | Path, use_mask: bool = False):
        directory = Path(directory)
        self.meta = json.loads((directory / "meta.json").read_text())
        self.seq_len = self.meta["seq_len"]
        self.num_records = self.meta["num_records"]
        shape = (self.num_records, self.seq_len + 1)
        self.tokens = np.memmap(directory / "tokens.bin", dtype=np.uint16, mode="r", shape=shape)
        self.mask = None
        if use_mask:
            if not self.meta["has_mask"]:
                raise ValueError(f"{directory} nao tem mask.bin, exigido pelo SFT")
            self.mask = np.memmap(directory / "mask.bin", dtype=np.uint8, mode="r", shape=shape)

    def __len__(self) -> int:
        return self.num_records

    def batch(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        records = np.asarray(self.tokens[indices])
        masks = None if self.mask is None else np.asarray(self.mask[indices])
        return make_targets(records, masks)


class EpochSampler:
    """Permutacao sem reposicao por epoca, calculada a partir da posicao global.

    Nao ha estado alem do numero de registros ja vistos, entao retomar o treino, mesmo com outro
    numero de GPUs, continua exatamente do ponto em que parou.
    """

    def __init__(self, num_records: int, seed: int):
        self.num_records = num_records
        self.seed = seed
        self._epoch = -1
        self._permutation = np.empty(0, dtype=np.int64)

    def _permutation_for(self, epoch: int) -> np.ndarray:
        if epoch != self._epoch:
            rng = np.random.default_rng([self.seed, epoch])
            self._permutation = rng.permutation(self.num_records)
            self._epoch = epoch
        return self._permutation

    def indices(self, start: int, count: int, stride: int = 1) -> np.ndarray:
        positions = start + np.arange(count) * stride
        result = np.empty(count, dtype=np.int64)
        for i, position in enumerate(positions):
            epoch, offset = divmod(int(position), self.num_records)
            result[i] = self._permutation_for(epoch)[offset]
        return result
