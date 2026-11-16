"""Acumulo de gradiente: um passo com grad_accum micro-lotes tem de ser equivalente a um lote unico.

Rodam em CPU (float32), onde a equivalencia e exata a menos do arredondamento da soma em pedacos.
"""

import numpy as np
import pytest
import torch

from slm import checkpoint
from slm.data import EpochSampler, RecordDataset, write_records
from slm.model import SLM, ModelConfig
from slm.trainer import TrainConfig, _accumulated_step, train

from test_trainer import make_config, read_metrics


@pytest.fixture
def uneven_masked_dataset(tmp_path):
    """Registros com quantidades muito diferentes de tokens validos (padding e mascara).

    A media ingenua das medias de cada micro-lote falha aqui; a ponderacao por token valido, nao.
    """
    rng = np.random.default_rng(1)
    records = rng.integers(0, 50, size=(48, 17))
    mask = np.ones_like(records, dtype=np.uint8)
    for i in range(48):
        if i % 4 == 0:
            mask[i, 3:] = 0  # quase tudo fora da resposta
        elif i % 4 == 1:
            records[i, 6:] = 50260  # <|pad|>: alvos viram -100
    directory = tmp_path / "masked"
    write_records(directory, records, mask=mask, split_digits=False)
    return directory


def _gradients(model, cfg, data_dir, use_mask):
    data = RecordDataset(data_dir, use_mask=use_mask)
    sampler = EpochSampler(len(data), cfg.seed)
    model.zero_grad(set_to_none=True)
    if cfg.grad_accum == 1:
        inputs, targets = data.batch(sampler.indices(0, cfg.batch_size))
        _, loss = model(torch.from_numpy(inputs), torch.from_numpy(targets))
        loss.backward()
    else:
        loss = _accumulated_step(model, data, sampler, 0, cfg, 0, 1, torch.device("cpu"), False)
    return loss.detach(), {name: p.grad.detach().clone() for name, p in model.named_parameters()}


@pytest.mark.parametrize("dataset, use_mask", [("tiny_dataset", False), ("uneven_masked_dataset", True)])
def test_one_step_matches_single_batch(request, tmp_path, tiny_model_cfg, dataset, use_mask):
    """6 micro-lotes de 4 = 1 lote de 24: mesma loss e mesmo gradiente, inclusive com mascara desigual."""
    data_dir = request.getfixturevalue(dataset)
    torch.manual_seed(0)
    model = SLM(ModelConfig(**tiny_model_cfg))
    single = TrainConfig(run_dir=str(tmp_path), data_dir=str(data_dir), batch_size=24, grad_accum=1, use_loss_mask=use_mask)
    accum = TrainConfig(run_dir=str(tmp_path), data_dir=str(data_dir), batch_size=4, grad_accum=6, use_loss_mask=use_mask)

    loss_a, grads_a = _gradients(model, single, data_dir, use_mask)
    loss_b, grads_b = _gradients(model, accum, data_dir, use_mask)
    assert loss_b.item() == pytest.approx(loss_a.item(), rel=1e-5)
    for name in grads_a:
        torch.testing.assert_close(grads_b[name], grads_a[name], rtol=1e-4, atol=1e-6, msg=name)


def test_naive_average_would_differ(tmp_path, tiny_model_cfg, uneven_masked_dataset):
    """Controle do teste acima: a media simples das losses dos micro-lotes NAO bate com o lote unico."""
    torch.manual_seed(0)
    model = SLM(ModelConfig(**tiny_model_cfg))
    data = RecordDataset(uneven_masked_dataset, use_mask=True)
    sampler = EpochSampler(len(data), 0)
    inputs, targets = data.batch(sampler.indices(0, 24))
    _, single = model(torch.from_numpy(inputs), torch.from_numpy(targets))
    naive = []
    for k in range(6):
        inputs, targets = data.batch(sampler.indices(4 * k, 4))
        naive.append(model(torch.from_numpy(inputs), torch.from_numpy(targets))[1])
    assert abs(torch.stack(naive).mean().item() - single.item()) > 1e-3


def test_short_training_matches_single_batch(tmp_path, tiny_dataset, tiny_model_cfg):
    """Treino inteiro (LR, WSD, clipping, checkpoints): batch 8 x 1 contra batch 2 x 4."""
    single = tmp_path / "single"
    accum = tmp_path / "accum"
    train(make_config(single, tiny_dataset, tiny_model_cfg, batch_size=8, max_tokens=16 * 8 * 20, ckpt_every=50))
    train(make_config(accum, tiny_dataset, tiny_model_cfg, batch_size=2, grad_accum=4, max_tokens=16 * 8 * 20, ckpt_every=50))

    a = checkpoint.load(single / "final.pt")
    b = checkpoint.load(accum / "final.pt")
    assert a["step"] == b["step"] == 20
    assert a["records_seen"] == b["records_seen"] == 160
    for name, tensor in a["model"].items():
        torch.testing.assert_close(b["model"][name], tensor, rtol=1e-3, atol=1e-5, msg=name)
    loss = lambda run: {r["step"]: r["loss"] for r in read_metrics(run) if "loss" in r}
    assert loss(accum) == pytest.approx(loss(single), rel=1e-4)
    val = lambda run: {r["step"]: r["val_loss"] for r in read_metrics(run) if "val_loss" in r}
    assert val(accum) == pytest.approx(val(single), rel=1e-4)


def test_resume_with_accumulation(tmp_path, tiny_dataset, tiny_model_cfg):
    """Interromper e retomar com acumulo chega aos mesmos pesos de um treino sem interrupcao."""
    straight = tmp_path / "straight"
    train(make_config(straight, tiny_dataset, tiny_model_cfg, batch_size=2, grad_accum=2))

    resumed = tmp_path / "resumed"
    cfg = make_config(resumed, tiny_dataset, tiny_model_cfg, batch_size=2, grad_accum=2)
    assert train(cfg, stop_after_steps=17) == "interrupted"
    assert checkpoint.load(resumed / "ckpt_000000017.pt")["records_seen"] == 17 * 4
    assert train(cfg) == "finished"

    a = checkpoint.load(straight / "final.pt")
    b = checkpoint.load(resumed / "final.pt")
    for name, tensor in a["model"].items():
        assert torch.equal(tensor, b["model"][name]), name


def test_invalid_grad_accum(tmp_path, tiny_dataset, tiny_model_cfg):
    with pytest.raises(ValueError, match="grad_accum"):
        train(make_config(tmp_path / "x", tiny_dataset, tiny_model_cfg, grad_accum=0))
