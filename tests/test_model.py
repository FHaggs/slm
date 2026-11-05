import torch

from slm.model import SLM, ModelConfig


def test_final_config_parameter_count():
    # Valor conferido no benchmark; embeddings compartilhados contam uma so vez.
    cfg = ModelConfig(d_model=768, n_layer=12, n_head=12, n_kv_head=4, ffn_hidden=2048)
    with torch.device("meta"):
        model = SLM(cfg)
    assert sum(p.numel() for p in model.parameters()) == 114_117_120
    assert 50_000_000 <= model.num_params() <= 200_000_000


def test_embeddings_are_tied(tiny_model_config):
    model = SLM(tiny_model_config)
    assert model.lm_head.weight is model.tok_emb.weight


def test_output_shape_and_initial_loss(tiny_model_config):
    torch.manual_seed(0)
    model = SLM(tiny_model_config)
    # Alvos independentes da entrada: com embeddings compartilhados, repetir a entrada reduziria a loss.
    tokens = torch.randint(0, tiny_model_config.vocab_size, (8, 16))
    targets = torch.randint(0, tiny_model_config.vocab_size, (8, 16))
    logits, loss = model(tokens, targets)
    assert logits.shape == (8, 16, tiny_model_config.vocab_size)
    assert abs(loss.item() - torch.log(torch.tensor(50_261.0)).item()) < 0.1


def test_attention_is_causal(tiny_model_config):
    """Mudar um token nao pode alterar a previsao das posicoes anteriores."""
    model = SLM(tiny_model_config).eval()
    tokens = torch.randint(0, 1000, (1, 16))
    changed = tokens.clone()
    changed[0, 10] = 2000
    original, _ = model(tokens)
    modified, _ = model(changed)
    assert torch.allclose(original[0, :10], modified[0, :10])
    assert not torch.allclose(original[0, 10:], modified[0, 10:])


def test_position_matters(tiny_model_config):
    """Com RoPE, a mesma sequencia em outra ordem precisa gerar saidas diferentes."""
    model = SLM(tiny_model_config).eval()
    forward, _ = model(torch.tensor([[1, 2, 3, 4]]))
    backward, _ = model(torch.tensor([[4, 3, 2, 1]]))
    assert not torch.allclose(forward[0, -1], backward[0, -1])


def test_ignored_targets_do_not_change_loss(tiny_model_config):
    model = SLM(tiny_model_config).eval()
    tokens = torch.randint(0, 1000, (2, 16))
    targets = torch.randint(0, 1000, (2, 16))
    targets[:, 8:] = -100
    logits, loss = model(tokens, targets)
    expected = torch.nn.functional.cross_entropy(logits[:, :8].reshape(-1, logits.size(-1)), targets[:, :8].reshape(-1))
    assert torch.isclose(loss, expected)


def test_overfits_one_batch():
    """O teste sugerido no enunciado: um modelo pequeno precisa decorar um mini-batch."""
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=64, d_model=64, n_layer=2, n_head=4, n_kv_head=2, ffn_hidden=128, max_seq_len=32)
    model = SLM(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    data = torch.randint(0, cfg.vocab_size, (4, 33))
    first_loss = None
    for _ in range(300):
        optimizer.zero_grad()
        _, loss = model(data[:, :-1], data[:, 1:])
        loss.backward()
        optimizer.step()
        first_loss = first_loss or loss.item()
    assert first_loss > 3.5
    assert loss.item() < 0.05
