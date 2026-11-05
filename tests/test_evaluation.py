from pathlib import Path

import pytest
import torch

from slm import checkpoint
from slm.evaluation import (
    arithmetic_items,
    choice_log_likelihoods,
    clean_gsm_solution,
    gsm_answer,
    gsm_few_shot_prefix,
    leading_number,
    mcnemar_exact,
    run_arithmetic,
    run_gsm8k,
    run_multiple_choice,
    same_number,
    wilson_interval,
)
from slm.generate import greedy_batch, stream
from slm.model import SLM, ModelConfig
from slm.tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parent.parent


def test_clean_gsm_solution():
    raw = "Natalia sold 48/2 = <<48/2=24>>24 clips.\n#### 24"
    assert clean_gsm_solution(raw) == "Natalia sold 48/2 = 24 clips.\n#### 24"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("She has 3 and 5 left.\n#### 1,200", 1200.0),
        ("So 2 + 2 = 4", 4.0),
        ("answer 7\n\nQuestion: 99 apples", 7.0),
        ("no number at all", None),
        ("#### -3.5", -3.5),
    ],
)
def test_gsm_answer(text, expected):
    assert gsm_answer(text) == expected


def test_no_answer_is_never_correct():
    # O bug do nanochat: comparar None com None contava como acerto.
    assert not same_number(None, None)
    assert not same_number(None, 3.0)
    assert same_number(1200.0, 1200.0)


def test_leading_number_ignores_later_numbers():
    assert leading_number(" 42\nmore 7") == 42
    assert leading_number(" 1,024 is") == 1024
    assert leading_number(" the answer is 42") is None


def test_arithmetic_items_are_correct_and_reproducible():
    items = arithmetic_items()
    assert items == arithmetic_items()
    assert {item["cell"] for item in items} == {"+_1d", "+_2d", "-_1d", "-_2d", "*_1d", "*_2d"}
    assert sum(item["cell"] == "*_1d" for item in items) == 81
    assert sum(item["cell"] == "-_1d" for item in items) == 45
    assert sum(item["cell"] == "+_2d" for item in items) == 300
    for item in items:
        left, symbol, right, _ = item["prompt"].split()
        assert eval(f"{left}{symbol}{right}") == item["answer"]
        assert item["answer"] >= 0


def test_mcnemar_exact():
    assert mcnemar_exact([True, False], [True, False])["p_value"] == 1.0
    result = mcnemar_exact([False] * 10, [True] * 10)
    assert result["only_second"] == 10
    assert result["p_value"] == pytest.approx(2 / 2**10)
    balanced = mcnemar_exact([True] * 5 + [False] * 5, [False] * 5 + [True] * 5)
    assert balanced["p_value"] == 1.0


def test_wilson_interval_contains_the_rate():
    low, high = wilson_interval(26, 1319)
    assert low < 26 / 1319 < high
    assert 0.012 < low and high < 0.03


class AnswerModel(SLM):
    """Modelo falso cujo proximo token e sempre um valor fixo."""

    def __init__(self, cfg: ModelConfig, token: int):
        super().__init__(cfg)
        self.token = token

    def forward(self, tokens, targets=None):
        logits = torch.zeros(*tokens.shape, self.cfg.vocab_size)
        logits[..., self.token] = 10.0
        return logits, None


def test_greedy_and_stream_stop_tokens(tiny_model_config):
    model = AnswerModel(tiny_model_config, token=7)
    prompts = torch.tensor([[1, 2], [3, 4]])
    assert greedy_batch(model, prompts, 3).tolist() == [[7, 7, 7], [7, 7, 7]]
    assert list(stream(model, [1, 2], 3, temperature=0)) == [7, 7, 7]
    assert list(stream(model, [1, 2], 3, temperature=0, stop_tokens=frozenset({7}))) == []
    # A amostragem respeita top-k = 1, e o contexto nunca passa de max_seq_len.
    assert list(stream(model, list(range(30)), 2, temperature=1.0, top_k=1)) == [7, 7]


def test_arithmetic_scoring_with_known_answers(tiny_model_config):
    tokenizer = Tokenizer()
    # Sempre responde " 5": so os itens cujo resultado e 5 podem contar como acerto.
    answer_token = tokenizer.encode(" 5")[0]
    model = AnswerModel(tiny_model_config, token=answer_token)
    items = [
        {"cell": "+_1d", "prompt": "2 + 3 =", "answer": 5},
        {"cell": "+_1d", "prompt": "2 + 4 =", "answer": 6},
        {"cell": "*_1d", "prompt": "1 * 5 =", "answer": 5},
    ]
    # O texto " 5 5 5..." vira um numero so se nao houver espaco: usa-se um token por geracao.
    result = run_arithmetic(model, tokenizer, items, max_new_tokens=1)
    assert result["correct"] == [True, False, True]
    assert result["cells"] == {"+_1d": 0.5, "*_1d": 1.0}


def test_gsm_scoring_counts_unanswered_as_wrong(tiny_model_config):
    tokenizer = Tokenizer()
    model = AnswerModel(tiny_model_config, token=tokenizer.encode(" hello")[0])
    rows = [{"question": "How many?", "answer": "Because.\n#### 3"}]
    result = run_gsm8k(model, tokenizer, rows, gsm_few_shot_prefix(rows, shots=1), max_new_tokens=4)
    assert result["correct"] == [False]
    assert result["answered"] == 0.0


class PreferEndingModel(SLM):
    """Da alta probabilidade ao token favorito em qualquer contexto."""

    def __init__(self, cfg: ModelConfig, favorite: int):
        super().__init__(cfg)
        self.favorite = favorite

    def forward(self, tokens, targets=None):
        logits = torch.zeros(*tokens.shape, self.cfg.vocab_size)
        logits[..., self.favorite] = 12.0
        return logits, None


def test_multiple_choice_picks_highest_likelihood(tiny_model_config):
    tokenizer = Tokenizer()
    good = tokenizer.encode(" yes")
    model = PreferEndingModel(tiny_model_config, favorite=good[0])
    items = [
        {"context": "Is it?", "endings": [" no", " yes", " maybe"], "label": 1},
        {"context": "Is it?", "endings": [" no", " yes"], "label": 0},
    ]
    result = run_multiple_choice(model, tokenizer, items)
    assert result["correct"] == [True, False]
    assert result["accuracy"] == 0.5


def test_choice_scores_ignore_padding_and_crop_long_context(tiny_model_config):
    torch.manual_seed(0)
    model = SLM(tiny_model_config).eval()
    context = [1, 2, 3]
    short, long = [4], [5, 6, 7, 8]
    alone = choice_log_likelihoods(model, context, [short])[0]
    together = choice_log_likelihoods(model, context, [short, long])[0]
    assert alone[0] == pytest.approx(together[0], rel=1e-5)
    # Contexto maior que o modelo: nao pode falhar nem exceder max_seq_len.
    assert len(choice_log_likelihoods(model, list(range(100)), [short, long])) == 2


def test_load_model_restores_weights_and_tokenizer(tmp_path, tiny_model_config):
    model = SLM(tiny_model_config)
    path = tmp_path / "model.pt"
    checkpoint.save(path, model, split_digits=True)
    loaded, tokenizer = checkpoint.load_model(path)
    assert tokenizer.split_digits is True
    assert torch.equal(loaded.tok_emb.weight, model.tok_emb.weight)
    assert loaded.lm_head.weight is loaded.tok_emb.weight


def test_demo_answers_in_chat(tmp_path, tiny_model_config, monkeypatch):
    from streamlit.testing.v1 import AppTest

    path = tmp_path / "model.pt"
    checkpoint.save(path, SLM(tiny_model_config), split_digits=False)
    monkeypatch.setenv("SLM_CHECKPOINT", str(path))

    app = AppTest.from_file(str(ROOT / "app/chat.py"), default_timeout=60).run()
    assert not app.exception
    assert [slider.label for slider in app.sidebar.slider] == ["Temperatura", "Top-k", "Máximo de tokens"]

    app.chat_input[0].set_value("Hello").run()
    assert not app.exception
    assert app.session_state.messages[0] == {"role": "user", "content": "Hello"}
    assert app.session_state.messages[1]["role"] == "assistant"
    assert [message.name for message in app.chat_message] == ["user", "assistant"]
