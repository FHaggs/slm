import numpy as np

from slm.chat import merge_system, pack, render, render_prompt
from slm.data import EpochSampler, RecordDataset, evenly_spaced_indices, make_targets, write_records
from slm.schedule import wsd_lr
from slm.tokenizer import ASSISTANT, END, ENDOFTEXT, PAD, USER, VOCAB_SIZE, Tokenizer

CONVERSATION = [
    {"role": "user", "content": "What is 2 + 2?"},
    {"role": "assistant", "content": "It is 4."},
    {"role": "user", "content": "Thanks!"},
    {"role": "assistant", "content": "You are welcome."},
]


def test_tokenizer_round_trip_and_specials():
    tokenizer = Tokenizer()
    text = "Olá, mundo! 12345 <|end|>"
    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert END not in tokenizer.encode("<|end|>")
    assert tokenizer.decode([USER, ASSISTANT, END, PAD]) == "<|user|><|assistant|><|end|><|pad|>"
    assert PAD == VOCAB_SIZE - 1


def test_digit_split_only_changes_numbers():
    plain, split = Tokenizer(), Tokenizer(split_digits=True)
    text = "Paid 1234 dollars, then 56."
    tokens = split.encode(text)
    assert split.decode(tokens) == text
    assert [split.decode([t]) for t in tokens if split.decode([t]).isdigit()] == list("123456")
    assert split.encode("no numbers here") == plain.encode("no numbers here")
    assert split.encode_batch([text, text]) == [tokens, tokens]


def test_merge_system_goes_into_first_user_turn():
    merged = merge_system(
        [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
    )
    assert [m["role"] for m in merged] == ["user", "assistant"]
    assert merged[0]["content"] == "Be brief.\n\nHi"


def test_render_template_and_mask():
    tokenizer = Tokenizer()
    tokens, mask = render(CONVERSATION, tokenizer)
    assert tokens[0] == ENDOFTEXT
    assert tokens[1] == USER
    assert tokens[-1] == END
    assert len(tokens) == len(mask)
    # A mascara cobre so as respostas e o <|end|> delas, nunca a fala do usuario nem <|assistant|>.
    masked = tokenizer.decode(tokens[mask == 1].tolist())
    assert masked == "It is 4.<|end|>You are welcome.<|end|>"


def test_targets_are_shifted_and_masked_for_sft():
    """Teste central do SFT: os alvos validos sao exatamente as respostas do assistente."""
    tokenizer = Tokenizer()
    tokens, mask = render(CONVERSATION, tokenizer)
    records, masks = pack([(tokens, mask)], record_len=64)
    inputs, targets = make_targets(records, masks)

    assert inputs.shape == targets.shape == (1, 63)
    valid = targets[0] != -100
    assert tokenizer.decode(targets[0][valid].tolist()) == "It is 4.<|end|>You are welcome.<|end|>"
    # A primeira resposta precisa ser prevista a partir do token <|assistant|>.
    first_valid = int(np.argmax(valid))
    assert inputs[0, first_valid] == ASSISTANT
    # Cada alvo e o token seguinte da entrada, salvo no padding do fim.
    length = len(tokens)
    assert (inputs[0, 1 : length - 1] == records[0, 1 : length - 1]).all()


def test_targets_without_mask_cover_everything_except_padding():
    tokenizer = Tokenizer()
    tokens, mask = render(CONVERSATION, tokenizer)
    records, _ = pack([(tokens, mask)], record_len=64)
    _, targets = make_targets(records, None)
    length = len(tokens)
    assert (targets[0, : length - 1] == tokens[1:]).all()
    assert (targets[0, length - 1 :] == -100).all()


def test_pack_keeps_conversations_whole_and_truncates_long_ones():
    a = (np.arange(1, 11, dtype=np.uint16), np.ones(10, dtype=np.uint8))
    b = (np.arange(11, 19, dtype=np.uint16), np.ones(8, dtype=np.uint8))
    c = (np.arange(20, 60, dtype=np.uint16), np.ones(40, dtype=np.uint8))
    no_answer = (np.arange(5, dtype=np.uint16), np.zeros(5, dtype=np.uint8))
    records, masks = pack([a, b, no_answer, c], record_len=20)
    assert records.shape == (2, 20)
    assert records[0, :18].tolist() == list(range(1, 19))
    assert (records[0, 18:] == PAD).all()
    assert records[1].tolist() == list(range(20, 40))
    assert masks.sum() == 38


def test_render_prompt_ends_ready_for_the_answer():
    tokenizer = Tokenizer()
    prompt = render_prompt(CONVERSATION[:1], tokenizer)
    assert prompt[-1] == ASSISTANT
    assert prompt[-2] == END


def test_record_dataset_round_trip(tmp_path):
    tokens = np.arange(40, dtype=np.uint16).reshape(4, 10)
    mask = np.ones_like(tokens, dtype=np.uint8)
    mask[:, :5] = 0
    write_records(tmp_path, tokens, mask, split_digits=False)
    dataset = RecordDataset(tmp_path, use_mask=True)
    assert len(dataset) == 4 and dataset.seq_len == 9
    inputs, targets = dataset.batch(np.array([2]))
    assert inputs[0].tolist() == list(range(20, 29))
    assert targets[0].tolist() == [-100, -100, -100, -100, 25, 26, 27, 28, 29]


def test_evenly_spaced_indices():
    assert evenly_spaced_indices(100, 4).tolist() == [0, 25, 50, 75]
    assert evenly_spaced_indices(3, 10).tolist() == [0, 1, 2]
    assert len(set(evenly_spaced_indices(1000, 999).tolist())) == 999


def test_sampler_covers_every_record_once_per_epoch():
    sampler = EpochSampler(100, seed=1)
    first_epoch = sampler.indices(0, 100)
    second_epoch = sampler.indices(100, 100)
    assert sorted(first_epoch.tolist()) == list(range(100))
    assert sorted(second_epoch.tolist()) == list(range(100))
    assert first_epoch.tolist() != second_epoch.tolist()


def test_sampler_is_deterministic_and_shards_between_ranks():
    whole = EpochSampler(100, seed=1).indices(0, 20)
    rank0 = EpochSampler(100, seed=1).indices(0, 10, stride=2)
    rank1 = EpochSampler(100, seed=1).indices(1, 10, stride=2)
    assert rank0.tolist() == whole[::2].tolist()
    assert rank1.tolist() == whole[1::2].tolist()
    assert not set(rank0.tolist()) & set(rank1.tolist())


def test_wsd_schedule_shape():
    lr = lambda t: wsd_lr(t, total_tokens=1000, peak_lr=1.0, warmup_tokens=100, decay_frac=0.1)
    assert lr(0) < 0.02
    assert abs(lr(50) - 0.51) < 1e-9
    assert lr(100) == lr(500) == lr(899) == 1.0
    assert abs(lr(950) - 0.5) < 1e-9
    assert lr(1000) == 0.0
