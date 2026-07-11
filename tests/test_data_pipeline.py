"""Tests for the data pipeline: formatters, WeightedMultiplex, PackedTokenStream.

Uses a trivial FakeTokenizer instead of downloading GPT-2, so these tests need
no network access. Pure CPU, fast.
"""

import colab_trainable_dendritic_lm as m


class FakeTokenizer:
    """Deterministic 1-char-per-token stand-in; no download, no vocab file."""

    eos_token_id = -1

    def encode(self, text):
        return [ord(c) for c in text]


def test_formatters_prose_reads_text_field():
    formatters = m.make_formatters()
    r = formatters["math"]({"text": "some prose"})
    assert r == (False, "", "some prose")


def test_formatters_prose_returns_none_on_missing_field():
    formatters = m.make_formatters()
    assert formatters["math"]({}) is None


def test_formatters_coedit_grammar_pair():
    formatters = m.make_formatters()
    r = formatters["grammar"]({"src": "fix this sentance", "tgt": "fix this sentence"})
    assert r == (True, "fix this sentance", "fix this sentence")


def test_formatters_planning_reuses_prose_on_synthetic_rows():
    formatters = m.make_formatters()
    row = next(m.BlocksworldPlanningStream(seed=1))
    r = formatters["planning"](row)
    assert r[0] is False  # not an instruction row -- plain continuation
    assert r[2] == row["text"]


def test_encode_example_base_pretraining_fully_supervised():
    formatters = m.make_formatters()
    tok = FakeTokenizer()
    ids, mask = m.encode_example(
        "math", {"text": "hello"}, formatters, tok, mask_prompt=True, chat_format=False
    )
    assert len(ids) == len(mask) == len("hello") + 1  # + eos
    assert all(bit == 1 for bit in mask)  # prose is always fully supervised


def test_encode_example_chat_format_masks_prompt():
    formatters = m.make_formatters()
    tok = FakeTokenizer()
    ids, mask = m.encode_example(
        "grammar", {"src": "hi", "tgt": "bye"}, formatters, tok, mask_prompt=True, chat_format=True
    )
    assert 0 in mask and 1 in mask  # prompt tokens masked, response tokens supervised
    assert mask[-1] == 1  # eos token (end of response) is supervised


def test_weighted_multiplex_cycles_exhausted_finite_source():
    src_a = [{"v": 1}, {"v": 2}]
    src_b = [{"v": "b"}]
    mux = m.WeightedMultiplex([src_a, src_b], [0.999, 0.001], ["a", "b"], seed=0)
    seen = [next(mux) for _ in range(10)]  # far more than len(src_a), forces a cycle
    assert all(name in ("a", "b") for name, _ in seen)


def test_packed_token_stream_produces_full_blocks():
    formatters = m.make_formatters()
    tok = FakeTokenizer()
    rows = [{"text": "x" * 50} for _ in range(20)]
    mux = m.WeightedMultiplex([rows], [1.0], ["math"], seed=0)
    stream = m.PackedTokenStream(mux, formatters, tok, seq_len=16, mask_prompt=False, chat_format=False)
    x, y = stream.get_block("cpu")
    assert x.shape == (1, 16)
    assert y.shape == (1, 16)
