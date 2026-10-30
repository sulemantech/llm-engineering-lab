import math

import pytest
import torch

from llmlab.bigram import BigramLM
from llmlab.data import get_batch
from llmlab.tokenizer import CharTokenizer

pytestmark = pytest.mark.stage1

CORPUS = "hello world, hello transformer"


def test_tokenizer_round_trip():
    tok = CharTokenizer(CORPUS)
    assert tok.vocab_size == len(set(CORPUS))
    assert tok.decode(tok.encode("hello")) == "hello"


def test_tokenizer_ids_in_range():
    tok = CharTokenizer(CORPUS)
    ids = tok.encode(CORPUS)
    assert min(ids) == 0 and max(ids) == tok.vocab_size - 1


def test_tokenizer_unknown_char_raises():
    with pytest.raises(KeyError):
        CharTokenizer(CORPUS).encode("Z")


def test_targets_are_shifted_inputs():
    data = torch.arange(100)
    x, y = get_batch(data, batch_size=4, block_size=8, generator=torch.Generator().manual_seed(0))
    assert x.shape == y.shape == (4, 8)
    assert x.dtype == torch.long
    assert torch.equal(y, x + 1)  # with arange data, "next token" == value + 1


def test_bigram_shapes_and_initial_loss():
    torch.manual_seed(0)
    V = 27
    model = BigramLM(V)
    idx = torch.randint(0, V, (4, 8))
    logits, loss = model(idx, idx)
    assert logits.shape == (4, 8, V)
    # Near-uniform predictions at init -> loss close to ln(V). Large gap = bad init.
    assert abs(loss.item() - math.log(V)) < 1.5
    assert model(idx)[1] is None


def test_bigram_overfits_one_batch():
    torch.manual_seed(0)
    model = BigramLM(5)
    x = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2]])
    y = torch.tensor([[1, 2, 3, 4, 0, 1, 2, 3]])
    opt = torch.optim.AdamW(model.parameters(), lr=0.1)
    for _ in range(300):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.05


def test_bigram_generate_length():
    out = BigramLM(10).generate(torch.zeros(2, 1, dtype=torch.long), max_new_tokens=5)
    assert out.shape == (2, 6)
