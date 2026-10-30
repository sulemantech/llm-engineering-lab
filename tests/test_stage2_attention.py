import math

import pytest
import torch
import torch.nn.functional as F

from llmlab.attention import (
    MultiHeadAttention,
    causal_mask,
    scaled_dot_product_attention,
    sinusoidal_positional_encoding,
)

pytestmark = pytest.mark.stage2


def test_hand_computed_example():
    # One query identical to the first of two keys; d_k = 1.
    q = torch.tensor([[[1.0]]])
    k = torch.tensor([[[1.0], [0.0]]])
    v = torch.tensor([[[10.0], [20.0]]])
    out, w = scaled_dot_product_attention(q, k, v)
    expected_w = torch.softmax(torch.tensor([1.0, 0.0]), dim=0)
    assert torch.allclose(w[0, 0], expected_w)
    assert torch.allclose(out[0, 0, 0], expected_w @ torch.tensor([10.0, 20.0]))


@pytest.mark.parametrize("B,T,d", [(1, 1, 4), (2, 5, 8), (3, 16, 32)])
def test_matches_pytorch_reference(B, T, d):
    torch.manual_seed(0)
    q, k, v = (torch.randn(B, T, d) for _ in range(3))
    out, w = scaled_dot_product_attention(q, k, v)
    assert out.shape == (B, T, d) and w.shape == (B, T, T)
    assert torch.allclose(out, F.scaled_dot_product_attention(q, k, v), atol=1e-5)
    assert torch.allclose(w.sum(-1), torch.ones(B, T), atol=1e-5)


def test_causal_mask_orientation():
    m = causal_mask(4)
    assert m.dtype == torch.bool and m.shape == (4, 4)
    assert m[3, 0] and not m[0, 3]  # later positions see earlier ones, never the reverse


def test_causal_matches_pytorch_reference():
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 7, 8) for _ in range(3))
    out, w = scaled_dot_product_attention(q, k, v, mask=causal_mask(7))
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(out, ref, atol=1e-5)
    assert torch.all(w.triu(diagonal=1) == 0)


def test_padding_mask_blocks_keys():
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 4, 8) for _ in range(3))
    pad = torch.tensor([True, True, True, False]).view(1, 1, 4)  # last key is padding
    _, w = scaled_dot_product_attention(q, k, v, mask=pad)
    assert torch.all(w[..., 3] == 0)


@pytest.mark.parametrize("T", [5, 12])
def test_mha_no_future_leakage(T):
    """Changing token t+1 must not change outputs at positions <= t."""
    torch.manual_seed(0)
    mha = MultiHeadAttention(d_model=16, n_heads=4).eval()
    x = torch.randn(1, T, 16)
    t = T // 2
    x2 = x.clone()
    x2[:, t + 1 :] = torch.randn_like(x2[:, t + 1 :])
    mask = causal_mask(T)
    assert torch.allclose(mha(x, mask=mask)[:, : t + 1], mha(x2, mask=mask)[:, : t + 1])


def test_mha_shapes_and_gradients():
    torch.manual_seed(0)
    mha = MultiHeadAttention(d_model=32, n_heads=4)
    x_q, x_kv = torch.randn(2, 5, 32), torch.randn(2, 9, 32)
    out = mha(x_q, x_kv)  # cross-attention: different lengths
    assert out.shape == (2, 5, 32)
    out.sum().backward()
    for name, p in mha.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name


def test_mha_rejects_indivisible_heads():
    with pytest.raises((AssertionError, ValueError)):
        MultiHeadAttention(d_model=30, n_heads=4)


def test_sinusoidal_pe():
    pe = sinusoidal_positional_encoding(max_len=50, d_model=16)
    assert pe.shape == (50, 16)
    assert torch.allclose(pe[0, 0::2], torch.zeros(8))  # sin(0)
    assert torch.allclose(pe[0, 1::2], torch.ones(8))  # cos(0)
    assert pe[7, 0].item() == pytest.approx(math.sin(7.0), abs=1e-5)
    assert pe.abs().max() <= 1.0 + 1e-6
