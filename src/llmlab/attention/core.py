"""Stage 2: scaled dot-product attention, softmax(Q K^T / sqrt(d_k)) V."""

import torch


def causal_mask(seq_len: int, device: torch.device | str | None = None) -> torch.Tensor:
    """Boolean (seq_len, seq_len) mask, True where attention is ALLOWED (j <= i)."""
    raise NotImplementedError


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """q: (..., T_q, d_k), k: (..., T_k, d_k), v: (..., T_k, d_v).

    mask: boolean, broadcastable to (..., T_q, T_k); True = attend, False = blocked.
    Returns (output (..., T_q, d_v), attention weights (..., T_q, T_k)).
    """
    raise NotImplementedError
