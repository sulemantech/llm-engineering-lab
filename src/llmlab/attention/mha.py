"""Stage 2: multi-head attention (paper section 3.2.2)."""

import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    """x_q (B,T_q,d_model), x_kv (B,T_kv,d_model) -> (B,T_q,d_model).

    Self-attention when x_kv is None. Uses llmlab.attention.core.scaled_dot_product_attention.
    d_model must be divisible by n_heads.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        raise NotImplementedError

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError
