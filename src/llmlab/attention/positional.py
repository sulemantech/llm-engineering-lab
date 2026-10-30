"""Stage 2: sinusoidal positional encoding (paper section 3.5)."""

import torch


def sinusoidal_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """(max_len, d_model) with PE[pos, 2i] = sin(pos / 10000^(2i/d_model)),
    PE[pos, 2i+1] = cos(pos / 10000^(2i/d_model))."""
    raise NotImplementedError
