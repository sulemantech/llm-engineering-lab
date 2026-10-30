"""Stage 1: turning a token stream into (input, target) batches."""

import torch


def get_batch(
    data: torch.Tensor, batch_size: int, block_size: int, generator: torch.Generator | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample `batch_size` random windows from a 1-D tensor of token ids.

    Returns (x, y), both of shape (batch_size, block_size), dtype long, where
    y[b, t] is the token that follows x[b, t] in `data` (next-token targets).
    """
    raise NotImplementedError
