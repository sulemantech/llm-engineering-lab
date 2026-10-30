"""Stage 1: bigram language model, the simplest neural LM baseline."""

import torch
from torch import nn


class BigramLM(nn.Module):
    """Each token directly looks up the logits for the next token.

    forward(idx (B,T), targets (B,T) | None) -> (logits (B,T,V), loss scalar | None)
    generate(idx (B,T), max_new_tokens) -> (B, T + max_new_tokens)
    """

    def __init__(self, vocab_size: int):
        super().__init__()
        raise NotImplementedError

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        raise NotImplementedError

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        raise NotImplementedError
