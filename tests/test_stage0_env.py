import pytest
import torch

pytestmark = pytest.mark.stage0


def test_matmul_shape():
    a, b = torch.randn(2, 3, 4), torch.randn(4, 5)
    assert (a @ b).shape == (2, 3, 5)


def test_broadcasting():
    assert (torch.ones(3, 1) + torch.ones(1, 4)).shape == (3, 4)


def test_autograd_scalar():
    x = torch.tensor(3.0, requires_grad=True)
    (x**2 + 2 * x).backward()
    assert x.grad.item() == pytest.approx(8.0)  # d/dx = 2x + 2
