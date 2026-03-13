import torch

from main_classical import SimpleCNN


def test_classical_model_forward_smoke():
    """Create a tiny SimpleCNN and run a single forward pass.

    This is a pure smoke test: it only checks that the forward pass runs and
    that the output has the expected shape, not that training converges.
    """

    model = SimpleCNN(img_ch=1, num_classes=26)
    x = torch.randn(1, 1, 32, 32)
    out = model(x)
    assert out.shape == (1, 26)
