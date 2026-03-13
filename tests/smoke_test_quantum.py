import torch

from main_quantum import QuantumOnlyModel


def test_quantum_model_forward_smoke():
    """Create a tiny QuantumOnlyModel and run a single forward pass.

    This is a pure smoke test: it only checks that the forward pass runs and
    that the output has the expected shape, not that training converges.
    """

    model = QuantumOnlyModel(
        num_classes=26,
        img_ch=1,
        img_size=32,
        latent_dim=4,
        n_qubits=2,
        n_layers=1,
        q_output=8,
    )

    x = torch.randn(1, 1, 32, 32)
    out = model(x)

    assert out.shape == (1, 26)
