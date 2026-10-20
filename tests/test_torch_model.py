import importlib.util

import pytest


torch_available = importlib.util.find_spec("torch") is not None


@pytest.mark.skipif(not torch_available, reason="PyTorch is not installed")
def test_model_forward_shapes():
    import torch

    from protosurv.model import PriorGuidedSurvival

    torch.manual_seed(2)
    model = PriorGuidedSurvival(
        input_dim=16,
        hidden_dim=8,
        prototype_dim=16,
        projection_dim=8,
        num_classes=5,
        num_prototypes=2,
        num_gcn_layers=2,
        num_attention_heads=4,
        dropout=0.0,
    )
    features = torch.randn(7, 16)
    priors = torch.softmax(torch.randn(7, 5), dim=-1)
    coords = torch.rand(7, 2)
    output = model(features, priors, coords=coords)
    assert output.risk.ndim == 0
    assert output.prototypes.shape == (10, 16)
    assert output.attention.shape == (10, 7)
    assert torch.isfinite(output.alignment_loss)
    assert torch.isfinite(output.orthogonality_loss)

