import importlib.util
import numpy as np
import pytest
from protosurv.graph import coordinate_knn_edges, physical_coordinates, select_degree_matched_radius
from scripts.rasterize_annotations import patch_annotations
from scripts.evaluate_localization import _slide_metrics


def test_fractional_reference_area_is_not_a_hard_patch_label():
    mask = np.array([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=bool)
    fractions, area, tissue = patch_annotations(np.array([[1, 1], [3, 1]]), mask,
                                               np.ones_like(mask), 2, 1)
    np.testing.assert_allclose(fractions, [0.5, 0])
    result = _slide_metrics(np.array([1., 0.]), fractions, tissue, 0.5, area)
    assert result['iou'] == 0.5
    assert result['precision_at_20'] == 0.5


def test_incomplete_reference_coverage_is_excluded():
    mask = np.ones((2, 4), dtype=bool)
    valid = mask.copy(); valid[0, 0] = False
    _, _, tissue = patch_annotations(np.array([[1, 1], [3, 1]]), mask, valid, 2, 1)
    assert tissue.tolist() == [False, True]


def test_physical_radius_matching_uses_training_coordinates():
    grid = np.array([(x, y) for x in range(4) for y in range(4)])
    physical = physical_coordinates(grid, [0.5, 0.5])
    radius = select_degree_matched_radius([physical], k=4)
    assert 0.5 <= radius <= 2.2
    duplicate = coordinate_knn_edges(np.zeros((5, 2)), k=2)
    assert not np.any(duplicate[0] == duplicate[1])


@pytest.mark.skipif(importlib.util.find_spec('torch') is None, reason='PyTorch required')
def test_normalize_per_model_before_attention_ensemble():
    import torch
    from scripts.export_attention import normalized_class_attention
    a = normalized_class_attention(torch.tensor([[0., 10., 20.], [1., 1., 1.]]), 2, 1)
    b = normalized_class_attention(torch.tensor([[0., 1., 1.], [0., 1., 2.]]), 2, 1)
    torch.testing.assert_close(((a+b)/2)[:, 0], torch.tensor([0., 0.75, 1.]))
    assert torch.all(a[:, 1] == 0)


@pytest.mark.skipif(importlib.util.find_spec('torch') is None, reason='PyTorch required')
def test_cox_tied_events_share_breslow_risk_set():
    import torch
    from protosurv.losses import cox_partial_likelihood
    risk = torch.log(torch.tensor([2., 3., 5.])).requires_grad_()
    loss = cox_partial_likelihood(risk, torch.tensor([1., 1., 2.]), torch.tensor([1., 1., 0.]))
    expected = np.log(10.) - np.log(6.)/2
    assert float(loss.detach()) == pytest.approx(expected, abs=1e-6)
    loss.backward()
    assert torch.isfinite(risk.grad).all()
