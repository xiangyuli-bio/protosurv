import numpy as np
import pandas as pd

from protosurv.graph import coordinate_knn_edges, coordinate_radius_edges
from protosurv.metrics import concordance_index
from protosurv.priors import normalize_class_embeddings, soft_priors
from protosurv.splits import patient_folds


def test_soft_priors_are_normalized_and_nonnegative():
    rng = np.random.default_rng(4)
    image = rng.normal(size=(12, 6)).astype(np.float32)
    prompts = rng.normal(size=(5, 3, 6)).astype(np.float32)
    classes = normalize_class_embeddings(prompts)
    priors = soft_priors(image, classes)
    assert priors.shape == (12, 5)
    np.testing.assert_allclose(priors.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(priors >= 0)


def test_knn_edges_are_symmetric_without_self_edges():
    coords = np.asarray([[0, 0], [1, 0], [0, 1], [8, 8]], dtype=np.float32)
    edges = coordinate_knn_edges(coords, k=2, symmetrize=True)
    pairs = set(map(tuple, edges.T.tolist()))
    assert pairs
    assert all(source != target for source, target in pairs)
    assert all((target, source) in pairs for source, target in pairs)


def test_radius_edges_respect_radius_and_symmetry():
    coords = np.asarray([[0, 0], [1, 0], [0, 2], [8, 8]], dtype=np.float32)
    edges = coordinate_radius_edges(coords, radius=2.1, symmetrize=True)
    pairs = set(map(tuple, edges.T.tolist()))
    assert pairs == {(0, 1), (1, 0), (0, 2), (2, 0)}


def test_cindex_uses_larger_score_for_higher_risk():
    time = np.asarray([1.0, 2.0, 3.0])
    event = np.asarray([1, 1, 1])
    risk = np.asarray([3.0, 2.0, 1.0])
    assert concordance_index(time, risk, event) == 1.0


def test_patient_folds_do_not_split_patients():
    rows = []
    for index in range(20):
        rows.append(
            {
                "patient_id": f"p{index}",
                "slide_id": f"s{index}",
                "survival_time": float(index + 1),
                "event": index % 2,
                "feature_path": f"s{index}.h5",
            }
        )
    frame = pd.DataFrame(rows)
    folds = patient_folds(frame, n_splits=5, seed=3)
    for split in folds:
        assert not (set(split["train"]) & set(split["validation"]))
        assert not (set(split["train"]) & set(split["test"]))
        assert not (set(split["validation"]) & set(split["test"]))
