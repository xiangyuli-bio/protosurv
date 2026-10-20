import numpy as np

from scripts.evaluate_localization import _evaluate_directory, _select_threshold

def test_localization_metrics_and_calibration(tmp_path):
    predictions = tmp_path / "predictions"
    annotations = tmp_path / "annotations"
    predictions.mkdir()
    annotations.mkdir()
    scores = np.asarray(
        [
            [0.9, 0.1, 0.8, 0.2],
            [0.8, 0.2, 0.7, 0.3],
            [0.1, 0.9, 0.2, 0.8],
            [0.2, 0.8, 0.3, 0.7],
        ]
    )
    np.savez_compressed(
        predictions / "slide.npz",
        attention=scores,
        coords=np.arange(8, dtype=np.float32).reshape(4, 2),
    )
    np.savez_compressed(
        annotations / "slide.npz",
        tumor=np.asarray([1, 0, 1, 0], dtype=bool),
        tissue=np.ones(4, dtype=bool),
    )
    rows = _evaluate_directory(
        predictions,
        annotations,
        class_index=0,
        num_prototypes=2,
        score_key="attention",
        threshold=0.5,
    )
    assert rows[0]["positive"] is True
    assert rows[0]["iou"] == 1.0
    assert 0.0 <= rows[0]["precision_at_20"] <= 1.0
    assert _select_threshold(
        predictions,
        annotations,
        class_index=0,
        num_prototypes=2,
        score_key="attention",
    ) in np.arange(0.05, 0.951, 0.05)
