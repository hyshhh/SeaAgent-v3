import numpy as np

from eta.metrics import binary_metrics, calibration_metrics, stop_metrics


def test_binary_metrics_confusion_matrix():
    result = binary_metrics(
        np.asarray([0.1, 0.9, 0.8, 0.2]),
        np.asarray([0, 0, 1, 1]),
    )
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["accuracy"] == 0.5


def test_calibration_is_zero_for_exact_probabilities():
    probabilities = np.asarray([0.0, 0.0, 1.0, 1.0])
    labels = np.asarray([0, 0, 1, 1])
    result = calibration_metrics(probabilities, labels, bins=5)
    assert result["expected_calibration_error"] == 0.0


def test_stop_requires_both_admission_and_completeness_threshold():
    result = stop_metrics(
        np.asarray([0.9, 0.4, 0.9]),
        np.asarray([0.9, 0.9, 0.6]),
        np.asarray([1, 1, 1]),
        np.asarray([1, 1, 1]),
        threshold=0.8,
    )
    assert result["confusion_matrix"] == [[0, 0], [2, 1]]
