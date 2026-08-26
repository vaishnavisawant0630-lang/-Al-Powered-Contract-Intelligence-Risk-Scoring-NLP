import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")

from classification.calibrator import calibrate, fit_calibrators  # noqa: E402


def test_calibrator_output_shape_and_range():
    rng = np.random.default_rng(0)
    dev_probs = rng.random((100, 5))
    dev_labels = (dev_probs > 0.5).astype(int)  # correlated so isotonic has signal

    calibrators = fit_calibrators(dev_probs, dev_labels)
    calibrated = calibrate(dev_probs, calibrators)

    assert calibrated.shape == (100, 5)
    assert (calibrated >= 0).all() and (calibrated <= 1).all()


def test_calibrator_monotone_per_label():
    dev_probs = np.array([[0.1], [0.3], [0.5], [0.7], [0.9]])
    dev_labels = np.array([[0], [0], [1], [1], [1]])
    calibrators = fit_calibrators(dev_probs, dev_labels)

    a = calibrate(np.array([0.2]), calibrators)
    b = calibrate(np.array([0.8]), calibrators)
    assert a[0] <= b[0]


def test_degenerate_column_no_crash():
    # All-zero label column — must fall back to identity, not raise.
    dev_probs = np.array([[0.1, 0.2], [0.4, 0.6], [0.9, 0.1]])
    dev_labels = np.array([[0, 1], [0, 0], [0, 1]])
    calibrators = fit_calibrators(dev_probs, dev_labels)
    out = calibrate(dev_probs, calibrators)
    assert out.shape == dev_probs.shape
