"""Local-linear dependence-distance smoother (methodology sec:meth-distance)."""

import numpy as np
import pytest

from distance_smoothing import (
    kernel_weights,
    local_linear_predict,
    select_bandwidth,
    select_common_bandwidth,
    smooth_curve,
)


@pytest.mark.parametrize("kernel", ["epanechnikov", "gaussian"])
def test_local_linear_reproduces_a_line(kernel):
    # Local-linear smoothing is exact for an affine signal, any bandwidth.
    x = np.linspace(0.0, 300.0, 50)
    y = 1.2 + 0.0015 * x
    pred = local_linear_predict(x, y, np.array([40.0, 130.0, 260.0]), h=35.0, kernel=kernel)
    assert np.allclose(pred, 1.2 + 0.0015 * np.array([40.0, 130.0, 260.0]), atol=1e-9)


def test_epanechnikov_compact_support():
    t = np.array([-1.5, -0.5, 0.0, 0.5, 1.5])
    w = kernel_weights(t, "epanechnikov")
    assert w[0] == 0.0 and w[-1] == 0.0
    assert w[2] == 1.0 and np.all(w[1:4] > 0.0)


def test_gaussian_weights_positive():
    assert np.all(kernel_weights(np.linspace(-3, 3, 7), "gaussian") > 0.0)


def test_smoother_recovers_smooth_curve():
    rng = np.random.default_rng(0)
    f = lambda d: 1.0 + 0.7 * (1.0 - np.exp(-d / 100.0))
    x = np.sort(rng.uniform(10.0, 320.0, 528))
    y = f(x) + rng.normal(0.0, 0.03, x.size)
    grid = np.array([50.0, 100.0, 200.0])
    curve, h = smooth_curve(x, y, grid)
    assert np.all(np.abs(curve - f(grid)) < 0.02)
    assert h > 0.0


def test_select_bandwidth_returns_candidate():
    rng = np.random.default_rng(1)
    x = np.sort(rng.uniform(0.0, 300.0, 200))
    y = np.sin(x / 50.0) + rng.normal(0, 0.05, x.size)
    cands = np.array([5.0, 20.0, 60.0, 120.0])
    h = select_bandwidth(x, y, candidates=cands)
    assert h in cands


def test_common_bandwidth_single_value_for_two_series():
    rng = np.random.default_rng(2)
    x = np.sort(rng.uniform(0.0, 300.0, 200))
    y1 = 1.3 + 0.001 * x + rng.normal(0, 0.02, x.size)
    y2 = 1.5 + 0.001 * x + rng.normal(0, 0.02, x.size)
    cands = np.array([10.0, 40.0, 100.0])
    h = select_common_bandwidth(x, [y1, y2], candidates=cands)
    assert h in cands
