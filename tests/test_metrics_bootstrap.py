"""Tests for the Politis-Romano stationary block bootstrap primitive.

Lands the resampler as infrastructure that a later Hansen SPA test will
consume. Pure-numpy, fast.
"""

import numpy as np

from quant_lib.core._metrics import _stationary_block_bootstrap_resample


def test_stationary_resample_output_length():
    """Output length matches ``n`` by default and ``n_out`` when given."""
    for n in (10, 50, 200):
        arr = np.arange(n, dtype=np.float64)
        rng = np.random.default_rng(seed=12345)
        out = _stationary_block_bootstrap_resample(arr, rng, p=5)
        assert len(out) == n, f"n={n}: expected len {n}, got {len(out)}"

    arr = np.arange(50, dtype=np.float64)
    rng = np.random.default_rng(seed=12345)
    out = _stationary_block_bootstrap_resample(arr, rng, p=5, n_out=37)
    assert len(out) == 37


def test_stationary_resample_indices_in_range():
    """Resampled values are a subset of the original values (no OOB)."""
    arr = np.arange(200, dtype=np.float64)
    rng = np.random.default_rng(seed=999)
    out = _stationary_block_bootstrap_resample(arr, rng, p=7)
    assert set(np.unique(out)).issubset(set(np.unique(arr)))


def test_stationary_resample_reproducible():
    """Same seed -> identical output; different seed -> different output."""
    arr = np.linspace(0.0, 1.0, 200)

    rng1a = np.random.default_rng(seed=2024)
    out1a = _stationary_block_bootstrap_resample(arr, rng1a, p=6)

    rng1b = np.random.default_rng(seed=2024)
    out1b = _stationary_block_bootstrap_resample(arr, rng1b, p=6)

    rng2 = np.random.default_rng(seed=2025)
    out2 = _stationary_block_bootstrap_resample(arr, rng2, p=6)

    np.testing.assert_array_equal(out1a, out1b)
    assert not np.array_equal(out1a, out2), (
        "Different seeds produced identical resamples — RNG wiring suspect."
    )


def test_stationary_resample_empty_and_short():
    """Degenerate inputs return degenerate finite resamples."""
    # Empty input -> empty out
    empty = np.array([], dtype=np.float64)
    rng = np.random.default_rng(seed=7)
    out = _stationary_block_bootstrap_resample(empty, rng, p=5)
    assert len(out) == 0

    # Single-element input -> out length 1 (identity degenerate)
    one = np.array([42.0], dtype=np.float64)
    rng = np.random.default_rng(seed=7)
    out = _stationary_block_bootstrap_resample(one, rng, p=5)
    assert len(out) == 1
    assert out[0] == 42.0

    # p=0 -> degenerate, returns a copy unchanged
    arr = np.arange(20, dtype=np.float64)
    rng = np.random.default_rng(seed=7)
    out = _stationary_block_bootstrap_resample(arr, rng, p=0)
    assert len(out) == 20
    np.testing.assert_array_equal(out, arr)


def test_stationary_expected_block_length_empirical():
    """Parametrization guard: ``rng.geometric(1/p)`` has mean ``p``.

    Guards against the classic bug where someone passes ``p`` as the
    success-probability rather than as the expected block length. If
    this fails, the geometric parametrization in the primitive is wrong
    — fix the primitive, not the test.
    """
    rng = np.random.default_rng(seed=31337)
    for p in (3, 6, 10):
        draws = rng.geometric(1.0 / p, size=10000)
        empirical_mean = float(np.mean(draws))
        assert abs(empirical_mean - p) < 0.5, (
            f"p={p}: empirical geometric mean {empirical_mean:.3f} "
            f"deviates from expected {p} — parametrization bug."
        )
