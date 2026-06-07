"""Seasonal-block (pseudo-season-year) bootstrap primitives."""

import numpy as np
import pytest

from bootstrap import make_pseudo_year_blocks, percentile_band, resample_block_indices


def test_blocks_partition_all_indices():
    blocks = make_pseudo_year_blocks(3150, 35)
    assert len(blocks) == 35
    concat = np.concatenate(blocks)
    assert np.array_equal(np.sort(concat), np.arange(3150))
    # equal 90-day blocks when n = 35 * 90
    assert all(len(b) == 90 for b in blocks)


def test_blocks_uneven_sizes_differ_by_at_most_one():
    blocks = make_pseudo_year_blocks(500, 35)
    sizes = np.array([len(b) for b in blocks])
    assert sizes.sum() == 500
    assert sizes.max() - sizes.min() <= 1


def test_blocks_reject_bad_args():
    with pytest.raises(ValueError):
        make_pseudo_year_blocks(10, 0)
    with pytest.raises(ValueError):
        make_pseudo_year_blocks(10, 20)


def test_resample_preserves_length_and_uses_whole_blocks():
    blocks = make_pseudo_year_blocks(3150, 35)
    rng = np.random.default_rng(0)
    idx = resample_block_indices(blocks, rng)
    assert idx.size == 3150
    assert idx.min() >= 0 and idx.max() < 3150


def test_resample_is_seed_reproducible():
    blocks = make_pseudo_year_blocks(900, 30)
    a = resample_block_indices(blocks, np.random.default_rng(42))
    b = resample_block_indices(blocks, np.random.default_rng(42))
    assert np.array_equal(a, b)


def test_percentile_band_orders_and_brackets():
    rng = np.random.default_rng(1)
    samples = rng.normal(size=(1000, 4))
    lo, hi = percentile_band(samples, alpha=0.10)
    assert np.all(lo < hi)
    # ~90% of a column's values fall inside its band
    inside = ((samples >= lo) & (samples <= hi)).mean(axis=0)
    assert np.all(np.abs(inside - 0.90) < 0.05)
