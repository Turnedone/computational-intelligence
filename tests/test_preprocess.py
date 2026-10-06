import numpy as np

from part_a.preprocess import build_dataset, center, fill_missing, normalize_rows

R = np.array([[5, np.nan, 3, 1],
              [4, 4, np.nan, 2],
              [np.nan, 1, 1, 5]], dtype=float)


def test_center_gives_each_user_a_zero_mean():
    centered, means = center(R)
    np.testing.assert_allclose(np.nanmean(centered, axis=1), 0, atol=1e-12)
    np.testing.assert_allclose(means, [3, 10 / 3, 7 / 3])


def test_fill_missing_only_fills_gaps_within_one_std():
    centered, _ = center(R)
    filled = fill_missing(centered, np.random.default_rng(0))
    known = ~np.isnan(R)
    np.testing.assert_array_equal(filled[known], centered[known])
    sigma = np.nanstd(centered, axis=1, ddof=1)
    rows, _ = np.where(~known)
    assert np.all(np.abs(filled[~known]) <= sigma[rows])


def test_normalize_rows_maps_each_row_to_0_1_and_survives_constant_rows():
    m = np.array([[1.0, 3.0, 2.0], [7.0, 7.0, 7.0]])
    norm, lo, hi = normalize_rows(m)
    np.testing.assert_allclose(norm[0], [0, 1, 0.5])
    assert np.all(np.isfinite(norm[1]))


def test_to_stars_undoes_the_preprocessing_on_real_ratings():
    data = build_dataset(seed=1, ratings=R)
    users = np.arange(len(R))
    stars = data.to_stars(data.targets, users)
    np.testing.assert_allclose(stars[data.observed], R[data.observed])
    np.testing.assert_array_equal(data.inputs, np.eye(3))
