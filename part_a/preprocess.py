"""Part A, data preparation: turn the ratings into inputs and targets for the neural network.

The steps follow the assignment:

1. **Center** each user's ratings on that user's mean, so generous and strict raters become comparable.
2. **Fill** each missing rating with a random value in ``[-σ, σ]``, where σ is the standard deviation
   of that user's centered ratings.
3. **Normalize** each user's row to ``[0, 1]`` (min-max), the range of the network's targets.
4. **Encode** each user as a one-hot vector, the network's input.

Run ``python -m part_a.preprocess`` to print the statistics the assignment asks for.
"""
from dataclasses import dataclass

import numpy as np

from movielens import rating_matrix


@dataclass
class Dataset:
    inputs: np.ndarray    # users × users, one-hot user ids
    targets: np.ndarray   # users × movies, normalized ratings in [0, 1] (missing ones filled with noise)
    observed: np.ndarray  # users × movies, True where the user really rated the movie
    user_mean: np.ndarray  # per-user mean rating, to undo the centering
    row_min: np.ndarray    # per-user min and max of the centered, filled row, to undo the normalization
    row_max: np.ndarray

    def to_stars(self, normalized: np.ndarray, users: np.ndarray) -> np.ndarray:
        """Map normalized network outputs for the given users back to the 1–5 rating scale."""
        lo, hi = self.row_min[users, None], self.row_max[users, None]
        return normalized * (hi - lo) + lo + self.user_mean[users, None]


def center(ratings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Subtract each user's mean rating. Returns the centered matrix (NaNs kept) and the means."""
    means = np.nanmean(ratings, axis=1)
    return ratings - means[:, None], means


def fill_missing(centered: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Replace each NaN with a random value in [-σ, σ], σ being that user's sample standard deviation."""
    sigma = np.nan_to_num(np.nanstd(centered, axis=1, ddof=1))[:, None]
    noise = rng.uniform(-sigma, sigma, size=centered.shape)
    return np.where(np.isnan(centered), noise, centered)


def normalize_rows(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Min-max normalize each row to [0, 1]. Returns the result and each row's min and max."""
    lo, hi = matrix.min(axis=1), matrix.max(axis=1)
    span = np.where(hi > lo, hi - lo, 1.0)  # a constant row would otherwise divide by zero
    return (matrix - lo[:, None]) / span[:, None], lo, hi


def build_dataset(seed: int = 0, ratings: np.ndarray | None = None) -> Dataset:
    ratings = rating_matrix() if ratings is None else ratings
    rng = np.random.default_rng(seed)
    centered, means = center(ratings)
    filled = fill_missing(centered, rng)
    targets, lo, hi = normalize_rows(filled)
    return Dataset(inputs=np.eye(len(ratings)), targets=targets, observed=~np.isnan(ratings),
                   user_mean=means, row_min=lo, row_max=hi)


def main() -> None:
    ratings = rating_matrix()
    centered, means = center(ratings)
    data = build_dataset()
    print(f"{ratings.shape[0]} users x {ratings.shape[1]} movies, "
          f"{int((~np.isnan(ratings)).sum()):,} ratings ({(~np.isnan(ratings)).mean():.1%} of the matrix)")
    print(f"User mean rating:       {means.min():.2f} to {means.max():.2f}")
    print(f"Std before centering:   {np.nanstd(ratings, ddof=1):.3f}")
    print(f"Std after centering:    {np.nanstd(centered, ddof=1):.3f}")
    print(f"Centered rating range:  {np.nanmin(centered):.2f} to {np.nanmax(centered):.2f}")
    print(f"Largest |user mean| after centering: {np.abs(np.nanmean(centered, axis=1)).max():.1e}")
    print(f"Network input:  {data.inputs.shape} one-hot users")
    print(f"Network target: {data.targets.shape} normalized ratings in [0, 1]")


if __name__ == "__main__":
    main()
