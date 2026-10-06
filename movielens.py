"""Loading the MovieLens 100K ratings shared by both parts of the coursework."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_FILE = Path(__file__).resolve().parent / "data" / "u.data"


def load_ratings(path: Path = DATA_FILE) -> pd.DataFrame:
    """The raw ratings: one row per (user_id, item_id, rating, timestamp). Ids start at 1."""
    return pd.read_csv(path, sep="\t", names=["user_id", "item_id", "rating", "timestamp"])


def rating_matrix(ratings: pd.DataFrame | None = None) -> np.ndarray:
    """Users × movies matrix of ratings (1–5), with NaN where a user hasn't rated a movie.

    Row ``u`` is user ``u + 1`` and column ``i`` is movie ``i + 1``.
    """
    if ratings is None:
        ratings = load_ratings()
    matrix = np.full((ratings.user_id.max(), ratings.item_id.max()), np.nan)
    matrix[ratings.user_id - 1, ratings.item_id - 1] = ratings.rating
    return matrix
