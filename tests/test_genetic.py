import numpy as np
import pytest

from part_b.genetic import Config, correlate, crossover, fitness, mutate, nearest_neighbours, run, select

rng = np.random.default_rng(0)
# 30 users × 40 movies, about half rated
RATINGS = np.where(rng.random((30, 40)) < 0.5, rng.integers(1, 6, (30, 40)).astype(float), np.nan)


def test_correlate_matches_numpy():
    a, b = rng.random((3, 10)), rng.random((4, 10))
    expected = np.corrcoef(np.vstack([a, b]))[:3, 3:]
    np.testing.assert_allclose(correlate(a, b), expected)


def test_a_user_is_never_their_own_neighbour():
    for user in range(5):
        assert user not in nearest_neighbours(RATINGS, user, 10)


def test_fitness_is_highest_for_a_row_identical_to_the_neighbours():
    neighbours = np.tile(np.arange(1, 6, dtype=float), (3, 2))
    population = np.vstack([neighbours[0], neighbours[0][::-1]])
    scores = fitness(population, neighbours)
    assert scores[0] == pytest.approx(3.0) and scores[1] < scores[0]


def test_selection_returns_independent_copies():
    population = rng.integers(1, 6, (6, 8)).astype(float)
    picked = select(population, np.arange(6.0), elite=1, rng=np.random.default_rng(1))
    picked[0, 0] = 99
    assert 99 not in population
    assert len({id(row.base) for row in picked}) == 1  # one fresh array, not views into the old population


def test_crossover_and_mutation_never_touch_fixed_genes():
    population = rng.integers(1, 6, (10, 20)).astype(float)
    free = np.zeros(20, dtype=bool)
    free[::2] = True
    before = population.copy()
    crossover(population, free, rate=1.0, elite=0, rng=np.random.default_rng(2))
    mutate(population, free, rate=1.0, elite=0, rng=np.random.default_rng(3))
    np.testing.assert_array_equal(population[:, ~free], before[:, ~free])
    assert not np.array_equal(population[:, free], before[:, free])


def test_crossover_swaps_genes_between_partners():
    population = np.array([[1.0] * 6, [5.0] * 6])
    crossover(population, np.ones(6, dtype=bool), rate=1.0, elite=0, rng=np.random.default_rng(4))
    assert sorted(population[:, 0]) == [1.0, 5.0]  # each gene moved, none duplicated
    np.testing.assert_array_equal(population.sum(axis=0), [6.0] * 6)


def test_run_keeps_real_ratings_and_never_gets_worse_with_elitism():
    user = 3
    result = run(RATINGS, user, Config(neighbours=5, max_generations=30), seed=0)
    known = ~np.isnan(RATINGS[user])
    np.testing.assert_array_equal(result.best[known], RATINGS[user, known])
    assert set(np.unique(result.best[~known])) <= {1, 2, 3, 4, 5}
    assert all(b >= a - 1e-12 for a, b in zip(result.history, result.history[1:]))
