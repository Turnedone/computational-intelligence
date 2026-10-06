"""Part B: fill in a user's missing ratings with a genetic algorithm.

Each individual is a full row of ratings for the chosen user: the movies the user really rated are
fixed, every other movie gets a rating 1–5 that evolves. Fitness rewards rows that agree with the
user's most similar users (Pearson correlation), on the idea that you rate like people who rate like you.

1. **Neighbours:** the K users whose ratings correlate best with the chosen user's.
2. **Population:** N rows, missing ratings drawn uniformly from 1–5.
3. **Fitness:** sum over the neighbours of (Pearson(row, neighbour) + 1) / 2, so each neighbour adds 0–1.
4. **Selection:** rank-based roulette wheel, plus elitism (the best rows survive unchanged).
5. **Crossover:** uniform crossover of the free genes between pairs, with probability ``crossover_rate``.
6. **Mutation:** each free gene is redrawn with probability ``mutation_rate``.
7. **Stopping:** after ``max_generations``, or once the best fitness has improved by less than 1% over
   the last ``window`` generations (checked after ``min_generations``).

    python -m part_b.genetic --user 2
    python -m part_b.genetic --user 2 --population 50 --mutation-rate 0.02 --seed 7

Cleaned up in 2026 from the 2020 coursework (git tag ``coursework-2020``), fixing bugs where mutation and
crossover changed the user's real ratings instead of the missing ones, crossover didn't really swap genes,
and selection shared individuals so that mutating one silently changed its copies.
"""
import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from movielens import rating_matrix

PLOTS = Path(__file__).resolve().parent / "plots"


@dataclass
class Config:
    neighbours: int = 10
    population: int = 20
    crossover_rate: float = 0.6
    mutation_rate: float = 0.05
    elite: int = 1
    max_generations: int = 100
    min_generations: int = 50
    window: int = 10
    min_improvement: float = 0.01


@dataclass
class Result:
    best: np.ndarray                    # the best row of ratings found
    best_fitness: float
    history: list[float] = field(default_factory=list)  # best fitness per generation
    neighbours: np.ndarray | None = None


def correlate(rows: np.ndarray, others: np.ndarray) -> np.ndarray:
    """Pearson correlation of every row in ``rows`` with every row in ``others`` (shape rows × others).

    Rows with no variance get correlation 0 instead of NaN.
    """
    a = rows - rows.mean(axis=1, keepdims=True)
    b = others - others.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(a, axis=1)[:, None] * np.linalg.norm(b, axis=1)[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = (a @ b.T) / norms
    return np.nan_to_num(corr)


def nearest_neighbours(ratings: np.ndarray, user: int, k: int) -> np.ndarray:
    """Indices of the k users most correlated with ``user`` (missing ratings count as 0, as in the original)."""
    filled = np.nan_to_num(ratings)
    corr = correlate(filled[user:user + 1], filled)[0]
    corr[user] = -np.inf  # never your own neighbour
    return np.argsort(corr)[::-1][:k]


def fitness(population: np.ndarray, neighbour_rows: np.ndarray) -> np.ndarray:
    """Each individual's summed, 0–1 scaled Pearson correlation with the neighbours."""
    return ((correlate(population, neighbour_rows) + 1) / 2).sum(axis=1)


def select(population: np.ndarray, scores: np.ndarray, elite: int, rng: np.random.Generator) -> np.ndarray:
    """Rank-based roulette-wheel selection. Returns a new array, so no two individuals share memory."""
    order = np.argsort(scores)[::-1]
    n = len(population)
    weights = np.arange(n, 0, -1, dtype=float)  # best rank gets n, worst gets 1
    picked = order[rng.choice(n, size=n - elite, p=weights / weights.sum())]
    return np.concatenate([population[order[:elite]], population[picked]]).copy()


def crossover(population: np.ndarray, free: np.ndarray, rate: float, elite: int,
              rng: np.random.Generator) -> None:
    """Uniform crossover, in place, between consecutive pairs after the elite. Fixed genes are never touched."""
    for i in range(elite, len(population) - 1, 2):
        if rng.random() < rate:
            swap = free & (rng.random(population.shape[1]) < 0.5)
            population[i, swap], population[i + 1, swap] = population[i + 1, swap], population[i, swap].copy()


def mutate(population: np.ndarray, free: np.ndarray, rate: float, elite: int,
           rng: np.random.Generator) -> None:
    """Redraw each free gene (outside the elite) with probability ``rate``, in place."""
    hit = (rng.random(population.shape) < rate) & free
    hit[:elite] = False
    population[hit] = rng.integers(1, 6, size=hit.sum())


def converged(history: list[float], cfg: Config) -> bool:
    if len(history) <= max(cfg.min_generations, cfg.window):
        return False
    before = history[-1 - cfg.window]
    return (history[-1] - before) / before < cfg.min_improvement


def run(ratings: np.ndarray, user: int, cfg: Config = Config(), seed: int = 0) -> Result:
    """Evolve the missing ratings of ``user`` (a 0-based row index of ``ratings``)."""
    rng = np.random.default_rng(seed)
    known = ~np.isnan(ratings[user])
    free = ~known
    neighbours = nearest_neighbours(ratings, user, cfg.neighbours)
    neighbour_rows = np.nan_to_num(ratings[neighbours])

    population = rng.integers(1, 6, size=(cfg.population, ratings.shape[1])).astype(float)
    population[:, known] = ratings[user, known]
    scores = fitness(population, neighbour_rows)
    history = [float(scores.max())]

    for _ in range(cfg.max_generations):
        population = select(population, scores, cfg.elite, rng)
        crossover(population, free, cfg.crossover_rate, cfg.elite, rng)
        mutate(population, free, cfg.mutation_rate, cfg.elite, rng)
        scores = fitness(population, neighbour_rows)
        history.append(float(scores.max()))
        if converged(history, cfg):
            break

    best = int(np.argmax(scores))
    return Result(best=population[best].copy(), best_fitness=float(scores[best]), history=history,
                  neighbours=neighbours)


def plot(history: list[float], user: int, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=110)
    ax.plot(history, lw=2, color="tab:green")
    ax.set(title=f"Part B: best fitness per generation (user {user})", xlabel="Generation",
           ylabel="Fitness (sum over 10 neighbours, max 10)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--user", type=int, default=2, help="MovieLens user id, 1–943 (default 2)")
    parser.add_argument("--neighbours", type=int, default=Config.neighbours)
    parser.add_argument("--population", type=int, default=Config.population)
    parser.add_argument("--crossover-rate", type=float, default=Config.crossover_rate)
    parser.add_argument("--mutation-rate", type=float, default=Config.mutation_rate)
    parser.add_argument("--generations", type=int, default=Config.max_generations)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", type=Path, default=PLOTS / "fitness.png")
    args = parser.parse_args()

    ratings = rating_matrix()
    cfg = Config(neighbours=args.neighbours, population=args.population, crossover_rate=args.crossover_rate,
                 mutation_rate=args.mutation_rate, max_generations=args.generations)
    user = args.user - 1
    result = run(ratings, user, cfg, seed=args.seed)

    known = ~np.isnan(ratings[user])
    print(f"user {args.user}: {known.sum()} rated, {(~known).sum()} to fill in")
    print(f"neighbours: users {', '.join(str(n + 1) for n in result.neighbours)}")
    print(f"fitness: {result.history[0]:.3f} -> {result.best_fitness:.3f} "
          f"after {len(result.history) - 1} generations")
    assert np.array_equal(result.best[known], ratings[user, known]), "real ratings must never change"
    counts = np.bincount(result.best[~known].astype(int), minlength=6)[1:]
    print("filled-in ratings by stars: " + ", ".join(f"{s}: {c}" for s, c in enumerate(counts, 1)))
    plot(result.history, args.user, args.plot)
    print(f"fitness curve: {args.plot}")


if __name__ == "__main__":
    main()
