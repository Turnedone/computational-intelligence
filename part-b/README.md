# Computational Intelligence, Part B: filling in ratings with a genetic algorithm

University coursework (2020) for the *Computational Intelligence* (Υπολογιστική Νοημοσύνη) course:
completing a user's missing movie ratings on the [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) dataset
with a **genetic algorithm** guided by the user's most similar neighbours.
Part A, which uses a neural network on the same problem, is in [ypologistiki-noimosini](https://github.com/Turnedone/ypologistiki-noimosini).

> **Archived coursework.** This repository is kept as a record and isn't maintained.

## How it works

Everything is in [`main.py`](main.py):

1. **Neighbours:** for the chosen user, find the 10 most similar users by Pearson correlation of their ratings.
2. **Population:** 20 candidate solutions, each the user's real ratings with every missing rating filled with a random 1–5.
3. **Fitness:** the sum of the candidate's Pearson correlations (scaled to 0–1) with each of the 10 neighbours.
4. **Selection:** rank-based roulette-wheel selection.
5. **Crossover** (probability 0.6) and **mutation** (probability 0.4 per gene; real ratings are never changed).
6. **Stopping:** after 100 generations. A "stop when fitness improves by less than 1%" check (`conditionImprove`)
   is written but not wired into the loop.

It prints the best fitness per generation.

## Running it

Put the MovieLens 100K ratings file `u.data` next to `main.py` (it's included in the
[Part A repository](https://github.com/Turnedone/ypologistiki-noimosini) or available from GroupLens), then:

```bash
pip install pandas numpy
python main.py
```

## Data

MovieLens 100K: F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.*
ACM Transactions on Interactive Intelligent Systems 5(4): 19. https://doi.org/10.1145/2827872.
