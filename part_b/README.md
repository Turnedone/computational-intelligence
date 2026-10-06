# Part B: filling in ratings with a genetic algorithm

Complete a user's missing movie ratings so they agree with the users who rate most like them.
Part A tackles the same problem with a neural network: see [`part_a/`](../part_a).

## How it works

Everything is in [`genetic.py`](genetic.py). An individual is a full row of 1,682 ratings for the chosen user:
the movies the user really rated are **fixed**, every other movie is a **free gene** rated 1–5.

1. **Neighbours:** the 10 users whose ratings correlate best (Pearson) with the chosen user's.
2. **Population:** 20 rows, free genes drawn uniformly from 1–5.
3. **Fitness:** the sum over the neighbours of (Pearson(row, neighbour) + 1) / 2, so the maximum is 10.
4. **Selection:** rank-based roulette wheel, keeping the best row unchanged (elitism).
5. **Crossover:** uniform crossover of the free genes between pairs (probability 0.6).
6. **Mutation:** each free gene is redrawn with probability 0.05.
7. **Stopping:** after 100 generations, or earlier once the best fitness improves by less than 1% over 10 generations.

```bash
python -m part_b.genetic --user 2
python -m part_b.genetic --user 50 --population 50 --mutation-rate 0.02 --seed 7
python -m part_b.genetic --help
```

```
user 2: 62 rated, 1620 to fill in
neighbours: users 701, 931, 460, 131, 104, 735, 413, 15, 834, 768
fitness: 5.445 -> 5.695 after 50 generations
filled-in ratings by stars: 1: 357, 2: 318, 3: 332, 4: 308, 5: 305
```

It also saves the fitness curve to `plots/fitness.png`.

## History: bugs fixed in the 2026 cleanup

The 2020 version ([`coursework-2020` tag](https://github.com/Turnedone/computational-intelligence/tree/coursework-2020/part-b))
had the right idea but some bugs that stopped it from doing what it set out to do:

- **It evolved the wrong genes.** The mask that marks which ratings may change was inverted: mutation and
  crossover changed the user's *real* ratings and left the missing ones at their first random guess.
- **Crossover read the mask with the wrong index** (the individual's position instead of the gene's), and its
  "swap" wasn't one: the second child copied a gene the first child had already overwritten.
- **Selection shared individuals.** Picking the same individual twice stored the same object twice, so mutating
  one silently mutated its copies and changed the recorded "best of generation" scores after the fact.
- **The early-stopping check would have crashed** (it read `.score` from a list) and was never called.
- The mutation rate was 0.4 per gene; across ~1,600 free genes that's close to a random restart every
  generation. The default is now 0.05 (`--mutation-rate 0.4` reproduces the original); over the same
  50 generations it improves fitness about four times as much.

It's also vectorized (whole population at once instead of nested Python loops) and tested
([`tests/test_genetic.py`](../tests/test_genetic.py)), including checks that real ratings never change.
