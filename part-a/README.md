# Computational Intelligence, Part A: rating prediction with a neural network

University coursework (2020) for the *Computational Intelligence* (Υπολογιστική Νοημοσύνη) course:
predicting users' movie ratings on the [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) dataset with a neural network.
Part B, which uses a genetic algorithm on the same problem, is in [`part-b/`](../part-b).

## What's here

| File | What it does |
|---|---|
| `dataProcess.py` | Loads the ratings (`u.data`), centers each user's ratings on their mean, and prepares the data for the network |
| `crossvalidation.py` | Shuffles the prepared data and writes the train/test files for 5-fold cross-validation |
| `maketest.py` | Splits the encoded dataset into inputs and targets |
| `plots/` | Training and test loss curves from the network |
| `u.data` | MovieLens 100K ratings (user, movie, rating, timestamp) |

The training script (`training.py`) mentioned in the original notes below was never committed, so the network itself isn't in this repository; the loss curves in `plots/` come from it.

![Model loss over 400 epochs](plots/plot0.png)

## Original notes (Greek)

Για το project δημιούργησα 3 συνολικά python αρχεία.
1) Το dataProcess.py που αφορά τα πρώτα ερωτήματα για τη προετοιμασία των δεδομένων
2) Το crossvalidation.py όπου υλοποιείται το 5 cross validation
3) To training.py όπου γίνεται η υλοποίηση και η μάθηση του νευρωνικού δικτύου.

## Data

MovieLens 100K: F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.*
ACM Transactions on Interactive Intelligent Systems 5(4): 19. https://doi.org/10.1145/2827872.
Provided by [GroupLens Research](https://grouplens.org) under its own terms.
