# Part A: predicting ratings with a neural network

Predict every movie rating of a user from nothing but who the user is.
Part B tackles the same problem with a genetic algorithm: see [`part_b/`](../part_b).

## Method

**Preparation** ([`preprocess.py`](preprocess.py)), as the assignment specifies:

1. **Center** each user's ratings on their own mean, so strict and generous raters become comparable.
2. **Fill** each missing rating with a random value in [−σ, σ] of that user's centered ratings.
3. **Normalize** each user's row to [0, 1].
4. **Encode** each user as a one-hot vector.

**Network** ([`train.py`](train.py)): one-hot user (943 inputs) → 20 sigmoid hidden units → 1,682 outputs,
one normalized rating per movie. Trained with SGD + momentum for 400 epochs, evaluated with 5-fold
cross-validation over users.

```bash
python -m part_a.preprocess   # statistics of each preparation step
python -m part_a.train        # train, cross-validate and plot (about a minute)
python -m part_a.train --hidden 50 --epochs 200 --lr 0.01
```

## Results

![Train and test loss per epoch](plots/loss.png)

| | Test RMSE (stars, real ratings only) |
|---|---|
| Neural network | 1.080 |
| Baseline: each movie's average rating among the training users | 1.080 |

**The network learns nothing a movie average doesn't already know, and that's the interesting part.**
Users are split between training and testing, so a test user's one-hot input is a position the network has
never seen. All it can output for them is what it learned from the bias terms: roughly each movie's average.
The plot shows it: training loss falls as the network memorizes its training users, while test loss is
flat from the first epoch.

To predict ratings for a *known* user, the split has to be over ratings instead of users (hold out some of
each user's ratings), or the input has to describe the user (for example their known ratings, as in an
autoencoder) rather than just identify them.

## History

This is a 2026 cleanup of the 2020 coursework. The original scripts (`dataProcess.py`, `crossvalidation.py`,
`maketest.py`) and plots are at the [`coursework-2020`](https://github.com/Turnedone/computational-intelligence/tree/coursework-2020/part-a) tag.
The original `training.py` was never committed, so [`train.py`](train.py) is a reconstruction from the assignment
and the surviving plots. Its final test loss (≈0.04) matches the original plots.

The rewrite does the same preparation with vectorized NumPy instead of cell-by-cell loops over the
1.6-million-cell matrix, keeps everything in memory instead of passing data between scripts through CSV files,
and adds tests ([`tests/test_preprocess.py`](../tests/test_preprocess.py)).

Original notes (Greek):

> Για το project δημιούργησα 3 συνολικά python αρχεία.
> 1) Το dataProcess.py που αφορά τα πρώτα ερωτήματα για τη προετοιμασία των δεδομένων
> 2) Το crossvalidation.py όπου υλοποιείται το 5 cross validation
> 3) To training.py όπου γίνεται η υλοποίηση και η μάθηση του νευρωνικού δικτύου.
