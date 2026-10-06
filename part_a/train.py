"""Part A, training: a neural network that predicts a user's ratings for every movie.

**Reconstruction (2026).** The original ``training.py`` was never committed. This rebuilds it from the
assignment and the surviving loss plots: the input is a user's one-hot id, the output is that user's
normalized rating for each of the 1,682 movies, trained with SGD and evaluated with 5-fold
cross-validation over users.

Besides the loss on the (partly noise-filled) targets, it reports the error in stars on the ratings
users actually gave, next to a simple baseline: each movie's average among the training users.

    python -m part_a.train                  # 5 folds, 400 epochs, 20 hidden units
    python -m part_a.train --hidden 50 --epochs 200
"""
import argparse
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor

from part_a.preprocess import Dataset, build_dataset

PLOTS = Path(__file__).resolve().parent / "plots"


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def rmse_stars(data: Dataset, predicted: np.ndarray, users: np.ndarray) -> float:
    """RMSE in stars (1–5) over the ratings these users actually gave."""
    stars = np.clip(data.to_stars(predicted, users), 1, 5)
    truth = data.to_stars(data.targets[users], users)
    mask = data.observed[users]
    return float(np.sqrt(np.mean((stars[mask] - truth[mask]) ** 2)))


def train_fold(data: Dataset, train: np.ndarray, test: np.ndarray, args) -> dict:
    net = MLPRegressor(hidden_layer_sizes=(args.hidden,), activation="logistic", solver="sgd",
                       learning_rate_init=args.lr, momentum=args.momentum, batch_size=args.batch,
                       random_state=args.seed)
    X, Y = data.inputs, data.targets
    train_loss, test_loss = [], []
    for _ in range(args.epochs):
        net.partial_fit(X[train], Y[train])
        train_loss.append(mse(net.predict(X[train]), Y[train]))
        test_loss.append(mse(net.predict(X[test]), Y[test]))
    baseline = np.repeat(Y[train].mean(axis=0, keepdims=True), len(test), axis=0)
    return {"train_loss": train_loss, "test_loss": test_loss,
            "rmse": rmse_stars(data, net.predict(X[test]), test),
            "baseline_rmse": rmse_stars(data, baseline, test)}


def plot(results: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=110)
    for i, r in enumerate(results):
        ax.plot(r["train_loss"], color="tab:blue", alpha=0.25, lw=1)
        ax.plot(r["test_loss"], color="tab:orange", alpha=0.25, lw=1)
    ax.plot(np.mean([r["train_loss"] for r in results], axis=0), color="tab:blue", lw=2, label="Train (mean of folds)")
    ax.plot(np.mean([r["test_loss"] for r in results], axis=0), color="tab:orange", lw=2, label="Test (mean of folds)")
    ax.set(title="Part A: loss per epoch, 5-fold cross-validation", xlabel="Epoch", ylabel="MSE (normalized ratings)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hidden", type=int, default=20, help="hidden units (default 20)")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.01, help="SGD learning rate")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", type=Path, default=PLOTS / "loss.png")
    args = parser.parse_args()

    import warnings
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    data = build_dataset(seed=args.seed)
    folds = KFold(args.folds, shuffle=True, random_state=args.seed).split(data.inputs)
    results = []
    for k, (train, test) in enumerate(folds, 1):
        r = train_fold(data, train, test, args)
        results.append(r)
        print(f"fold {k}: train MSE {r['train_loss'][-1]:.4f}  test MSE {r['test_loss'][-1]:.4f}  "
              f"test RMSE {r['rmse']:.3f} stars  (movie-average baseline {r['baseline_rmse']:.3f})")
    print(f"mean:   test RMSE {np.mean([r['rmse'] for r in results]):.3f} stars, "
          f"baseline {np.mean([r['baseline_rmse'] for r in results]):.3f}")
    plot(results, args.plot)
    print(f"loss curves: {args.plot}")


if __name__ == "__main__":
    main()
