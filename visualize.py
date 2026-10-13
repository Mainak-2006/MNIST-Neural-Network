"""Visualise the dataset, the training curves, and the model's mistakes.

Produces artifacts/plots.png with three panels:
  1. a grid of sample training digits
  2. loss and accuracy curves over the training epochs
  3. test images the model got wrong, labelled true -> predicted

Run:  python visualize.py             (uses artifacts/ from train.py)
      python visualize.py --model artifacts/model.npz
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")                   # render to file, no display needed
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import data
from network import MLP


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="artifacts/model.npz")
    p.add_argument("--history", default="artifacts/history.npz")
    p.add_argument("--out", default="artifacts/plots.png")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    (x_train, y_train), (x_test, y_test) = data.load(one_hot_labels=False)
    model = MLP.load(args.model)
    history = np.load(args.history)

    fig = plt.figure(figsize=(11, 9))

    # 1. sample digits -----------------------------------------------------
    ax = fig.add_subplot(3, 1, 1)
    order = rng.permutation(len(x_train))[:20]
    grid = np.zeros((2 * 28, 10 * 28), dtype=np.float32)
    for k, i in enumerate(order):
        r, c = divmod(k, 10)
        grid[r * 28:(r + 1) * 28, c * 28:(c + 1) * 28] = x_train[i].reshape(28, 28)
    ax.imshow(grid, cmap="gray")
    ax.set_title("Sample training images")
    ax.axis("off")

    # 2. loss and accuracy curves ------------------------------------------
    ax = fig.add_subplot(3, 1, 2)
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], "o-", color="tab:red", label="train loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss", color="tab:red")
    ax.tick_params(axis="y", labelcolor="tab:red")

    ax2 = ax.twinx()
    ax2.plot(epochs, history["train_acc"] * 100, "s-", color="tab:blue",
             label="train acc")
    ax2.plot(epochs, history["test_acc"] * 100, "d-", color="tab:green",
             label="test acc")
    ax2.set_ylabel("accuracy (%)")
    ax2.set_ylim(80, 100)
    ax.set_title("Training curves")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="center right")

    # Box out train loss, train acc, and test acc on the graph
    loss_vals = history["train_loss"]
    loss_min, loss_max = float(np.min(loss_vals)), float(np.max(loss_vals))
    pad_loss = 0.08 * max(1e-6, loss_max - loss_min)
    ax.add_patch(Rectangle((epochs[0] - 0.3, loss_min - pad_loss),
                           epochs[-1] - epochs[0] + 0.6, loss_max - loss_min + 2 * pad_loss,
                           edgecolor="tab:red", linewidth=2, facecolor="none", zorder=10))

    train_acc_vals = history["train_acc"] * 100
    test_acc_vals = history["test_acc"] * 100
    pad_acc = 0.5
    tmin, tmax = float(np.min(train_acc_vals)), float(np.max(train_acc_vals))
    ax2.add_patch(Rectangle((epochs[0] - 0.3, tmin - pad_acc),
                            epochs[-1] - epochs[0] + 0.6, tmax - tmin + 2 * pad_acc,
                            edgecolor="tab:blue", linewidth=2, facecolor="none", zorder=10))

    smin, smax = float(np.min(test_acc_vals)), float(np.max(test_acc_vals))
    ax2.add_patch(Rectangle((epochs[0] - 0.3, smin - pad_acc),
                            epochs[-1] - epochs[0] + 0.6, smax - smin + 2 * pad_acc,
                            edgecolor="tab:green", linewidth=2, facecolor="none", zorder=10))

    # 3. misclassified test images -----------------------------------------
    preds = model.predict(x_test)
    wrong = np.where(preds != y_test)[0]
    chosen = rng.permutation(wrong)[:10] if len(wrong) else []

    ax = fig.add_subplot(3, 1, 3)
    if len(chosen):
        grid = np.zeros((28, 10 * 28), dtype=np.float32)
        labels = []
        for k, i in enumerate(chosen):
            grid[:, k * 28:(k + 1) * 28] = x_test[i].reshape(28, 28)
            labels.append(f"{y_test[i]}\u2192{preds[i]}")
        ax.imshow(grid, cmap="gray")
        ax.set_xticks(np.arange(len(chosen)) * 28 + 14)
        ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks([])
    ax.set_title(f"Misclassified test digits (true\u2192pred) "
                 f"\u2014 {len(wrong)}/{len(x_test)} wrong, "
                 f"{(1 - len(wrong) / len(x_test)) * 100:.2f}% correct")

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()
