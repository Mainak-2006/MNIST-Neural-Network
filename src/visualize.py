"""Visualise the dataset, the training curves, the model's mistakes, and
what the first layer learned.

Produces artifacts/plots.png with four panels:
  1. a grid of sample training digits
  2. loss and accuracy curves over the training epochs
  3. test images the model got wrong, labelled true -> predicted
  4. the learned first-layer weights/filters (red = positive, blue = negative)

and artifacts/weights.png, a larger grid of the same first-layer filters.

Run:  python visualize.py                        (uses artifacts/ from train.py)
      python visualize.py --model artifacts/model.npz
      python visualize.py --weight-count 32 --weight-layer 0
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")                   # render to file, no display needed
import matplotlib.pyplot as plt

import data
from models import load_model, as_model_input

CMAP = plt.get_cmap("RdBu_r")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="artifacts/model.npz")
    p.add_argument("--history", default="artifacts/history.npz")
    p.add_argument("--out", default="artifacts/plots.png")
    p.add_argument("--weights-out", default="artifacts/weights.png")
    p.add_argument("--weight-layer", type=int, default=0,
                   help="index of the weight layer to visualise (0 = first)")
    p.add_argument("--weight-count", type=int, default=64,
                   help="how many units/filters to show in the plots.png panel")
    p.add_argument("--weight-all", action="store_true",
                   help="show every unit in weights.png (default: same as count)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def first_layer_filters(model, layer_idx=0):
    """Return a list of 2-D arrays: each is one first-layer unit/filter.

    A Linear layer has W of shape (fan_in, units): column j is unit j's
    incoming weights, reshaped back into the image it sees. A Conv2d layer
    has W of shape (out, in, k, k): filter o (on the single input channel)
    is already a small 2-D image.
    """
    layer = model.weight_layers[layer_idx]
    W = layer.W
    if W.ndim == 4:                                    # Conv2d
        return [W[o, 0] for o in range(W.shape[0])]
    side = int(round(np.sqrt(W.shape[0])))             # Linear
    return [W[:, j].reshape(side, side) for j in range(W.shape[1])]


def make_filter_grid(filters, gap=1):
    """Tile 2-D filters into one 2-D array (NaN padding shows as blank)."""
    n = len(filters)
    h, w = filters[0].shape
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    grid = np.full((rows * (h + gap) - gap, cols * (w + gap) - gap), np.nan)
    for k, f in enumerate(filters):
        r, c = divmod(k, cols)
        grid[r * (h + gap):r * (h + gap) + h,
             c * (w + gap):c * (w + gap) + w] = f
    return grid


def draw_filters(ax, filters, title, count=None):
    """Draw a filter grid on `ax` with a symmetric diverging colour scale."""
    shown = filters[:count] if count else filters
    vmax = max(float(np.abs(f).max()) for f in shown) or 1.0
    grid = make_filter_grid(shown)
    cmap = CMAP.copy()
    cmap.set_bad("white")
    ax.imshow(grid, cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    (x_train, y_train), (x_test, y_test) = data.load(one_hot_labels=False)
    model = load_model(args.model)
    history = np.load(args.history)

    fig = plt.figure(figsize=(11, 12))

    # 1. sample digits -----------------------------------------------------
    ax = fig.add_subplot(4, 1, 1)
    order = rng.permutation(len(x_train))[:20]
    grid = np.zeros((2 * 28, 10 * 28), dtype=np.float32)
    for k, i in enumerate(order):
        r, c = divmod(k, 10)
        grid[r * 28:(r + 1) * 28, c * 28:(c + 1) * 28] = x_train[i].reshape(28, 28)
    ax.imshow(grid, cmap="gray")
    ax.set_title("Sample training images")
    ax.axis("off")

    # 2. loss and accuracy curves ------------------------------------------
    ax = fig.add_subplot(4, 1, 2)
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

    # 3. misclassified test images -----------------------------------------
    preds = model.predict(as_model_input(model, x_test))
    wrong = np.where(preds != y_test)[0]
    chosen = rng.permutation(wrong)[:10] if len(wrong) else []

    ax = fig.add_subplot(4, 1, 3)
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

    # 4. learned first-layer features --------------------------------------
    filters = first_layer_filters(model, args.weight_layer)
    ax = fig.add_subplot(4, 1, 4)
    draw_filters(ax, filters,
                 f"First-layer features (layer {args.weight_layer}, "
                 f"showing {min(args.weight_count, len(filters))} of {len(filters)})",
                 count=args.weight_count)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"Saved plot to {args.out}")

    # A larger, standalone grid of every first-layer unit.
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    count = None if args.weight_all else args.weight_count
    draw_filters(ax2, filters,
                 f"First-layer features ({len(filters)} units, layer {args.weight_layer})",
                 count=count)
    fig2.tight_layout()
    fig2.savefig(args.weights_out, dpi=130)
    print(f"Saved weights to {args.weights_out}")


if __name__ == "__main__":
    main()
