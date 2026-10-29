"""Study the bias/variance trade-off by varying network depth.

We train MLPs with the same width but a different number of hidden layers
(0, 1, 2, 3, ...) on the same data and seed, then compare their train and
test accuracy:

  * too few layers  -> high bias: the model underfits (train acc low),
  * too many layers -> high variance: train acc keeps rising while the gap
                       to test acc widens (overfitting).

Run:  python experiments.py
      python experiments.py --depths 0 1 2 3 4 --width 256 --epochs 15
      python experiments.py --depths 1 2 3 --epochs 3 --subset 6000
"""

import argparse
import os
from argparse import Namespace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import data
from network import MLP
from train import train_model, make_optimizer


def parse_args():
    p = argparse.ArgumentParser(description="Depth / capacity study on MNIST.")
    p.add_argument("--depths", type=int, nargs="+", default=[0, 1, 2, 3],
                   help="numbers of hidden layers to compare")
    p.add_argument("--width", type=int, default=256,
                   help="width of each hidden layer")
    p.add_argument("--optimizer", choices=["sgd", "adam"], default="adam")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr-schedule", default="none",
                   choices=["exp", "step", "cosine", "none"])
    p.add_argument("--subset", type=int, default=0,
                   help="if >0, train on only this many examples (for fast runs)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="artifacts/depth_study.png")
    return p.parse_args()


def main():
    args = parse_args()
    (x_train, y_train), (x_test, y_test) = data.load()
    if args.subset > 0:
        x_train, y_train = x_train[:args.subset], y_train[:args.subset]

    results = []
    for depth in args.depths:
        hidden = [args.width] * depth
        layer_sizes = [x_train.shape[1], *hidden, 10]
        model = MLP(layer_sizes, seed=args.seed, dropout=args.dropout)
        optimizer = make_optimizer(args.optimizer, model.weight_layers,
                                   lr=args.lr)

        run = Namespace(
            lr=args.lr,
            lr_schedule=args.lr_schedule,
            lr_decay=1.0,
            lr_step_size=5,
            lr_final_frac=0.0,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
        )

        n_params = sum(l.W.size + l.b.size for l in model.weight_layers)
        print(f"\n=== depth {depth} ({' -> '.join(map(str, layer_sizes))}), "
              f"{n_params:,} params ===")
        model, history = train_model(model, x_train, y_train, x_test, y_test,
                                     optimizer=optimizer, args=run)
        results.append({
            "depth": depth,
            "params": n_params,
            "history": history,
            "train_acc": history["train_acc"][-1],
            "test_acc": history["test_acc"][-1],
        })

    print("\nSummary")
    print(f"{'depth':>6} {'params':>10} {'train_acc':>10} {'test_acc':>10} {'gap':>8}")
    for r in results:
        gap = r["train_acc"] - r["test_acc"]
        print(f"{r['depth']:>6} {r['params']:>10,} "
              f"{r['train_acc'] * 100:>9.2f}% {r['test_acc'] * 100:>9.2f}% "
              f"{gap * 100:>7.2f}%")

    # Plot: accuracy vs depth, plus per-depth learning curves.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    depths = [r["depth"] for r in results]
    ax1.plot(depths, [r["train_acc"] * 100 for r in results], "s-",
             color="tab:blue", label="train acc")
    ax1.plot(depths, [r["test_acc"] * 100 for r in results], "d-",
             color="tab:green", label="test acc")
    ax1.set_xlabel("number of hidden layers")
    ax1.set_ylabel("accuracy (%)")
    ax1.set_title("Accuracy vs depth (fixed width)")
    ax1.set_xticks(depths)
    ax1.grid(alpha=0.3)
    ax1.legend()

    for r in results:
        h = r["history"]
        ax2.plot(np.arange(1, len(h["test_acc"]) + 1),
                 np.array(h["test_acc"]) * 100, label=f"depth {r['depth']}")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("test accuracy (%)")
    ax2.set_title("Test accuracy over training")
    ax2.grid(alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"\nSaved study to {args.out}")


if __name__ == "__main__":
    main()
