"""Test the trained network on many digits at once, or on your own image.

Two modes:

  Batch mode (default): run the model over N test digits in a SINGLE
  batched forward pass and report accuracy, per-class accuracy, and the
  confidently-wrong cases. Also saves a labelled grid image.

      python predict.py                 # random 100 test digits
      python predict.py --n 500 --seed 1

  Image mode: load one or more PNG/JPG files, preprocess them into the
  same format as MNIST, and predict.

      python predict.py --image my_digit.png
      python predict.py --image a.png b.jpg --center

The whole point is that inference is vectorised: passing an (N, 784)
array does all N predictions in one set of matrix multiplies, so 100 (or
10,000) digits cost exactly the same code as one.
"""

import argparse
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import data
from network import MLP

GREEN = "#2ca02c"
RED = "#d62728"


def parse_args():
    p = argparse.ArgumentParser(description="Test the MNIST model.")
    p.add_argument("--model", default="artifacts/model.npz")
    p.add_argument("--n", type=int, default=100, help="digits in batch mode")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--first", action="store_true",
                   help="use the first N test digits instead of a random sample")
    p.add_argument("--image", nargs="+", metavar="PATH",
                   help="predict on image file(s) instead of the test set")
    p.add_argument("--invert", action="store_true",
                   help="force inverting the image (use for dark-on-light scans)")
    p.add_argument("--no-invert", action="store_true",
                   help="never invert; assume the image is already white-on-black")
    p.add_argument("--center", action="store_true",
                   help="center the digit by its centre of mass (matches MNIST)")
    p.add_argument("--out-dir", default="artifacts")
    p.add_argument("--save-preview", action="store_true",
                   help="save the exact 28x28 tensor(s) the model sees")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------
def run_batch(args, model):
    (x_test, y_test) = data.load(one_hot_labels=False)[1]
    n = min(args.n, len(x_test))

    if args.first:
        idx = np.arange(n)
    else:
        idx = np.random.default_rng(args.seed).permutation(len(x_test))[:n]

    x_batch = x_test[idx]                     # (n, 784)
    labels = y_test[idx]

    probs = model.predict_proba(x_batch)      # (n, 10)  <-- one batched pass
    preds = probs.argmax(axis=1)
    conf = probs[np.arange(n), labels]        # probability of the true class
    correct = preds == labels

    print(f"Testing {n} {'first' if args.first else 'random'} test digits"
          f"{'' if args.first else f' (seed {args.seed})'}")
    print(f"\nOverall accuracy: {correct.mean() * 100:.2f}%  "
          f"({correct.sum()}/{n})\n")

    print("Per-class accuracy:")
    for c in range(10):
        m = labels == c
        if m.sum():
            print(f"  {c}: {correct[m].sum():>3}/{m.sum():<3} "
                  f"{correct[m].mean() * 100:5.1f}%")

    wrong = np.where(~correct)[0]
    print(f"\nMispredicted ({len(wrong)}):")
    for i in wrong:
        print(f"  [#{idx[i]:>5}] true {labels[i]} -> pred {preds[i]}  "
              f"(p_true={conf[i]:.3f})")

    out = os.path.join(args.out_dir, f"test_{n}.png")
    save_grid(x_batch.reshape(-1, 28, 28), labels, preds, out,
              title=f"{correct.sum()}/{n} correct ({correct.mean() * 100:.2f}%)")
    print(f"\nSaved grid to {out}")


def save_grid(images, labels, preds, out_path, title):
    n = len(images)
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * 1.0, rows * 1.0 + 0.6))
    axes = np.atleast_1d(axes).ravel()

    for ax in axes:
        ax.axis("off")
    for i, ax in enumerate(axes[:n]):
        ok = preds[i] == labels[i]
        color = GREEN if ok else RED
        ax.imshow(images[i], cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(color)
            spine.set_linewidth(2)
        ax.set_title(f"{labels[i]}\u2192{preds[i]}", fontsize=8, color=color)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Image mode
# ---------------------------------------------------------------------------
def center_by_mass(a):
    """Shift the digit so its centre of mass sits at the image centre.

    MNIST digits are centred this way; matching it helps a lot for photos
    where the digit is off-centre.
    """
    total = a.sum()
    if total == 0:
        return a
    ys, xs = np.indices(a.shape)
    cy = (ys * a).sum() / total
    cx = (xs * a).sum() / total
    return np.roll(np.roll(a, int(round(14 - cy)), axis=0),
                   int(round(14 - cx)), axis=1)


def load_image(path, invert=None, center=False):
    """Read an image file and turn it into a (784,) MNIST-style vector."""
    from PIL import Image
    img = Image.open(path).convert("L").resize((28, 28), Image.LANCZOS)
    a = np.asarray(img, dtype=np.float32)

    # MNIST is white digit on black background. If the image is mostly
    # bright, it is probably dark-on-light, so invert it. `invert=None`
    # means auto-detect from the average brightness.
    a = np.asarray(img, dtype=np.float32)
    if invert is None:
        invert = a.mean() > 127.0
    if invert:
        a = 255.0 - a
    a /= 255.0
    if center:
        a = center_by_mass(a)
    return a.reshape(-1)


def run_images(args, model):
    if args.invert:
        invert = True
    elif args.no_invert:
        invert = False
    else:
        invert = None                          # auto-detect per image

    os.makedirs(args.out_dir, exist_ok=True)
    for path in args.image:
        vec = load_image(path, invert=invert, center=args.center)
        probs = model.predict_proba(vec.reshape(1, -1))[0]
        pred = int(probs.argmax())

        print(f"\n{path}")
        print(f"  predicted: {pred}")
        order = probs.argsort()[::-1]
        print("  probabilities:")
        for c in order:
            bar = "#" * int(round(probs[c] * 40))
            print(f"    {c}: {probs[c]:6.3f} {bar}")

        if args.save_preview:
            prev = os.path.join(
                args.out_dir, f"preview_{os.path.splitext(os.path.basename(path))[0]}.png")
            fig, ax = plt.subplots(figsize=(2.2, 2.2))
            ax.imshow(vec.reshape(28, 28), cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"pred {pred}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.tight_layout()
            fig.savefig(prev, dpi=130)
            plt.close(fig)
            print(f"  saved preview: {prev}")


def main():
    args = parse_args()
    model = MLP.load(args.model)
    if args.image:
        run_images(args, model)
    else:
        run_batch(args, model)


if __name__ == "__main__":
    main()
