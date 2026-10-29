"""Animate the model classifying a whole batch of digits at once.

Where animate_training.py shows the network *learning*, this shows it
*doing inference*. All N digits are predicted in a single batched forward
pass (the point of predict.py), but we reveal the results one digit at a
time so you can watch:

  * the 10x10 grid filling up, each digit boxed green (correct) or red
    (wrong) with a "true -> predicted" caption;
  * a probability bar chart for the digit currently being revealed;
  * a running accuracy counter climbing toward the final score.

Exports both a GIF and an MP4.

Run:  python animate_predict.py
      python animate_predict.py --n 100 --seed 0 --fps 15
"""

import argparse
import math
import os
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.patches import Rectangle

import data
from network import MLP

GREEN = "#2ca02c"
RED = "#d62728"
BLUE = "#1f77b4"
COLS = 10


def parse_args():
    p = argparse.ArgumentParser(description="Animate batch prediction.")
    p.add_argument("--n", type=int, default=100, help="number of digits")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--first", action="store_true",
                   help="use the first N test digits instead of a random sample")
    p.add_argument("--model", default="artifacts/model.npz")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--hold", type=float, default=2.5,
                   help="seconds to hold the final frame")
    p.add_argument("--out-dir", default="artifacts")
    return p.parse_args()


def main():
    args = parse_args()
    (x_test, y_test) = data.load(one_hot_labels=False)[1]
    n = min(args.n, len(x_test))

    if args.first:
        idx = np.arange(n)
    else:
        idx = np.random.default_rng(args.seed).permutation(len(x_test))[:n]

    images = x_test[idx].reshape(-1, 28, 28)
    labels = y_test[idx]

    model = MLP.load(args.model)

    # The whole batch goes through in ONE forward pass.
    probs = model.predict_proba(x_test[idx])       # (n, 10)
    preds = probs.argmax(axis=1)
    correct = preds == labels
    print(f"Predicted {n} digits in one batched forward pass: "
          f"{correct.sum()}/{n} correct ({correct.mean() * 100:.2f}%)")

    # --- figure layout: an N-cell grid plus a probability bar chart --------
    rows = int(math.ceil(n / COLS))
    fig = plt.figure(figsize=(12, 2 + rows * 1.05 + 3))
    gs = fig.add_gridspec(rows + 3, COLS, height_ratios=[1] * rows + [1, 1, 1],
                          hspace=0.35, wspace=0.05,
                          left=0.04, right=0.98, top=0.92, bottom=0.07)

    ims, borders, captions = [], [], []
    for i in range(n):
        r, c = divmod(i, COLS)
        ax = fig.add_subplot(gs[r, c])
        im = ax.imshow(images[i], cmap="gray", vmin=0, vmax=1, alpha=0.0)
        ax.set_xticks([])
        ax.set_yticks([])
        border = Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False,
                           lw=2, edgecolor=GREEN, clip_on=False, visible=False)
        ax.add_patch(border)
        cap = ax.set_title("", fontsize=7, visible=False)
        ims.append(im)
        borders.append(border)
        captions.append(cap)

    ax_bar = fig.add_subplot(gs[rows:rows + 3, :])
    ax_bar.set_xlim(-0.5, 9.5)
    ax_bar.set_ylim(0, 1.0)
    ax_bar.set_xticks(range(10))
    ax_bar.set_xticklabels(range(10))
    ax_bar.set_xlabel("class")
    ax_bar.set_ylabel("probability")
    bars = ax_bar.bar(range(10), np.zeros(10), color=BLUE)
    bar_title = ax_bar.set_title("", fontsize=12)
    suptitle = fig.suptitle("", fontsize=14)

    hold = int(args.hold * args.fps)
    total = n + hold

    def update(f):
        revealed = min(f + 1, n)
        for i in range(n):
            vis = i < revealed
            ims[i].set_alpha(1.0 if vis else 0.0)
            borders[i].set_visible(vis)
            captions[i].set_visible(vis)
            if vis:
                color = GREEN if correct[i] else RED
                borders[i].set_edgecolor(color)
                captions[i].set_text(f"{int(labels[i])}\u2192{int(preds[i])}")
                captions[i].set_color(color)

        j = revealed - 1
        pj = probs[j]
        ok = bool(correct[j])
        for c in range(10):
            bars[c].set_height(pj[c])
            bars[c].set_color(GREEN if (c == preds[j] and ok)
                              else RED if c == preds[j] else BLUE)
        bar_title.set_text(
            f"digit #{idx[j]}   true {int(labels[j])}   predicted "
            f"{int(preds[j])}   (p={pj[preds[j]]:.2f})")

        n_ok = int(correct[:revealed].sum())
        suptitle.set_text(
            f"Testing {revealed}/{n} digits at once  \u2014  "
            f"{n_ok}/{revealed} correct ({n_ok / revealed * 100:.1f}%)"
            + (f"    FINAL: {correct.sum()}/{n} = "
               f"{correct.mean() * 100:.2f}%" if revealed == n else ""))
        return ims + borders + captions + list(bars) + [bar_title, suptitle]

    ani = FuncAnimation(fig, update, frames=total,
                        interval=1000 / args.fps, blit=False, repeat=False)

    os.makedirs(args.out_dir, exist_ok=True)

    gif_path = os.path.join(args.out_dir, "predict.gif")
    print(f"Writing {gif_path} ...")
    ani.save(gif_path, writer=PillowWriter(fps=args.fps))

    if shutil.which("ffmpeg"):
        mp4_path = os.path.join(args.out_dir, "predict.mp4")
        print(f"Writing {mp4_path} ...")
        ani.save(mp4_path, writer=FFMpegWriter(fps=args.fps, bitrate=3000))
    else:
        print("ffmpeg not found; skipped MP4.")

    print("Done.")


if __name__ == "__main__":
    main()
