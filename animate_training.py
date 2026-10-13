"""Animate a full MNIST training run and export it as an MP4 and a GIF.

This script trains the from-scratch MLP while periodically capturing a
frame. Each frame shows two things:

  * a grid of 24 fixed test digits. The coloured border and the caption
    "true -> predicted" update as the network learns; wrong predictions
    are red, correct ones green.
  * the training loss and test accuracy curves, drawn up to the current
    point in training, with a marker on the latest value.

Watching the boxes flip from red to green as the accuracy curve climbs
from ~10% (chance) toward ~98% is a visceral view of gradient descent at
work.

Run:  python animate_training.py
      python animate_training.py --epochs 6 --fps 12 --capture-every 25
"""

import argparse
import os
import shutil
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")                       # render to files, no display needed
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.patches import Rectangle

import data
from network import MLP, SGDMomentum

GREEN = "#2ca02c"
RED = "#d62728"


def parse_args():
    p = argparse.ArgumentParser(description="Animate MNIST training.")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--hidden", type=int, nargs="+", default=[128])
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--capture-every", type=int, default=25,
                   help="capture a frame every N mini-batches")
    p.add_argument("--samples", type=int, default=24,
                   help="number of test digits shown in the grid")
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="artifacts")
    return p.parse_args()


def softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def train_and_capture(args):
    """Train the model, recording a frame every `capture_every` batches."""
    (x_train, y_train), (x_test, y_test) = data.load()
    x_test_img, y_test_int = data.load(one_hot_labels=False)[1]

    rng = np.random.default_rng(args.seed)
    model = MLP([x_train.shape[1], *args.hidden, 10], seed=args.seed)
    optimizer = SGDMomentum(model.weight_layers, lr=args.lr,
                            momentum=args.momentum)

    # A fixed set of digits we watch across the whole run.
    sample_idx = rng.permutation(len(x_test))[:args.samples]
    x_show = x_test[sample_idx]
    y_show = y_test_int[sample_idx]
    images = x_test_img[sample_idx].reshape(-1, 28, 28)

    batches_per_epoch = int(np.ceil(len(x_train) / args.batch_size))
    total_batches = batches_per_epoch * args.epochs
    frames, seen, running_loss = [], 0, 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        idx = rng.permutation(len(x_train))
        for start in range(0, len(x_train), args.batch_size):
            batch = idx[start:start + args.batch_size]
            xb, yb = x_train[batch], y_train[batch]

            loss = model.loss(xb, yb)       # forward
            model.backward()                # backprop
            optimizer.step()                # update
            running_loss += loss * len(xb)
            seen += len(xb)

            done = (epoch - 1) * batches_per_epoch + start // args.batch_size + 1
            if done % args.capture_every == 0 or done == total_batches:
                probs = softmax(model.forward(x_show))
                preds = probs.argmax(axis=1)
                frames.append({
                    "progress": done / batches_per_epoch,   # in epochs
                    "loss": running_loss / seen,
                    "test_acc": model.accuracy(x_test, y_test),
                    "preds": preds,
                    "correct": preds == y_show,
                })

        print(f"epoch {epoch}/{args.epochs}  "
              f"loss {frames[-1]['loss']:.4f}  "
              f"test acc {frames[-1]['test_acc'] * 100:5.2f}%  "
              f"frames {len(frames)}  ({time.time() - t0:.1f}s)")

    return images, y_show, frames


def build_figure(images, y_show, epochs):
    """Create the figure once. Returns the artists the update() will edit."""
    k = len(images)
    cols = 6
    rows = int(np.ceil(k / cols))

    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(rows + 2, cols, height_ratios=[1] * rows + [1.4, 1.4],
                          hspace=0.55, wspace=0.12,
                          left=0.06, right=0.96, top=0.9, bottom=0.08)

    borders, titles = [], []
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        border = Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False,
                           lw=3, edgecolor=GREEN, clip_on=False)
        ax.add_patch(border)
        borders.append(border)
        titles.append(ax.set_title("", fontsize=9))

    ax_loss = fig.add_subplot(gs[rows:rows + 2, :])
    ax_acc = ax_loss.twinx()
    ax_loss.set_xlim(0, epochs)
    ax_loss.set_ylim(0, 2.5)
    ax_acc.set_ylim(0, 100)
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("train loss", color="tab:red")
    ax_acc.set_ylabel("test accuracy (%)", color="tab:blue")
    ax_loss.tick_params(axis="y", labelcolor="tab:red")
    ax_acc.tick_params(axis="y", labelcolor="tab:blue")
    ax_acc.axhline(10, color="gray", ls="--", lw=1, alpha=0.6)   # chance level
    ln_loss, = ax_loss.plot([], [], color="tab:red", lw=2)
    ln_acc, = ax_acc.plot([], [], color="tab:blue", lw=2)
    pt_loss, = ax_loss.plot([], [], "o", color="tab:red", ms=8)
    pt_acc, = ax_acc.plot([], [], "o", color="tab:blue", ms=8)
    suptitle = fig.suptitle("", fontsize=13)

    artists = dict(fig=fig, borders=borders, titles=titles, ln_loss=ln_loss,
                   ln_acc=ln_acc, pt_loss=pt_loss, pt_acc=pt_acc,
                   suptitle=suptitle)
    return artists


def main():
    args = parse_args()
    images, y_show, frames = train_and_capture(args)
    print(f"Captured {len(frames)} frames.")

    # Hold the final frame for a couple of seconds at the end of the video.
    frames = frames + [frames[-1]] * (args.fps * 2)

    a = build_figure(images, y_show, args.epochs)
    progress = np.array([f["progress"] for f in frames])
    losses = np.array([f["loss"] for f in frames])
    accs = np.array([f["test_acc"] * 100 for f in frames])

    def update(f):
        fr = frames[f]
        for i, border in enumerate(a["borders"]):
            ok = fr["correct"][i]
            border.set_edgecolor(GREEN if ok else RED)
            a["titles"][i].set_text(f"{int(y_show[i])}\u2192{int(fr['preds'][i])}")
            a["titles"][i].set_color(GREEN if ok else RED)
        x = progress[:f + 1]
        a["ln_loss"].set_data(x, losses[:f + 1])
        a["ln_acc"].set_data(x, accs[:f + 1])
        a["pt_loss"].set_data([progress[f]], [losses[f]])
        a["pt_acc"].set_data([progress[f]], [accs[f]])
        a["suptitle"].set_text(
            f"epoch {fr['progress']:4.1f}    train loss {fr['loss']:.3f}    "
            f"test accuracy {fr['test_acc'] * 100:5.2f}%    "
            f"{int(fr['correct'].sum())}/{len(y_show)} shown correct")
        return (a["borders"] + a["titles"] +
                [a["ln_loss"], a["ln_acc"], a["pt_loss"], a["pt_acc"],
                 a["suptitle"]])

    ani = FuncAnimation(a["fig"], update, frames=len(frames),
                        interval=1000 / args.fps, blit=False)

    os.makedirs(args.out_dir, exist_ok=True)

    gif_path = os.path.join(args.out_dir, "training.gif")
    print(f"Writing {gif_path} ...")
    ani.save(gif_path, writer=PillowWriter(fps=args.fps))

    if shutil.which("ffmpeg"):
        mp4_path = os.path.join(args.out_dir, "training.mp4")
        print(f"Writing {mp4_path} ...")
        ani.save(mp4_path, writer=FFMpegWriter(fps=args.fps, bitrate=2400))
    else:
        print("ffmpeg not found; skipped MP4.")

    print("Done.")


if __name__ == "__main__":
    main()
