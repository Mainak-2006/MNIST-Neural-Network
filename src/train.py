"""Train the from-scratch network(s) on MNIST.

The training loop is the classic mini-batch stochastic gradient descent:

    for each epoch:
        shuffle the training data
        for each mini-batch:
            forward pass  -> logits
            loss          -> softmax cross-entropy
            backward pass -> gradients for every weight and bias
            optimiser step -> nudge the weights down the gradient

Why mini-batches? The full gradient over 60,000 images is expensive and
redundant per step; a single image is far too noisy. A batch of ~128 gives
a good estimate of the true gradient at a fraction of the cost, and the
noise even helps escape shallow traps. Shuffling each epoch keeps batches
roughly representative of the whole dataset.

Two models share this loop:

  * --model mlp  (default) the multi-layer perceptron from network.py,
  * --model cnn            the small convolutional net from convnet.py.

Run:  python train.py                        (MLP, SGD)
      python train.py --optimizer adam --lr 1e-3 --epochs 10
      python train.py --dropout 0.2 --nesterov --lr-schedule cosine
      python train.py --model cnn --optimizer adam --epochs 8
"""

import argparse
import math
import os
import time

import numpy as np

import data
from network import MLP, make_optimizer
from convnet import ConvNet


def parse_args():
    p = argparse.ArgumentParser(description="Train a network on MNIST from scratch.")
    p.add_argument("--model", choices=["mlp", "cnn"], default="mlp")
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 128],
                   help="sizes of the MLP hidden layers (default: 256 128)")
    p.add_argument("--dropout", type=float, default=0.0,
                   help="dropout probability on MLP hidden layers")
    p.add_argument("--channels", type=int, nargs="+", default=[8, 16],
                   help="CNN conv output channels (default: 8 16)")
    p.add_argument("--fc", type=int, default=128, help="CNN hidden Linear width")
    p.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    p.add_argument("--lr", type=float, default=0.1,
                   help="learning rate (sgd default 0.1, adam default 1e-3)")
    p.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    p.add_argument("--nesterov", action="store_true",
                   help="use Nesterov accelerated momentum (SGD only)")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr-schedule", choices=["exp", "step", "cosine", "none"],
                   default="exp",
                   help="exp: x lr-decay each epoch; step: x lr-decay every "
                        "lr-step-size epochs; cosine: anneal to lr-final-frac")
    p.add_argument("--lr-decay", type=float, default=1.0,
                   help="decay factor for the exp/step schedules")
    p.add_argument("--lr-step-size", type=int, default=5,
                   help="epochs between drops for the step schedule")
    p.add_argument("--lr-final-frac", type=float, default=0.0,
                   help="final lr as a fraction of the base for cosine")
    p.add_argument("--subset", type=int, default=0,
                   help="if >0, train on only this many examples (for fast runs)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="artifacts")
    return p.parse_args()


def lr_at(base_lr, epoch, args):
    """Learning rate for a 1-indexed epoch under the chosen schedule."""
    s = args.lr_schedule
    if s == "none":
        return base_lr
    if s == "exp":
        return base_lr * (args.lr_decay ** (epoch - 1))
    if s == "step":
        return base_lr * (args.lr_decay ** ((epoch - 1) // args.lr_step_size))
    if s == "cosine":
        # Divide by `epochs` (not epochs-1) so the last epoch lands just
        # above the floor rather than exactly on it, wasting no updates.
        t = (epoch - 1) / max(1, args.epochs)
        cos = 0.5 * (1.0 + math.cos(math.pi * t))
        return base_lr * (args.lr_final_frac + (1.0 - args.lr_final_frac) * cos)
    raise ValueError(f"unknown schedule: {s!r}")


def iterate_minibatches(x, y, batch_size, rng):
    """Yield shuffled mini-batches (x_batch, y_batch) for one epoch."""
    idx = rng.permutation(len(x))
    for start in range(0, len(x), batch_size):
        batch = idx[start:start + batch_size]
        yield x[batch], y[batch]


def build_model(args, in_shape, num_classes=10):
    """Construct the requested model for the given input shape."""
    if args.model == "cnn":
        return ConvNet(in_shape=in_shape, num_classes=num_classes,
                       seed=args.seed, channels=tuple(args.channels), fc=args.fc)
    return MLP([in_shape[0], *args.hidden, num_classes],
               seed=args.seed, dropout=args.dropout)


def train_model(model, x_train, y_train, x_test, y_test, *,
                optimizer, args, log=print):
    """Run the mini-batch training loop and return (model, history)."""
    rng = np.random.default_rng(args.seed)
    history = {"train_loss": [], "train_acc": [], "test_acc": [], "lr": []}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        lr = lr_at(args.lr, epoch, args)
        optimizer.lr = lr
        model.train()                       # enable dropout while training
        running_loss, seen = 0.0, 0

        for xb, yb in iterate_minibatches(x_train, y_train, args.batch_size, rng):
            loss = model.loss(xb, yb)       # forward + softmax cross-entropy
            model.backward()                # backpropagate the gradients
            optimizer.step()                # update every weight and bias
            running_loss += loss * len(xb)
            seen += len(xb)

        train_loss = running_loss / seen
        train_acc = model.accuracy(x_train, y_train)   # eval mode (dropout off)
        test_acc = model.accuracy(x_test, y_test)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["lr"].append(lr)

        log(f"epoch {epoch:2d}/{args.epochs}  "
            f"loss {train_loss:.4f}  "
            f"train_acc {train_acc * 100:5.2f}%  "
            f"test_acc {test_acc * 100:5.2f}%  "
            f"lr {lr:.4g}  "
            f"({time.time() - t0:.1f}s)")
    return model, history


def main():
    args = parse_args()
    if args.optimizer == "adam" and args.lr == 0.1:
        args.lr = 1e-3                      # friendlier default for Adam

    cnn = args.model == "cnn"
    print("Loading MNIST ...")
    (x_train, y_train), (x_test, y_test) = data.load(flatten=not cnn)
    if cnn:
        # Convolution expects (B, channels, height, width).
        x_train = x_train.reshape(len(x_train), 1, 28, 28)
        x_test = x_test.reshape(len(x_test), 1, 28, 28)
    if args.subset > 0:
        x_train, y_train = x_train[:args.subset], y_train[:args.subset]
    print(f"  train: {x_train.shape}   test: {x_test.shape}")

    in_shape = (784,) if not cnn else (1, 28, 28)
    model = build_model(args, in_shape)
    optimizer = make_optimizer(args.optimizer, model.weight_layers, lr=args.lr,
                               momentum=args.momentum,
                               weight_decay=args.weight_decay,
                               nesterov=args.nesterov)

    if cnn:
        print(f"Model: CNN channels={args.channels} fc={args.fc}")
    else:
        print(f"Model: {' -> '.join(map(str, model.layer_sizes))}"
              f"{f'  dropout={args.dropout}' if args.dropout else ''}")
    print(f"Optimiser: {args.optimizer}, lr={args.lr}, batch={args.batch_size}, "
          f"schedule={args.lr_schedule}"
          f"{', nesterov' if args.nesterov and args.optimizer == 'sgd' else ''}\n")

    os.makedirs(args.out_dir, exist_ok=True)
    model, history = train_model(model, x_train, y_train, x_test, y_test,
                                 optimizer=optimizer, args=args)

    model_path = os.path.join(args.out_dir, "model.npz")
    hist_path = os.path.join(args.out_dir, "history.npz")
    model.save(model_path)
    np.savez(hist_path, **history)
    print(f"\nSaved model to   {model_path}")
    print(f"Saved history to {hist_path}")
    print(f"Final test accuracy: {history['test_acc'][-1] * 100:.2f}%")


if __name__ == "__main__":
    main()
