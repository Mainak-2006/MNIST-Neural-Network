"""Train the from-scratch MLP on MNIST.

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

Run:  python train.py                 (defaults: 2 hidden layers, SGD)
      python train.py --optimizer adam --lr 1e-3 --epochs 10
"""

import argparse
import os
import time

import numpy as np

import data
from network import MLP, make_optimizer


def parse_args():
    p = argparse.ArgumentParser(description="Train an MLP on MNIST from scratch.")
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 128],
                   help="sizes of the hidden layers (default: 256 128)")
    p.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    p.add_argument("--lr", type=float, default=0.1,
                   help="learning rate (sgd default 0.1, adam default 1e-3)")
    p.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr-decay", type=float, default=1.0,
                   help="multiply the learning rate by this after each epoch")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="artifacts")
    return p.parse_args()


def iterate_minibatches(x, y, batch_size, rng):
    """Yield shuffled mini-batches (x_batch, y_batch) for one epoch."""
    idx = rng.permutation(len(x))
    for start in range(0, len(x), batch_size):
        batch = idx[start:start + batch_size]
        yield x[batch], y[batch]


def main():
    args = parse_args()
    if args.optimizer == "adam" and args.lr == 0.1:
        args.lr = 1e-3                      # friendlier default for Adam

    print(f"Loading MNIST ...")
    (x_train, y_train), (x_test, y_test) = data.load()
    print(f"  train: {x_train.shape}   test: {x_test.shape}")

    layer_sizes = [x_train.shape[1], *args.hidden, 10]
    model = MLP(layer_sizes, seed=args.seed)
    optimizer = make_optimizer(args.optimizer, model.weight_layers, lr=args.lr,
                               momentum=args.momentum,
                               weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)

    print(f"Model: {' -> '.join(map(str, layer_sizes))}")
    print(f"Optimiser: {args.optimizer}, lr={args.lr}, batch={args.batch_size}\n")

    os.makedirs(args.out_dir, exist_ok=True)
    history = {"train_loss": [], "train_acc": [], "test_acc": []}
    lr = args.lr

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        running_loss, seen = 0.0, 0

        for xb, yb in iterate_minibatches(x_train, y_train, args.batch_size, rng):
            loss = model.loss(xb, yb)       # forward + softmax cross-entropy
            model.backward()                # backpropagate the gradients
            optimizer.step()                # update every weight and bias
            running_loss += loss * len(xb)
            seen += len(xb)

        train_loss = running_loss / seen
        train_acc = model.accuracy(x_train, y_train)
        test_acc = model.accuracy(x_test, y_test)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        print(f"epoch {epoch:2d}/{args.epochs}  "
              f"loss {train_loss:.4f}  "
              f"train_acc {train_acc * 100:5.2f}%  "
              f"test_acc {test_acc * 100:5.2f}%  "
              f"({time.time() - t0:.1f}s)")

        lr *= args.lr_decay                 # optional learning-rate decay
        optimizer.lr = lr

    model_path = os.path.join(args.out_dir, "model.npz")
    hist_path = os.path.join(args.out_dir, "history.npz")
    model.save(model_path)
    np.savez(hist_path, **history)
    print(f"\nSaved model to   {model_path}")
    print(f"Saved history to {hist_path}")
    print(f"Final test accuracy: {history['test_acc'][-1] * 100:.2f}%")


if __name__ == "__main__":
    main()
