"""A feed-forward neural network written from scratch in NumPy.

Nothing here is hidden: the forward pass, backpropagation, and the
parameter update are all spelled out so you can read the math directly
in the code. The model is a multi-layer perceptron (MLP):

    input(784) -> Linear -> ReLU -> Linear -> ReLU -> ... -> Linear(10)
                 -> softmax + cross-entropy loss

--------------------------------------------------------------------------
THE MATH, END TO END
--------------------------------------------------------------------------
Notation: X is a batch of images, shape (B, D_in). For one Linear layer
with weight W (D_in, D_out) and bias b (D_out,):

    forward:   Z = X W + b
    ReLU:      A = max(0, Z)
    softmax:   p_k = exp(z_k) / sum_j exp(z_j)
    loss:      L = -(1/B) * sum_i sum_k  y_ik * log(p_ik)

Backprop is just the chain rule applied layer by layer, from the loss
back to the first weight. For the softmax + cross-entropy combination
the derivative with respect to the logits is famously simple:

    dL/dZ  = (p - y) / B

Then, for a Linear layer given dL/dZ coming from the layer above:

    dL/dW  = X^T @ dL/dZ          (shape D_in x D_out)
    dL/db  = sum over batch of dL/dZ
    dL/dX  = dL/dZ @ W^T          (passed down as the next dL/dZ)

and ReLU simply gates the gradient with its own derivative:

    dL/dZ  = dL/dA * (Z > 0)

That is the entire algorithm.
--------------------------------------------------------------------------
"""

import numpy as np


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------
class Linear:
    """Fully-connected layer: Z = X @ W + b.

    Shapes: X (B, in_dim), W (in_dim, out_dim), b (out_dim,), Z (B, out_dim).
    """

    def __init__(self, in_dim, out_dim, rng):
        # He initialisation: std = sqrt(2 / in_dim).
        # ReLU zeroes roughly half its inputs; drawing weights with this
        # standard deviation keeps the *variance* of the activations
        # approximately constant from layer to layer, so signals neither
        # explode nor vanish as they flow through the network.
        self.W = rng.normal(0.0, np.sqrt(2.0 / in_dim),
                            size=(in_dim, out_dim)).astype(np.float32)
        self.b = np.zeros(out_dim, dtype=np.float32)
        # Caches filled during forward(), used by backward().
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x                       # remember input for the backward pass
        return x @ self.W + self.b

    def backward(self, dZ):
        # dZ is dL/dZ, same shape as the forward output.
        self.dW = self.x.T @ dZ          # dL/dW = X^T dZ
        self.db = dZ.sum(axis=0)         # dL/db = sum_batch dZ
        return dZ @ self.W.T             # dL/dX = dZ W^T, handed to the layer below


class ReLU:
    """Elementwise rectifier: f(z) = max(0, z), f'(z) = 1 if z > 0 else 0."""

    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, dA):
        # Chain rule: upstream gradient times the local derivative (the mask).
        return dA * self.mask


class Dropout:
    """Randomly zeroes each activation with probability `p` during training.

    Dropout is a regulariser: by hiding a random subset of units every step,
    no unit can rely on a particular partner being present, so the network
    learns redundant, more robust features and overfits less.

    We use *inverted* dropout: at train time, units that survive are scaled
    by 1 / (1 - p) so the expected activation is unchanged. That means
    inference is just the identity -- no scaling bookkeeping at test time.

        train:  mask ~ Bernoulli(1 - p);  out = x * mask / (1 - p)
        eval:   out = x

    The backward pass re-uses the same mask and scale, routing gradient only
    through the units that were kept.
    """

    def __init__(self, p=0.5, rng=None):
        self.p = p
        self.rng = rng or np.random.default_rng(0)
        self.training = True
        self.mask = None

    def train(self, mode=True):
        self.training = mode
        return self

    def eval(self):
        return self.train(False)

    def forward(self, x):
        if not self.training or self.p <= 0.0:
            self.mask = None
            return x
        keep = 1.0 - self.p
        self.mask = (self.rng.random(x.shape) < keep) / keep
        return x * self.mask

    def backward(self, dA):
        if self.mask is None:
            return dA
        return dA * self.mask


class SoftmaxCrossEntropy:
    """Softmax followed by cross-entropy loss, computed together.

    Combining them (a) lets us subtract the max logit for numerical
    stability, and (b) yields the clean gradient dL/dZ = (p - y) / B.

    y is one-hot, so cross-entropy reduces to -log(p_correct), and this
    gradient means: push the probability of the true class up and every
    other class down, in proportion to the current error.
    """

    def __init__(self):
        self.probs = None
        self.y = None

    def forward(self, logits, y):
        # Subtract the per-row max before exp so we never overflow. This is
        # mathematically a no-op because softmax is shift-invariant.
        z = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(z)
        self.probs = exp / exp.sum(axis=1, keepdims=True)
        self.y = y
        eps = 1e-12                       # guard against log(0)
        self.loss = -np.sum(y * np.log(self.probs + eps)) / len(logits)
        return self.loss

    def backward(self):
        return (self.probs - self.y) / len(self.y)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class MLP:
    """A stack of Linear layers separated by ReLU, ending in softmax+CE.

    Example: MLP([784, 256, 128, 10]) builds
        784 -> Linear+ReLU -> Linear+ReLU -> Linear(10) -> softmax+CE
    Note the last Linear has no ReLU before the loss.

    With dropout > 0 a Dropout layer is inserted after every hidden ReLU
    (never before the logits). Call .train() / .eval() to switch dropout on
    and off; inference methods do this automatically.
    """

    def __init__(self, layer_sizes, seed=0, dropout=0.0):
        rng = np.random.default_rng(seed)
        self.layer_sizes = list(layer_sizes)
        self.dropout = float(dropout)
        self.training = True
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Linear(layer_sizes[i], layer_sizes[i + 1], rng))
            if i < len(layer_sizes) - 2:      # no ReLU after the final layer
                self.layers.append(ReLU())
                if self.dropout > 0.0:
                    # Dropout only on the hidden activations, never the logits.
                    self.layers.append(Dropout(self.dropout, rng))
        self.loss_fn = SoftmaxCrossEntropy()

    # -- shapes helpers -----------------------------------------------------
    @property
    def weight_layers(self):
        return [l for l in self.layers if isinstance(l, Linear)]

    @property
    def dropout_layers(self):
        return [l for l in self.layers if isinstance(l, Dropout)]

    # -- train / eval mode --------------------------------------------------
    def train(self, mode=True):
        """Turn dropout (and any future stochastic layers) on or off."""
        self.training = mode
        for layer in self.dropout_layers:
            layer.training = mode
        return self

    def eval(self):
        return self.train(False)

    # -- forward / backward -------------------------------------------------
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x                              # logits

    def loss(self, x, y):
        return self.loss_fn.forward(self.forward(x), y)

    def backward(self):
        d = self.loss_fn.backward()           # dL/dZ of the final Linear
        for layer in reversed(self.layers):
            d = layer.backward(d)

    def predict(self, x):
        # argmax of the logits equals argmax of the softmax probabilities.
        return self._eval_forward(x).argmax(axis=1)

    def predict_proba(self, x):
        """Return class probabilities, shape (N, 10).

        Applies a numerically stable softmax to the logits. Works on any
        batch size N in a single vectorised forward pass. Always runs in
        eval mode so dropout never perturbs a prediction.
        """
        logits = self._eval_forward(x)
        z = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def _eval_forward(self, x):
        """Forward pass with dropout disabled, restoring the previous mode."""
        was_training = self.training
        self.eval()
        out = self.forward(x)
        self.train(was_training)
        return out


    def accuracy(self, x, y):
        if y.ndim > 1:
            y = y.argmax(axis=1)
        return float((self.predict(x) == y).mean())

    # -- persistence --------------------------------------------------------
    def save(self, path):
        arrays = {}
        for i, layer in enumerate(self.weight_layers):
            arrays[f"W{i}"] = layer.W
            arrays[f"b{i}"] = layer.b
        arrays["layer_sizes"] = np.array(self.layer_sizes)
        arrays["dropout"] = np.array(self.dropout)
        arrays["kind"] = np.array("mlp")
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        dropout = float(data["dropout"]) if "dropout" in data else 0.0
        model = cls(data["layer_sizes"].tolist(), dropout=dropout)
        for i, layer in enumerate(model.weight_layers):
            layer.W = data[f"W{i}"]
            layer.b = data[f"b{i}"]
        return model


# ---------------------------------------------------------------------------
# Optimisers
# ---------------------------------------------------------------------------
class SGDMomentum:
    """Stochastic gradient descent with momentum.

        v     <- momentum * v - lr * grad
        param <- param + v

    Momentum keeps a running average of past gradients. In a long, narrow
    valley of the loss surface, the across-valley components oscillate and
    cancel while the along-valley components reinforce, so we move faster
    and more smoothly toward the minimum.

    With nesterov=True we use Nesterov accelerated momentum: we take the
    step using the gradient evaluated a little *ahead* of the current point,
    which looks before it leaps and damps overshoot near the minimum.

        v     <- momentum * v + grad
        param <- param - lr * (grad + momentum * v)

    (Here `grad` is still the gradient at the current parameter, matching
    the single forward/backward pass per step that the training loop does.)

    weight_decay adds L2 regularisation: it shrinks weights slightly each
    step, discouraging large weights and reducing overfitting.
    """

    def __init__(self, params, lr=0.1, momentum=0.9, weight_decay=0.0,
                 nesterov=False):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.v = {id(p): (np.zeros_like(p.W), np.zeros_like(p.b)) for p in params}

    def step(self):
        for p in self.params:
            gW = p.dW + self.weight_decay * p.W
            gb = p.db
            vW, vb = self.v[id(p)]
            if self.nesterov:
                vW = self.momentum * vW + gW
                vb = self.momentum * vb + gb
                p.W -= self.lr * (gW + self.momentum * vW)
                p.b -= self.lr * (gb + self.momentum * vb)
            else:
                vW = self.momentum * vW - self.lr * gW
                vb = self.momentum * vb - self.lr * gb
                p.W += vW
                p.b += vb
            self.v[id(p)] = (vW, vb)


class Adam:
    """Adam: per-parameter adaptive step sizes.

        m <- b1*m + (1-b1)*g            (exponential average of gradients)
        v <- b2*v + (1-b2)*g**2         (exponential average of squared grads)
        m_hat <- m / (1 - b1**t)        (bias correction for the zero init)
        v_hat <- v / (1 - b2**t)
        p <- p - lr * m_hat / (sqrt(v_hat) + eps)

    Dividing by sqrt(v) gives each parameter its own effective learning
    rate: noisy, large-gradient parameters take smaller steps. The bias
    correction compensates for m and v starting at zero.
    """

    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8,
                 weight_decay=0.0):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = {id(p): (np.zeros_like(p.W), np.zeros_like(p.b)) for p in params}
        self.v = {id(p): (np.zeros_like(p.W), np.zeros_like(p.b)) for p in params}

    def step(self):
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        bc1 = 1 - b1 ** self.t
        bc2 = 1 - b2 ** self.t
        for p in self.params:
            mW, mb = self.m[id(p)]
            vW, vb = self.v[id(p)]
            gW = p.dW + self.weight_decay * p.W
            gb = p.db

            mW = b1 * mW + (1 - b1) * gW
            mb = b1 * mb + (1 - b1) * gb
            vW = b2 * vW + (1 - b2) * gW ** 2
            vb = b2 * vb + (1 - b2) * gb ** 2

            p.W -= self.lr * (mW / bc1) / (np.sqrt(vW / bc2) + self.eps)
            p.b -= self.lr * (mb / bc1) / (np.sqrt(vb / bc2) + self.eps)

            self.m[id(p)] = (mW, mb)
            self.v[id(p)] = (vW, vb)


def make_optimizer(name, params, lr, momentum=0.9, weight_decay=0.0,
                   nesterov=False):
    name = name.lower()
    if name == "sgd":
        return SGDMomentum(params, lr=lr, momentum=momentum,
                           weight_decay=weight_decay, nesterov=nesterov)
    if name == "adam":
        return Adam(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"unknown optimiser: {name!r} (use 'sgd' or 'adam')")
