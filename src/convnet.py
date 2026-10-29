"""A small convolutional network, also from scratch in NumPy.

The MLP in network.py flattens each image into 784 numbers and loses all
notion of which pixel sits next to which. A convolutional network keeps the
2-D layout and slides a small learnable filter across it, so it can detect
local patterns (edges, strokes) wherever they appear. This is exactly the
structure MNIST rewards.

Architecture (default):

    1x28x28
      -> Conv2d(8, 3x3, pad=1) -> ReLU -> MaxPool2d(2)     # 8x14x14
      -> Conv2d(16, 3x3, pad=1) -> ReLU -> MaxPool2d(2)    # 16x7x7
      -> Flatten                                            # 784
      -> Linear(128) -> ReLU -> Linear(10) -> softmax+CE

--------------------------------------------------------------------------
THE MATH
--------------------------------------------------------------------------
Convolution. For a filter W of shape (C_out, C_in, k, k) and input X of
shape (B, C_in, H, W), the output at spatial location (oh, ow) is

    Z[b, o, oh, ow] = sum_{c,i,j} W[o, c, i, j] * Xpad[b, c, oh*s+i, ow*s+j] + bias[o]

We implement the forward pass with *im2col*: gather every k x k patch into a
column, then the convolution becomes one matrix multiply

    cols  (B, C_in*k*k, OH*OW)
    Zr    (B, C_out,   OH*OW) = Wr @ cols + bias      with Wr = W.reshape(C_out, -1)

which reuses fast BLAS matmuls instead of Python loops. The backward pass is
then the usual Linear-like rules, followed by col2im to scatter the input
gradient back into place.

Max pooling. A 2x2 max pool takes the largest value in each window and
routes the gradient back to the *argmax* location only -- the maximum is
locally linear in exactly one input.

ReLU, the final Linear layers, and softmax + cross-entropy are reused
verbatim from network.py; they are shape-agnostic once the feature maps are
flattened.
--------------------------------------------------------------------------
"""

import numpy as np
from numpy.lib.stride_tricks import as_strided

from network import Linear, ReLU, SoftmaxCrossEntropy


def _im2col(x, k, stride, oh, ow):
    """Gather (B, C, oh, ow) windows of size k x k into (B, C*k*k, oh*ow)."""
    B, C, H, W = x.shape
    shape = (B, C, k, k, oh, ow)
    strides = (x.strides[0], x.strides[1],
               x.strides[2], x.strides[3],
               stride * x.strides[2], stride * x.strides[3])
    cols = as_strided(x, shape=shape, strides=strides)
    return cols.reshape(B, C * k * k, oh * ow)


def _col2im(dcols, xp_shape, k, stride, oh, ow, pad):
    """Inverse of _im2col: accumulate patch gradients back into the image."""
    B, C, Hp, Wp = xp_shape
    dcols = dcols.reshape(B, C, k, k, oh, ow)
    dxp = np.zeros((B, C, Hp, Wp), dtype=dcols.dtype)
    for i in range(k):
        for j in range(k):
            dxp[:, :, i:i + stride * (oh - 1) + 1:stride,
                j:j + stride * (ow - 1) + 1:stride] += dcols[:, :, i, j]
    if pad:
        return dxp[:, :, pad:pad + Hp - 2 * pad, pad:pad + Wp - 2 * pad]
    return dxp


class Conv2d:
    """2-D convolution with zero padding and stride, via im2col.

    Exposes the same .W/.b/.dW/.db interface as Linear, so the existing
    optimisers in network.py drive it without modification.
    """

    def __init__(self, in_ch, out_ch, k=3, stride=1, pad=0, rng=None):
        rng = rng or np.random.default_rng(0)
        # He initialisation: fan_in is the number of inputs each output sees.
        fan_in = in_ch * k * k
        self.W = rng.normal(0.0, np.sqrt(2.0 / fan_in),
                            size=(out_ch, in_ch, k, k)).astype(np.float32)
        self.b = np.zeros(out_ch, dtype=np.float32)
        self.k, self.stride, self.pad = k, stride, pad
        self.in_ch = in_ch
        self.cols = None
        self.xp_shape = None
        self.oh = self.ow = None

    def forward(self, x):
        B, C, H, W = x.shape
        k, s, p = self.k, self.stride, self.pad
        # Pad the input with zeros on the border.
        xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))
        xp = np.ascontiguousarray(xp)
        oh = (H + 2 * p - k) // s + 1
        ow = (W + 2 * p - k) // s + 1

        self.xp_shape = xp.shape
        self.oh, self.ow = oh, ow
        self.cols = _im2col(xp, k, s, oh, ow)          # (B, C*k*k, oh*ow)

        out_ch = self.W.shape[0]
        Wr = self.W.reshape(out_ch, -1)                # (out_ch, C*k*k)
        zr = np.einsum("oi,bip->bop", Wr, self.cols)   # (B, out_ch, oh*ow)
        zr += self.b[None, :, None]
        return zr.reshape(B, out_ch, oh, ow)

    def backward(self, dZ):
        B, out_ch, oh, ow = dZ.shape
        dZr = dZ.reshape(B, out_ch, oh * ow)
        Wr = self.W.reshape(out_ch, -1)

        self.db = dZr.sum(axis=(0, 2))                          # sum_batch
        dWr = np.einsum("bop,bip->oi", dZr, self.cols)          # (out, C*k*k)
        self.dW = dWr.reshape(self.W.shape)

        dcols = np.einsum("oi,bop->bip", Wr, dZr)               # (B, C*k*k, oh*ow)
        dX = _col2im(dcols, self.xp_shape, self.k, self.stride,
                     self.oh, self.ow, self.pad)
        return dX


class MaxPool2d:
    """Max pooling: take the max of each k x k window (stride defaults to k)."""

    def __init__(self, k=2, stride=None):
        self.k = k
        self.stride = stride or k
        self.x_shape = None
        self.idx = None
        self.oh = self.ow = None

    def forward(self, x):
        B, C, H, W = x.shape
        k, s = self.k, self.stride
        oh = (H - k) // s + 1
        ow = (W - k) // s + 1
        self.x_shape, self.oh, self.ow = x.shape, oh, ow

        xc = np.ascontiguousarray(x)
        shape = (B, C, oh, ow, k, k)
        strides = (xc.strides[0], xc.strides[1],
                   s * xc.strides[2], s * xc.strides[3],
                   xc.strides[2], xc.strides[3])
        windows = as_strided(xc, shape=shape, strides=strides)
        windows = windows.reshape(B, C, oh, ow, k * k)
        self.idx = windows.argmax(axis=-1)            # (B, C, oh, ow)
        return windows.max(axis=-1)

    def backward(self, dZ):
        B, C, H, W = self.x_shape
        k, s = self.k, self.stride
        dx = np.zeros(self.x_shape, dtype=dZ.dtype)
        for i in range(k):
            for j in range(k):
                mask = self.idx == i * k + j
                dx[:, :, i:i + s * (self.oh - 1) + 1:s, j:j + s * (self.ow - 1) + 1:s] += dZ * mask
        return dx


class Flatten:
    """Reshape (B, C, H, W) feature maps into (B, C*H*W) rows for Linear."""

    def __init__(self):
        self.shape = None

    def forward(self, x):
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, d):
        return d.reshape(self.shape)


class ConvNet:
    """A small CNN mirroring the MLP's forward/backward/predict interface."""

    def __init__(self, in_shape=(1, 28, 28), num_classes=10, seed=0,
                 channels=(8, 16), fc=128):
        rng = np.random.default_rng(seed)
        self.in_shape = tuple(in_shape)
        self.num_classes = num_classes
        self.channels = tuple(channels)
        self.fc = fc

        c1, c2 = self.channels
        self.layers = [
            Conv2d(self.in_shape[0], c1, k=3, pad=1, rng=rng),
            ReLU(),
            MaxPool2d(2),
            Conv2d(c1, c2, k=3, pad=1, rng=rng),
            ReLU(),
            MaxPool2d(2),
            Flatten(),
        ]
        # Discover the flattened feature size with a dummy forward pass.
        with np.errstate(over="ignore"):
            flat_dim = int(self._features(np.zeros((1, *self.in_shape),
                                                   dtype=np.float32)).shape[1])
        self.layers += [
            Linear(flat_dim, fc, rng),
            ReLU(),
            Linear(fc, num_classes, rng),
        ]
        self.loss_fn = SoftmaxCrossEntropy()

    def _features(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    # -- shapes helpers -----------------------------------------------------
    @property
    def weight_layers(self):
        # Conv2d and Linear both expose W/b/dW/db, so optimisers work on both.
        return [l for l in self.layers if isinstance(l, (Linear, Conv2d))]

    @property
    def dropout_layers(self):
        return []

    # -- train / eval mode (no stochastic layers yet, kept for parity) ------
    def train(self, mode=True):
        self.training = mode
        return self

    def eval(self):
        return self.train(False)

    # -- forward / backward -------------------------------------------------
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x                                     # logits

    def loss(self, x, y):
        return self.loss_fn.forward(self.forward(x), y)

    def backward(self):
        d = self.loss_fn.backward()
        for layer in reversed(self.layers):
            d = layer.backward(d)

    def predict(self, x):
        return self.forward(x).argmax(axis=1)

    def predict_proba(self, x):
        logits = self.forward(x)
        z = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

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
        arrays["in_shape"] = np.array(self.in_shape)
        arrays["num_classes"] = np.array(self.num_classes)
        arrays["channels"] = np.array(self.channels)
        arrays["fc"] = np.array(self.fc)
        arrays["kind"] = np.array("cnn")
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        model = cls(in_shape=tuple(data["in_shape"]),
                    num_classes=int(data["num_classes"]),
                    channels=tuple(data["channels"]), fc=int(data["fc"]))
        for i, layer in enumerate(model.weight_layers):
            layer.W = data[f"W{i}"]
            layer.b = data[f"b{i}"]
        return model
