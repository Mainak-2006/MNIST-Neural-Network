"""Verify backpropagation against numerical gradients.

The best way to trust your backward pass is to compare it with a brute
force numerical derivative. By definition,

    dL/dw  ~=  (L(w + eps) - L(w - eps)) / (2 * eps)

for a tiny eps. If our analytic gradients from backprop match these
"central difference" estimates, the chain rule implementation is correct.

This is checked for two models:

  * the MLP in network.py (Linear + ReLU + softmax cross-entropy), and
  * the small CNN in convnet.py (Conv2d + MaxPool + the same head),

so the im2col convolution and the max-pool gradient routing are verified
too. We check only a handful of weights per tensor because each numerical
estimate costs two extra forward passes.
"""

import numpy as np

from network import MLP
from convnet import ConvNet


def numerical_grads(model, x, y, num=5, eps=1e-5, rng=None):
    """Return {(layer_index, 'W'|'b', flat_index): grad} for a few params."""
    rng = rng or np.random.default_rng(0)
    grads = {}

    def loss_at():
        return model.loss(x, y)

    for li, layer in enumerate(model.weight_layers):
        for name, param in (("W", layer.W), ("b", layer.b)):
            flat = param.reshape(-1)
            idxs = rng.choice(flat.size, size=min(num, flat.size), replace=False)
            for idx in idxs:
                orig = flat[idx]
                flat[idx] = orig + eps
                lp = loss_at()
                flat[idx] = orig - eps
                lm = loss_at()
                flat[idx] = orig
                grads[(li, name, int(idx))] = (lp - lm) / (2 * eps)
    return grads


def check_model(model, x, y, label, tol=1e-4, seed=0):
    """Compare analytic vs numerical gradients for one model."""
    # Run the check in float64. Central differences divide a tiny numerator
    # by 2*eps, so float32 round-off would swamp the signal and produce
    # spurious ~1e-2 relative errors even when backprop is exactly right.
    for layer in model.weight_layers:
        layer.W = layer.W.astype(np.float64)
        layer.b = layer.b.astype(np.float64)

    # Analytic gradients from the backward pass (loss is already averaged).
    loss = model.loss(x, y)
    model.backward()
    analytic = {}
    for li, layer in enumerate(model.weight_layers):
        for name, grad in (("W", layer.dW), ("b", layer.db)):
            for idx, g in enumerate(grad.reshape(-1)):
                analytic[(li, name, idx)] = float(g)

    numeric = numerical_grads(model, x, y, num=5,
                              rng=np.random.default_rng(seed + 2))

    max_rel = 0.0
    print(f"\n=== {label} ===  loss = {loss:.6f}")
    print(f"{'param':>16} {'analytic':>14} {'numeric':>14} {'rel.err':>10}")
    for key, ng in numeric.items():
        ag = analytic[key]
        denom = max(1e-8, abs(ag) + abs(ng))
        rel = abs(ag - ng) / denom
        max_rel = max(max_rel, rel)
        print(f"{str(key):>16} {ag:>14.8f} {ng:>14.8f} {rel:>10.2e}")

    passed = max_rel < tol
    status = "PASS" if passed else "FAIL"
    print(f"{status}: max relative error {max_rel:.2e} "
          f"{'<' if passed else '>='} {tol:.0e}")
    return passed


def check_mlp():
    rng = np.random.default_rng(1)
    B, D, C = 8, 20, 5
    model = MLP([D, 16, 12, C], seed=0)
    x = rng.normal(size=(B, D)).astype(np.float64)
    y = np.zeros((B, C), dtype=np.float64)
    y[np.arange(B), rng.integers(0, C, B)] = 1.0
    return check_model(model, x, y, "MLP", seed=1)


def check_convnet():
    rng = np.random.default_rng(3)
    B, C = 2, 5
    model = ConvNet(in_shape=(1, 8, 8), num_classes=C, seed=0,
                    channels=(2, 3), fc=6)
    x = rng.normal(size=(B, 1, 8, 8)).astype(np.float64)
    y = np.zeros((B, C), dtype=np.float64)
    y[np.arange(B), rng.integers(0, C, B)] = 1.0
    return check_model(model, x, y, "ConvNet", seed=4)


def main():
    ok_mlp = check_mlp()
    ok_cnn = check_convnet()
    print()
    if ok_mlp and ok_cnn:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
