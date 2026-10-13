"""Verify backpropagation against numerical gradients.

The best way to trust your backward pass is to compare it with a brute
force numerical derivative. By definition,

    dL/dw  ~=  (L(w + eps) - L(w - eps)) / (2 * eps)

for a tiny eps. If our analytic gradients from backprop match these
"central difference" estimates, the chain rule implementation is correct.

We check only a handful of weights per layer because each numerical
estimate costs two extra forward passes.
"""

import numpy as np

from network import MLP


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


def main():
    rng = np.random.default_rng(1)
    B, D, C = 8, 20, 5
    model = MLP([D, 16, 12, C], seed=0)
    # Run the check in float64. Central differences divide a tiny numerator
    # by 2*eps, so float32 round-off would swamp the signal and produce
    # spurious ~1e-2 relative errors even when backprop is exactly right.
    for layer in model.weight_layers:
        layer.W = layer.W.astype(np.float64)
        layer.b = layer.b.astype(np.float64)

    x = rng.normal(size=(B, D)).astype(np.float64)
    y = np.zeros((B, C), dtype=np.float64)
    y[np.arange(B), rng.integers(0, C, B)] = 1.0

    # Analytic gradients from the backward pass (loss is already averaged).
    loss = model.loss(x, y)
    model.backward()
    analytic = {}
    for li, layer in enumerate(model.weight_layers):
        for name, grad, param in (("W", layer.dW, layer.W), ("b", layer.db, layer.b)):
            for idx, g in enumerate(grad.reshape(-1)):
                analytic[(li, name, idx)] = float(g)

    numeric = numerical_grads(model, x, y, num=5, rng=np.random.default_rng(2))

    max_rel = 0.0
    print(f"loss = {loss:.6f}")
    print(f"{'param':>12} {'analytic':>14} {'numeric':>14} {'rel.err':>10}")
    for key, ng in numeric.items():
        ag = analytic[key]
        denom = max(1e-8, abs(ag) + abs(ng))
        rel = abs(ag - ng) / denom
        max_rel = max(max_rel, rel)
        print(f"{str(key):>12} {ag:>14.8f} {ng:>14.8f} {rel:>10.2e}")

    tol = 1e-4
    if max_rel < tol:
        print(f"\nPASS: max relative error {max_rel:.2e} < {tol:.0e}")
    else:
        print(f"\nFAIL: max relative error {max_rel:.2e} >= {tol:.0e}")


if __name__ == "__main__":
    main()
