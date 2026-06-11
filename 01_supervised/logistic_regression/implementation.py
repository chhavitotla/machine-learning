"""Logistic regression from scratch — pure standard library, zero dependencies.

Run it:  python3 implementation.py

The model:   p = sigmoid(w1*x1 + w2*x2 + b)     # probability of class 1
The loss:    cross-entropy = -(1/n) * sum(y*log(p) + (1-y)*log(1-p))
The update:  w -= alpha * dL/dw,  b -= alpha * dL/db

The gradient derivation is the famous "clean" one:

    dL/dz = p - y                # derivative of cross-entropy through sigmoid
    dL/dw = (p - y) * x          # chain rule: dz/dw = x
    dL/db = (p - y)              # chain rule: dz/db = 1

All the messy terms — the 1/p from the log, the p(1-p) from the sigmoid —
cancel out, leaving just (prediction - truth) * input. This same form
reappears as the output-layer gradient of every softmax classifier.
"""

import math
import random


def sigmoid(z: float) -> float:
    """Squash any real number into (0, 1) — a score becomes a probability.

    Clamped to avoid math.exp overflow for very confident (wrong) scores.
    """
    if z < -35:
        return 1e-15
    if z > 35:
        return 1 - 1e-15
    return 1.0 / (1.0 + math.exp(-z))


def predict_proba(w: list[float], b: float, x: list[float]) -> float:
    """P(y=1 | x): a linear score pushed through the sigmoid."""
    z = sum(wi * xi for wi, xi in zip(w, x)) + b
    return sigmoid(z)


def cross_entropy(w: list[float], b: float, xs: list[list[float]], ys: list[int]) -> float:
    """Average negative log-likelihood: punishes confident wrong answers hard.

    Predicting p=0.01 when y=1 costs -log(0.01) ≈ 4.6; predicting p=0.6
    costs only 0.51. The loss is infinite at p=0 — total confidence in the
    wrong class is unforgivable.
    """
    n = len(xs)
    total = 0.0
    for x, y in zip(xs, ys):
        p = predict_proba(w, b, x)
        total += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return total / n


def fit(xs: list[list[float]], ys: list[int],
        alpha: float = 0.5, epochs: int = 500) -> tuple[list[float], float]:
    """Gradient descent on cross-entropy. The gradient is just (p - y) * x."""
    w = [0.0, 0.0]
    b = 0.0
    n = len(xs)
    for epoch in range(epochs + 1):
        dw = [0.0, 0.0]
        db = 0.0
        for x, y in zip(xs, ys):
            err = predict_proba(w, b, x) - y     # (p - y): the whole gradient story
            dw[0] += err * x[0] / n
            dw[1] += err * x[1] / n
            db += err / n
        if epoch % (epochs // 10) == 0:
            print(f"epoch {epoch:4d}   loss {cross_entropy(w, b, xs, ys):.4f}   "
                  f"w [{w[0]:+.3f}, {w[1]:+.3f}]   b {b:+.3f}")
        w[0] -= alpha * dw[0]
        w[1] -= alpha * dw[1]
        b -= alpha * db
    return w, b


def accuracy(w: list[float], b: float, xs: list[list[float]], ys: list[int]) -> float:
    """Fraction of points where thresholding p at 0.5 gets the class right."""
    hits = sum(1 for x, y in zip(xs, ys) if (predict_proba(w, b, x) >= 0.5) == bool(y))
    return hits / len(xs)


if __name__ == "__main__":
    random.seed(42)

    # Two gaussian blobs in 2D: class 0 centered at (-1.5, -1), class 1 at (+1.5, +1).
    # They overlap a little, so 100% accuracy is impossible — exactly the regime
    # where probabilities (not just labels) earn their keep.
    xs, ys = [], []
    for _ in range(100):
        xs.append([random.gauss(-1.5, 1.0), random.gauss(-1.0, 1.0)])
        ys.append(0)
        xs.append([random.gauss(1.5, 1.0), random.gauss(1.0, 1.0)])
        ys.append(1)

    print("— gradient descent on cross-entropy —")
    w, b = fit(xs, ys)

    print(f"\nfinal accuracy: {accuracy(w, b, xs, ys):.1%} on {len(xs)} points")
    print(f"decision boundary: {w[0]:.3f}*x1 + {w[1]:.3f}*x2 + {b:.3f} = 0")
    print("\nLoss falls smoothly because cross-entropy + sigmoid is convex —")
    print("one global minimum, no luck required.")
