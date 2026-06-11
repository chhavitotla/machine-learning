"""Linear soft-margin SVM from scratch — pure standard library, zero dependencies.

Run it:  python3 implementation.py

The model:   f(x) = w·x + b,  predict sign(f(x))
The loss:    (lambda/2)·||w||² + (1/n) Σ max(0, 1 - y·f(x))   (hinge + L2)
The update:  Pegasos-style subgradient descent with step size 1/(lambda·t)

The hinge loss is zero only when a point is on the CORRECT side of its
margin line (y·f(x) >= 1). Points inside or on the margin push back on w —
those are the support vectors; every other point contributes nothing.
"""

import random


def hinge_objective(w, b, X, y, lam):
    """(lambda/2)||w||² + mean hinge loss — the thing Pegasos minimizes."""
    n = len(X)
    hinge = sum(max(0.0, 1.0 - yi * (w[0] * xi[0] + w[1] * xi[1] + b))
                for xi, yi in zip(X, y)) / n
    return 0.5 * lam * (w[0] ** 2 + w[1] ** 2) + hinge


def fit_svm(X, y, lam=0.01, epochs=200, seed=0):
    """Pegasos: subgradient descent on the regularized hinge loss.

    Each step, for every point with margin violation (y·f(x) < 1) the
    subgradient of the hinge term is -y·x; the L2 term always contributes
    lam·w (which shrinks w, i.e. WIDENS the margin 2/||w||).
    """
    rng = random.Random(seed)
    w, b = [0.0, 0.0], 0.0
    t = 0
    idx = list(range(len(X)))
    for _ in range(epochs):
        rng.shuffle(idx)
        for i in idx:
            t += 1
            eta = 1.0 / (lam * t)          # decaying step size — Pegasos schedule
            xi, yi = X[i], y[i]
            margin = yi * (w[0] * xi[0] + w[1] * xi[1] + b)
            # L2 shrink happens every step; hinge pull only on violators
            w[0] -= eta * lam * w[0]
            w[1] -= eta * lam * w[1]
            if margin < 1.0:
                w[0] += eta * yi * xi[0]
                w[1] += eta * yi * xi[1]
                b += eta * yi
    return w, b


if __name__ == "__main__":
    random.seed(42)

    # Two gaussian blobs, nearly separable (a little overlap so the soft
    # margin actually has work to do)
    X, y = [], []
    for _ in range(60):
        X.append([random.gauss(2.0, 0.8), random.gauss(2.0, 0.8)]); y.append(+1)
        X.append([random.gauss(-2.0, 0.8), random.gauss(-2.0, 0.8)]); y.append(-1)

    lam = 0.01
    w, b = fit_svm(X, y, lam=lam)

    correct = sum(1 for xi, yi in zip(X, y)
                  if (1 if w[0] * xi[0] + w[1] * xi[1] + b >= 0 else -1) == yi)
    norm_w = (w[0] ** 2 + w[1] ** 2) ** 0.5

    print(f"learned  w = [{w[0]:+.4f}, {w[1]:+.4f}]   b = {b:+.4f}")
    print(f"accuracy      {correct}/{len(X)} = {correct / len(X):.1%}")
    print(f"||w||         {norm_w:.4f}")
    print(f"margin width  2/||w|| = {2 / norm_w:.4f}")
    print(f"objective     {hinge_objective(w, b, X, y, lam):.4f}")

    # Support vectors: points with y·f(x) <= 1 (on or inside the margin).
    # Only these points determine the boundary — delete the rest and
    # retraining gives the same w.
    sv = [i for i, (xi, yi) in enumerate(zip(X, y))
          if yi * (w[0] * xi[0] + w[1] * xi[1] + b) <= 1.0 + 1e-6]
    print(f"\nsupport vectors ({len(sv)} of {len(X)} points define the boundary):")
    for i in sv:
        m = y[i] * (w[0] * X[i][0] + w[1] * X[i][1] + b)
        print(f"  point {i:3d}  x = [{X[i][0]:+.2f}, {X[i][1]:+.2f}]  "
              f"y = {y[i]:+d}  margin = {m:+.3f}")
