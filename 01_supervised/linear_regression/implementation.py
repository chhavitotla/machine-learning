"""Linear regression from scratch — pure standard library, zero dependencies.

Run it:  python3 implementation.py

The model:   y_hat = w*x + b
The loss:    MSE = (1/n) * sum((y - y_hat)^2)
The update:  w -= alpha * dL/dw,  b -= alpha * dL/db

This is the same gradient-descent loop that trains every neural network;
linear regression is just the smallest model you can attach it to.
"""

import random


def predict(w: float, b: float, x: float) -> float:
    """The entire model: a line."""
    return w * x + b


def mse_loss(w: float, b: float, xs: list[float], ys: list[float]) -> float:
    """Mean squared error: average squared vertical distance to the line.

    Squared (not absolute) so that (a) it's differentiable everywhere and
    (b) big misses are punished disproportionately — one 10-unit error
    costs as much as a hundred 1-unit errors.
    """
    n = len(xs)
    return sum((y - predict(w, b, x)) ** 2 for x, y in zip(xs, ys)) / n


def gradients(w: float, b: float, xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Hand-derived gradients of MSE.

    dL/dw = -(2/n) * sum(x * (y - y_hat))   # each point votes to tilt the
                                            # line, with leverage proportional
                                            # to its x
    dL/db = -(2/n) * sum(y - y_hat)         # the average residual shifts the
                                            # line up or down
    """
    n = len(xs)
    dw = db = 0.0
    for x, y in zip(xs, ys):
        residual = y - predict(w, b, x)
        dw += -2.0 * x * residual / n
        db += -2.0 * residual / n
    return dw, db


def fit(
    xs: list[float],
    ys: list[float],
    alpha: float = 0.05,
    epochs: int = 2000,
    verbose: bool = True,
) -> tuple[float, float]:
    """Gradient descent: start anywhere, repeatedly step downhill."""
    w, b = 0.0, 0.0
    for epoch in range(epochs):
        dw, db = gradients(w, b, xs, ys)
        w -= alpha * dw
        b -= alpha * db
        if verbose and epoch % (epochs // 10) == 0:
            print(f"epoch {epoch:5d}   loss {mse_loss(w, b, xs, ys):.6f}   "
                  f"w {w:+.4f}   b {b:+.4f}")
    return w, b


def fit_closed_form(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """The normal equation, specialized to one feature.

    For simple regression the matrix algebra (X^T X)^-1 X^T y collapses to:
        w = cov(x, y) / var(x)
        b = mean(y) - w * mean(x)

    One shot, no iterations — but the general version costs O(d^3) in the
    number of features, which is why gradient descent wins at scale.
    """
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    w = cov_xy / var_x
    b = mean_y - w * mean_x
    return w, b


if __name__ == "__main__":
    random.seed(42)

    # Synthetic data: y = 2.5x + 1.0 + gaussian noise
    TRUE_W, TRUE_B = 2.5, 1.0
    xs = [random.uniform(0, 5) for _ in range(100)]
    ys = [TRUE_W * x + TRUE_B + random.gauss(0, 0.5) for x in xs]

    print(f"ground truth:          w {TRUE_W:+.4f}   b {TRUE_B:+.4f}\n")

    print("— gradient descent —")
    w_gd, b_gd = fit(xs, ys)
    print(f"\ngradient descent fit:  w {w_gd:+.4f}   b {b_gd:+.4f}")

    w_cf, b_cf = fit_closed_form(xs, ys)
    print(f"closed-form fit:       w {w_cf:+.4f}   b {b_cf:+.4f}")
    print("\nBoth recover the truth up to noise — gradient descent just gets"
          "\nthere by walking instead of jumping.")
