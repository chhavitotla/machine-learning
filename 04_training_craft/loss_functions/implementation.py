"""MSE, MAE, Huber, and binary cross-entropy from scratch — pure stdlib.

Run it:  python3 implementation.py

Each loss is a few lines: the value AND its gradient w.r.t. the
prediction, because the gradient is what actually trains the model.
The demo fits the same line twice — once minimizing MSE, once MAE —
on data with one corrupted point, so you can see the squared penalty
hand the outlier the steering wheel.
"""

import math


# ---------------------------------------------------------------------------
# The losses. Each returns (loss, dloss/dprediction) for ONE example.
# Averaging over a batch (and summing the gradients) is the caller's job.
# ---------------------------------------------------------------------------

def mse(y: float, pred: float) -> tuple[float, float]:
    """Squared error. Gradient 2e grows with the error — outliers shout."""
    e = pred - y
    return e * e, 2 * e


def mae(y: float, pred: float) -> tuple[float, float]:
    """Absolute error. Gradient is sign(e): every point gets one equal
    vote, however far away it is — outliers whisper. The flip side:
    the push never shrinks near the target, so convergence jitters."""
    e = pred - y
    return abs(e), (1.0 if e > 0 else -1.0 if e < 0 else 0.0)


def huber(y: float, pred: float, delta: float = 1.0) -> tuple[float, float]:
    """Quadratic inside |e| <= delta, linear outside. The pieces meet
    with matching value and slope, so it's smooth — and the gradient is
    literally clip(e, -delta, +delta): MSE with gradient clipping."""
    e = pred - y
    if abs(e) <= delta:
        return 0.5 * e * e, e
    return delta * (abs(e) - 0.5 * delta), delta * (1.0 if e > 0 else -1.0)


def bce(y: float, p: float, eps: float = 1e-12) -> tuple[float, float]:
    """Binary cross-entropy on a probability p for label y in {0, 1}.

    Cost = -log(probability assigned to the truth): confident wrongness
    is punished exponentially harder than honest uncertainty. The clamp
    matters — at p = 0 or 1 the log is infinite, which is why real
    frameworks fuse sigmoid+BCE into one stable op.
    """
    p = min(max(p, eps), 1 - eps)
    loss = -(y * math.log(p) + (1 - y) * math.log(1 - p))
    grad = (p - y) / (p * (1 - p))      # w.r.t. p; w.r.t. the logit it's just p - y
    return loss, grad


# ---------------------------------------------------------------------------
# The arena: fit y = w*x + b by gradient descent under a chosen loss.
# ---------------------------------------------------------------------------

def fit_line(data, loss_fn, lr=0.01, steps=8000):
    """Plain gradient descent. Chain rule: dL/dw = dL/dpred * x, dL/db = dL/dpred."""
    w, b = 0.0, 0.0
    n = len(data)
    for _ in range(steps):
        gw = gb = 0.0
        for x, y in data:
            _, g = loss_fn(y, w * x + b)
            gw += g * x / n
            gb += g / n
        w -= lr * gw
        b -= lr * gb
    return w, b


def avg_loss(data, loss_fn, w, b):
    return sum(loss_fn(y, w * x + b)[0] for x, y in data) / len(data)


if __name__ == "__main__":
    # nine clean points on y = 0.6x + 1, plus one corrupted label
    clean = [(x, 0.6 * x + 1 + j) for x, j in
             zip(range(1, 10), [0.3, -0.25, 0.18, -0.1, 0.35, -0.3, 0.12, -0.2, 0.22])]
    data = clean + [(8.0, 14.0)]        # true value ~5.8, recorded as 14

    print("Fitting y = w*x + b on 9 clean points + 1 outlier (true w=0.6, b=1)\n")
    print(f"{'loss':<8} {'w':>8} {'b':>8}   verdict")
    for name, fn in [("MSE", mse), ("MAE", mae), ("Huber", huber)]:
        w, b = fit_line(data, fn)
        pulled = abs(w - 0.6) > 0.1
        verdict = "dragged toward the outlier" if pulled else "stayed on the trend"
        print(f"{name:<8} {w:>8.3f} {b:>8.3f}   {verdict}")

    # same losses, same errors, very different prices
    print("\nPrice list — one prediction error, four bills (target y = 1, delta = 1):\n")
    print(f"{'error':>6} {'MSE':>10} {'MAE':>8} {'Huber':>8}     gradient: MSE vs MAE")
    for e in [0.1, 0.5, 1.0, 3.0, 10.0]:
        lm, gm = mse(1.0, 1.0 + e)
        la, ga = mae(1.0, 1.0 + e)
        lh, _ = huber(1.0, 1.0 + e)
        print(f"{e:>6.1f} {lm:>10.2f} {la:>8.2f} {lh:>8.2f}     {gm:>6.1f} vs {ga:.1f}")

    # cross-entropy prices CONFIDENCE, not distance
    print("\nBCE for a true label of 1 — the cost of (mis)placed confidence:\n")
    for p in [0.99, 0.9, 0.5, 0.1, 0.01]:
        loss, _ = bce(1.0, p)
        print(f"  predicted p = {p:<5}  loss = {loss:.3f}"
              + ("   <- confidently wrong: ~460x the confident-right bill" if p == 0.01 else ""))

    print(
        "\nSame data, same optimizer — only the loss changed, and only the"
        "\nMSE line chased the outlier. Squaring made one bad label worth"
        "\nmore than the nine good ones; MAE gave it a single equal vote,"
        "\nand Huber capped its gradient at delta. The loss IS the spec."
    )
