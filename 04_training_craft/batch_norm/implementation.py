"""Batch normalization from scratch — pure stdlib, zero dependencies.

Run it:  python3 implementation.py

BN, per feature, over the batch:
    train:     x_hat = (x - batch_mean) / sqrt(batch_var + eps)
               y     = gamma * x_hat + beta
               (and update running_mean / running_var for later)
    inference: x_hat = (x - running_mean) / sqrt(running_var + eps)

Demo 1 shows WHY: in a deep sigmoid net the per-layer pre-activation
spread collapses geometrically; BN pins it at ~1 every layer.
Demo 2 shows the classic BUG: using batch statistics at inference.
"""

import math
import random


class BatchNorm:
    """Batch norm for one layer of `dim` features."""

    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1):
        self.gamma = [1.0] * dim          # learnable scale (left at init here)
        self.beta = [0.0] * dim           # learnable shift
        self.running_mean = [0.0] * dim   # exponential average of batch means
        self.running_var = [1.0] * dim    # ... and of batch variances
        self.eps, self.momentum = eps, momentum

    def __call__(self, X: list[list[float]], training: bool) -> list[list[float]]:
        n, d = len(X), len(X[0])
        if training:
            # statistics of THIS batch — every example's output depends on
            # who else is in the batch (the heart of both BN's power and its bug)
            mean = [sum(row[j] for row in X) / n for j in range(d)]
            var = [sum((row[j] - mean[j]) ** 2 for row in X) / n for j in range(d)]
            for j in range(d):  # remember stats for inference time
                self.running_mean[j] += self.momentum * (mean[j] - self.running_mean[j])
                self.running_var[j] += self.momentum * (var[j] - self.running_var[j])
        else:
            # inference: frozen running statistics — output depends only on x
            mean, var = self.running_mean, self.running_var
        return [[self.gamma[j] * (row[j] - mean[j]) / math.sqrt(var[j] + self.eps)
                 + self.beta[j] for j in range(d)] for row in X]


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, z))))


def matmul(X: list[list[float]], W: list[list[float]]) -> list[list[float]]:
    """(n×d) @ (d×k) with plain lists."""
    return [[sum(x * w for x, w in zip(row, col)) for col in zip(*W)] for row in X]


def batch_std(Z: list[list[float]]) -> float:
    """Mean over units of the per-unit std across the batch."""
    n, d = len(Z), len(Z[0])
    total = 0.0
    for j in range(d):
        m = sum(row[j] for row in Z) / n
        total += math.sqrt(sum((row[j] - m) ** 2 for row in Z) / n)
    return total / d


if __name__ == "__main__":
    random.seed(1)
    LAYERS, DIM, BATCH = 6, 32, 64
    # Xavier-ish init: without BN the sigmoid's max slope of 1/4 still
    # shrinks the signal every layer — watch the std collapse below.
    Ws = [[[random.gauss(0, 1 / math.sqrt(DIM)) for _ in range(DIM)]
           for _ in range(DIM)] for _ in range(LAYERS)]
    X0 = [[random.gauss(0, 1) for _ in range(DIM)] for _ in range(BATCH)]

    print("Demo 1 — pre-activation std per layer, 6-layer sigmoid MLP")
    print(f"{'layer':>5} {'no BN':>10} {'with BN':>10}")
    h_plain, h_bn = X0, X0
    bns = [BatchNorm(DIM) for _ in range(LAYERS)]
    for layer in range(LAYERS):
        z_plain = matmul(h_plain, Ws[layer])
        z_bn = bns[layer](matmul(h_bn, Ws[layer]), training=True)
        print(f"{layer + 1:>5} {batch_std(z_plain):>10.4f} {batch_std(z_bn):>10.4f}")
        h_plain = [[sigmoid(z) for z in row] for row in z_plain]
        h_bn = [[sigmoid(z) for z in row] for row in z_bn]
    print("→ without BN the spread dies geometrically (sigmoid slope ≤ 1/4 per")
    print("  layer): later layers see near-constant inputs and learn nothing.")
    print("  With BN every layer is re-standardized to std ≈ 1.\n")

    print("Demo 2 — the train-vs-inference bug")
    bn = BatchNorm(2)
    # "train" long enough that running stats converge to the data's stats
    for _ in range(200):
        batch = [[random.gauss(5, 2), random.gauss(-3, 0.5)] for _ in range(BATCH)]
        bn(batch, training=True)

    a, b = [9.0, -3.0], [1.0, -3.0]   # two very different users at serving time
    # WRONG: training=True with batch size 1 → mean = the input itself,
    # var = 0 → x_hat = 0 → output = beta, IDENTICAL for every input.
    wrong_a, wrong_b = bn([a], training=True)[0], bn([b], training=True)[0]
    # RIGHT: frozen running statistics → inputs stay distinguishable.
    right_a, right_b = bn([a], training=False)[0], bn([b], training=False)[0]
    print(f"  input a={a}, b={b}")
    print(f"  batch stats (bug):  a → {[round(v, 3) for v in wrong_a]}   "
          f"b → {[round(v, 3) for v in wrong_b]}   ← indistinguishable!")
    print(f"  running stats:      a → {[round(v, 3) for v in right_a]}   "
          f"b → {[round(v, 3) for v in right_b]}")
    print("→ at inference, always normalize with running statistics: the model")
    print("  must be a deterministic function of one input, not of its batch.")
