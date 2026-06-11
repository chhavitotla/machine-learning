"""K-fold cross-validation from scratch — pure standard library, zero dependencies.

Run it:  python3 implementation.py

One train/test split gives you ONE number, and that number depends on which
rows happened to land in the test set. Cross-validation re-deals the cards K
times: every sample gets exactly one turn in the validation set, you train K
models, and the mean of the K scores is your estimate — with the std telling
you how much to trust it. Below we run both protocols on the same data and
watch the single splits swing while the CV mean stays put.
"""

import math
import random


def make_blobs(n: int = 120) -> list[tuple[float, float, int]]:
    """Two overlapping gaussian clouds in 2D — overlap so accuracy is < 100%
    and the luck-of-the-split effect has room to show itself."""
    pts = []
    for i in range(n):
        y = i % 2
        cx, cy = (0.8, 0.8) if y else (-0.8, -0.8)
        pts.append((random.gauss(cx, 1.0), random.gauss(cy, 1.0), y))
    return pts


def train_logreg(data: list[tuple[float, float, int]], steps: int = 300, lr: float = 0.5):
    """Tiny logistic regression by batch gradient descent.

    The model is sigmoid(w1*x1 + w2*x2 + b). The gradient of log-loss has the
    famously clean form (p - y) * x — predicted minus actual, times the input.
    """
    w1 = w2 = b = 0.0
    n = len(data)
    for _ in range(steps):
        g1 = g2 = gb = 0.0
        for x1, x2, y in data:
            p = 1.0 / (1.0 + math.exp(-(w1 * x1 + w2 * x2 + b)))
            g1 += (p - y) * x1
            g2 += (p - y) * x2
            gb += (p - y)
        w1 -= lr * g1 / n
        w2 -= lr * g2 / n
        b -= lr * gb / n
    return w1, w2, b


def accuracy(model, data) -> float:
    """Fraction of held-out points the trained model gets right."""
    w1, w2, b = model
    hits = sum(1 for x1, x2, y in data if (w1 * x1 + w2 * x2 + b > 0) == (y == 1))
    return hits / len(data)


def kfold_scores(data, k: int) -> list[tuple[float, int, int]]:
    """The whole algorithm: shuffle once, cut into K folds, and for each fold
    train on the OTHER K-1 folds and score on the one held out.
    Returns (score, train size, val size) per fold.

    Every sample is validated exactly once and trained on exactly K-1 times —
    no sample ever scores a model that saw it. The shuffle happens BEFORE the
    cut; shuffling after seeing the data per-fold would be leakage.
    """
    idx = list(range(len(data)))
    random.shuffle(idx)
    folds = [idx[i::k] for i in range(k)]          # striped split: sizes differ by at most 1
    scores = []
    for f in range(k):
        val = [data[i] for i in folds[f]]
        train = [data[i] for g in range(k) if g != f for i in folds[g]]
        scores.append((accuracy(train_logreg(train), val), len(train), len(val)))
    return scores


def mean_std(xs: list[float]) -> tuple[float, float]:
    """Sample mean and sample std (n-1 denominator, since the mean was estimated)."""
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(var)


def single_split_score(data, test_frac: float = 0.25) -> float:
    """The naive protocol: one random split, one number, no error bar."""
    idx = list(range(len(data)))
    random.shuffle(idx)
    cut = int(len(data) * (1 - test_frac))
    train = [data[i] for i in idx[:cut]]
    test = [data[i] for i in idx[cut:]]
    return accuracy(train_logreg(train), test)


if __name__ == "__main__":
    random.seed(7)
    data = make_blobs(120)

    print("Five single random 75/25 splits — same data, same model, different luck:")
    singles = [single_split_score(data) for _ in range(5)]
    print("  " + "  ".join(f"{s:.3f}" for s in singles))
    m_s, sd_s = mean_std(singles)
    print(f"  spread: min {min(singles):.3f}  max {max(singles):.3f}  "
          f"(a {max(singles) - min(singles):.3f} swing from the split alone)")
    print()

    k = 5
    print(f"{k}-fold cross-validation — every sample validated exactly once:")
    results = kfold_scores(data, k)
    for f, (s, n_train, n_val) in enumerate(results):
        print(f"  fold {f + 1}: trained on {n_train} samples, "
              f"validated on the other {n_val} -> accuracy {s:.3f}")
    scores = [s for s, _, _ in results]
    m, sd = mean_std(scores)
    print(f"  CV estimate: {m:.3f} ± {sd:.3f}   "
          f"(standard error of the mean ≈ {sd / math.sqrt(k):.3f})")
    print()
    print("The single-split numbers above swing by several points of accuracy on")
    print("identical data — report any one of them and you're reporting luck.")
    print("The CV mean averages that luck out, and the ± tells you how big the")
    print("luck was. Same compute budget as a few ad-hoc splits, but now you get")
    print("an error bar — and an error bar is the difference between 'model A")
    print("beats model B' and 'model A beat model B on one shuffle'.")
