"""ROC curve + AUC from scratch — pure standard library, zero dependencies.

Run it:  python3 implementation.py

The ROC curve asks: as you slide the decision threshold from strict to
lenient, how does the true-positive rate (recall) grow versus the
false-positive rate? AUC is the area under that curve — and, beautifully,
it equals the probability that a randomly chosen positive outranks a
randomly chosen negative. We verify that identity by brute force below.
"""

import random


def roc_curve(y_true: list[int], scores: list[float]) -> list[tuple[float, float]]:
    """Sweep the threshold from +inf down through every distinct score.

    Sorting by score descending, each sample we 'admit' as positive moves
    the curve: a real positive steps UP (TPR += 1/P), a negative steps
    RIGHT (FPR += 1/N). No re-counting per threshold — one sort, one pass.
    """
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    points = [(0.0, 0.0)]
    tp = fp = 0
    for i in order:
        if y_true[i] == 1:
            tp += 1
        else:
            fp += 1
        points.append((fp / n_neg, tp / n_pos))
    return points          # ends at (1, 1) by construction


def auc(points: list[tuple[float, float]]) -> float:
    """Trapezoid rule over the ROC points: sum of strip areas."""
    area = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        area += (x1 - x0) * (y0 + y1) / 2.0
    return area


def auc_by_counting(y_true: list[int], scores: list[float]) -> float:
    """The Mann–Whitney identity, computed the dumb honest way.

    AUC = P(score of random positive > score of random negative),
    with ties counting half. Compare every (positive, negative) pair
    directly — O(P*N), fine for a demo, and zero cleverness to trust.
    """
    pos = [s for y, s in zip(y_true, scores) if y == 1]
    neg = [s for y, s in zip(y_true, scores) if y == 0]
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def make_scores(separation: float, n_pos: int = 100, n_neg: int = 300) -> tuple[list[int], list[float]]:
    """Two gaussian score clouds; `separation` is the gap between class means."""
    y = [0] * n_neg + [1] * n_pos
    s = [random.gauss(0.5 - separation / 2, 0.12) for _ in range(n_neg)] + \
        [random.gauss(0.5 + separation / 2, 0.12) for _ in range(n_pos)]
    return y, s


if __name__ == "__main__":
    random.seed(42)

    for name, sep in (("well-separated", 0.5), ("overlapping", 0.1), ("pure noise", 0.0)):
        y, s = make_scores(sep)
        a_trap = auc(roc_curve(y, s))
        a_rank = auc_by_counting(y, s)
        print(f"{name:15s}  (mean gap {sep:.1f})   "
              f"AUC by trapezoid = {a_trap:.4f}   "
              f"AUC by pair-counting = {a_rank:.4f}")
        assert abs(a_trap - a_rank) < 1e-9, "the two AUCs must agree exactly"

    print()
    print("The trapezoid area and the pair-counting probability agree to machine")
    print("precision: AUC IS the probability that a random positive outranks a")
    print("random negative. Perfect ranking -> 1.0, coin-flip scores -> ~0.5 —")
    print("and note AUC never asked what the threshold was, or how rare the")
    print("positives were. That's both its power and its blind spot.")
