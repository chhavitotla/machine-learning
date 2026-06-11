"""Confusion matrix + precision/recall/F1/accuracy from scratch — pure stdlib.

Run it:  python3 implementation.py

A classifier outputs a *score*; a threshold turns it into a *decision*.
The confusion matrix counts the four ways a decision can go:

                      predicted +        predicted -
    actually +        TP (caught)        FN (missed)
    actually -        FP (false alarm)   TN (correct pass)

Every metric below is just an honest ratio over those four counts —
and moving the threshold trades FPs for FNs. That trade IS evaluation.
"""

import random


def confusion_matrix(y_true: list[int], scores: list[float], threshold: float) -> dict[str, int]:
    """Count TP / FP / TN / FN at a given decision threshold.

    score >= threshold  →  predict positive.  That's the entire 'model'
    layer here; the matrix is just bookkeeping on top of it.
    """
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for y, s in zip(y_true, scores):
        pred = 1 if s >= threshold else 0
        if pred == 1 and y == 1:
            counts["tp"] += 1
        elif pred == 1 and y == 0:
            counts["fp"] += 1
        elif pred == 0 and y == 0:
            counts["tn"] += 1
        else:
            counts["fn"] += 1
    return counts


def precision(m: dict[str, int]) -> float:
    """Of everything we flagged positive, how much was real?  TP/(TP+FP)."""
    denom = m["tp"] + m["fp"]
    return m["tp"] / denom if denom else 0.0


def recall(m: dict[str, int]) -> float:
    """Of all real positives, how many did we catch?  TP/(TP+FN)."""
    denom = m["tp"] + m["fn"]
    return m["tp"] / denom if denom else 0.0


def f1(m: dict[str, int]) -> float:
    """Harmonic mean of precision and recall — punishes whichever is worse."""
    p, r = precision(m), recall(m)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def accuracy(m: dict[str, int]) -> float:
    """Fraction of all decisions that were correct. Misleading under imbalance."""
    total = sum(m.values())
    return (m["tp"] + m["tn"]) / total if total else 0.0


def print_report(m: dict[str, int], threshold: float) -> None:
    print(f"threshold = {threshold:.2f}")
    print(f"                 predicted +   predicted -")
    print(f"    actually +   TP = {m['tp']:4d}     FN = {m['fn']:4d}")
    print(f"    actually -   FP = {m['fp']:4d}     TN = {m['tn']:4d}")
    print(f"    precision = {m['tp']}/({m['tp']}+{m['fp']}) = {precision(m):.3f}   "
          f"recall = {m['tp']}/({m['tp']}+{m['fn']}) = {recall(m):.3f}")
    print(f"    F1 = {f1(m):.3f}   accuracy = {accuracy(m):.3f}\n")


if __name__ == "__main__":
    random.seed(42)

    # Synthetic scores: negatives cluster near 0.35, positives near 0.65 —
    # overlapping gaussians, like any real classifier's output.
    y_true = [0] * 400 + [1] * 100          # 4:1 imbalance, like fraud or spam
    scores = [min(1, max(0, random.gauss(0.35, 0.12))) for _ in range(400)] + \
             [min(1, max(0, random.gauss(0.65, 0.12))) for _ in range(100)]

    print("Sweeping the threshold over the SAME scores — watch precision and")
    print("recall trade places, while accuracy barely blinks:\n")
    for t in (0.30, 0.50, 0.70):
        print_report(confusion_matrix(y_true, scores, t), t)

    print("Low threshold → catch everything (high recall), flag junk (low precision).")
    print("High threshold → only flag sure things (high precision), miss real ones")
    print("(low recall). Accuracy stays high throughout because 80% of the data is")
    print("negative — which is exactly why accuracy alone lies under imbalance.")
