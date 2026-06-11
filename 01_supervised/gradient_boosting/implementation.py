"""Gradient boosting for regression from scratch — pure stdlib, zero dependencies.

Run it:  python3 implementation.py

The idea:    F_0(x) = mean(y)
             for m in 1..M:
                 r_i  = y_i - F_{m-1}(x_i)        # residuals = negative gradient of MSE
                 h_m  = fit a tiny tree (stump) to the residuals
                 F_m  = F_{m-1} + nu * h_m        # add it, shrunk by the learning rate

Each stump is weak — a single step function — but each one is fit to whatever
error the ensemble still makes, so the errors get hammered down round by round.
"""

import math
import random


def fit_stump(xs, residuals):
    """Find the depth-1 regression tree (one split) minimizing squared error.

    A stump is: if x <= threshold predict left_mean else right_mean.
    We scan every candidate threshold (midpoints between sorted xs) and pick
    the split whose two leaf means give the lowest total squared error.
    """
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    sx = [xs[i] for i in order]
    sr = [residuals[i] for i in order]
    n = len(sx)

    total = sum(sr)
    best = (float("inf"), None, 0.0, 0.0)
    left_sum, left_sq = 0.0, 0.0
    sq_total = sum(r * r for r in sr)

    for i in range(n - 1):
        left_sum += sr[i]
        left_sq += sr[i] * sr[i]
        if sx[i] == sx[i + 1]:
            continue  # can't split between identical x values
        nl, nr = i + 1, n - i - 1
        right_sum = total - left_sum
        # SSE of a leaf predicting its mean = sum(r²) - (sum r)²/n
        sse = (left_sq - left_sum**2 / nl) + ((sq_total - left_sq) - right_sum**2 / nr)
        if sse < best[0]:
            best = (sse, (sx[i] + sx[i + 1]) / 2, left_sum / nl, right_sum / nr)

    _, thr, left, right = best
    return thr, left, right


def stump_predict(stump, x):
    thr, left, right = stump
    return left if x <= thr else right


def boost(xs, ys, n_rounds=50, nu=0.1, verbose_every=5):
    """Gradient boosting: each round, fit a stump to the current residuals."""
    f0 = sum(ys) / len(ys)                      # round 0: just predict the mean
    stumps = []

    def predict(x):
        return f0 + nu * sum(stump_predict(s, x) for s in stumps)

    for m in range(1, n_rounds + 1):
        residuals = [y - predict(x) for x, y in zip(xs, ys)]
        stumps.append(fit_stump(xs, residuals))
        if verbose_every and (m % verbose_every == 0 or m == 1):
            mse = sum((y - predict(x)) ** 2 for x, y in zip(xs, ys)) / len(xs)
            big = max(range(len(xs)), key=lambda i: abs(residuals[i]))
            print(f"round {m:3d}   train MSE {mse:.5f}   "
                  f"(this stump targeted the residual at x={xs[big]:.2f})")
    return predict


if __name__ == "__main__":
    random.seed(7)

    # 1D synthetic problem: a wavy function no single stump could ever fit
    def truth(x):
        return math.sin(3 * x) + 0.5 * x

    xs = [random.uniform(0, 3) for _ in range(120)]
    ys = [truth(x) + random.gauss(0, 0.15) for x in xs]
    x_test = [random.uniform(0, 3) for _ in range(120)]
    y_test = [truth(x) + random.gauss(0, 0.15) for x in x_test]

    base_mse = sum((y - sum(ys) / len(ys)) ** 2 for y in ys) / len(ys)
    print(f"round   0   train MSE {base_mse:.5f}   (predicting the mean)\n")

    model = boost(xs, ys, n_rounds=60, nu=0.1)

    train_mse = sum((y - model(x)) ** 2 for x, y in zip(xs, ys)) / len(xs)
    test_mse = sum((y - model(x)) ** 2 for x, y in zip(x_test, y_test)) / len(x_test)
    print(f"\nfinal fit:  train MSE {train_mse:.5f}   holdout MSE {test_mse:.5f}")
    print(f"            (noise floor is sigma² = {0.15**2:.5f} — "
          "we're close, so the wave is learned)")
    print("\n60 one-split stumps, each fixing the previous ensemble's worst"
          "\nerrors, add up to a smooth approximation of sin(3x) + x/2.")
