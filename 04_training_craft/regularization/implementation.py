"""Regularization from scratch — ridge (L2) and lasso (L1), pure stdlib.

Run it:  python3 implementation.py

Ridge:  minimize  ||y - Xw||² + λ||w||²   → closed form: (XᵀX + λI) w = Xᵀy
Lasso:  minimize  ||y - Xw||² + λ||w||₁   → no closed form; coordinate descent

The demo fits both on the same data at three λ values. Watch L2 shrink
every coefficient smoothly toward zero while L1 snaps the useless ones
to EXACTLY zero — that's the sparsity story in one table.
"""

import random


def ridge_fit(X: list[list[float]], y: list[float], lam: float) -> list[float]:
    """Ridge regression via the normal equations: solve (XᵀX + λI) w = Xᵀy.

    The λI on the diagonal does two jobs: it penalizes large weights, and it
    makes the matrix invertible even when features are perfectly correlated.
    """
    n, d = len(X), len(X[0])
    # A = XᵀX + λI  (d×d),  b = Xᵀy  (d)
    A = [[sum(X[k][i] * X[k][j] for k in range(n)) + (lam if i == j else 0.0)
          for j in range(d)] for i in range(d)]
    b = [sum(X[k][i] * y[k] for k in range(n)) for i in range(d)]
    return gauss_solve(A, b)


def gauss_solve(A: list[list[float]], b: list[float]) -> list[float]:
    """Solve A w = b by Gaussian elimination with partial pivoting."""
    d = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]   # augmented matrix
    for col in range(d):
        # pivot: swap up the row with the largest entry in this column
        piv = max(range(col, d), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        # eliminate everything below the pivot
        for r in range(col + 1, d):
            f = M[r][col] / M[col][col]
            for c in range(col, d + 1):
                M[r][c] -= f * M[col][c]
    # back-substitution
    w = [0.0] * d
    for r in range(d - 1, -1, -1):
        w[r] = (M[r][d] - sum(M[r][c] * w[c] for c in range(r + 1, d))) / M[r][r]
    return w


def lasso_fit(X: list[list[float]], y: list[float], lam: float,
              iters: int = 200) -> list[float]:
    """Lasso via coordinate descent (features assumed standardized).

    Cycle through coordinates; each 1-D subproblem has an exact solution:
    the soft-threshold. rho_j is what feature j 'wants' its weight to be
    (its correlation with the residual); soft-thresholding subtracts λ
    from |rho_j| — and if the feature can't even pay the λ tax, its
    weight is EXACTLY zero. That's where sparsity comes from.
    """
    n, d = len(X), len(X[0])
    w = [0.0] * d
    z = [sum(X[k][j] ** 2 for k in range(n)) for j in range(d)]  # Σ x_j²
    for _ in range(iters):
        for j in range(d):
            # residual ignoring feature j's current contribution
            rho = sum(X[k][j] * (y[k] - sum(X[k][i] * w[i] for i in range(d) if i != j))
                      for k in range(n))
            # soft-threshold: shrink toward 0, clip to exactly 0 inside [-λ, λ]
            if rho < -lam:
                w[j] = (rho + lam) / z[j]
            elif rho > lam:
                w[j] = (rho - lam) / z[j]
            else:
                w[j] = 0.0
    return w


def standardize(X: list[list[float]]) -> list[list[float]]:
    """Zero-mean, unit-variance columns — required so one λ is fair to all."""
    n, d = len(X), len(X[0])
    mu = [sum(row[j] for row in X) / n for j in range(d)]
    sd = [max(1e-12, (sum((row[j] - mu[j]) ** 2 for row in X) / n) ** 0.5)
          for j in range(d)]
    return [[(row[j] - mu[j]) / sd[j] for j in range(d)] for row in X]


if __name__ == "__main__":
    random.seed(0)
    # 8 features: only the first 3 matter; the rest are pure noise.
    TRUE_W = [3.0, -2.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0]
    n, d = 60, 8
    X = [[random.gauss(0, 1) for _ in range(d)] for _ in range(n)]
    X = standardize(X)
    y = [sum(wj * xj for wj, xj in zip(TRUE_W, row)) + random.gauss(0, 0.5)
         for row in X]
    y = [yi - sum(y) / n for yi in y]   # center y so we can skip the intercept

    print("true weights:  " + "  ".join(f"{w:+5.2f}" for w in TRUE_W))
    print("(features 4-8 are pure noise — watch what each penalty does to them)\n")

    for lam in (0.0, 5.0, 50.0):
        wr = ridge_fit(X, y, lam)
        wl = lasso_fit(X, y, lam)
        nz = sum(1 for w in wl if w == 0.0)
        print(f"λ = {lam:g}")
        print("  ridge (L2):  " + "  ".join(f"{w:+5.2f}" for w in wr))
        print(f"  lasso (L1):  " + "  ".join(f"{w:+5.2f}" for w in wl)
              + f"   ← {nz} weights exactly 0")
        print()

    print("L2 shrinks every weight smoothly but never to zero;")
    print("L1's soft-threshold zeroes the noise features outright — built-in")
    print("feature selection, which is why lasso models are easier to read.")
