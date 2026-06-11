"""Linear algebra from scratch — pure standard library, zero dependencies.

Run it:  python3 implementation.py

The big ideas:
    matvec:  a matrix is a function that moves vectors
    matmul:  composing two of those functions
    det:     how much the function scales area (negative = it flips space)
    inverse: the function that undoes the function
    power iteration: repeatedly applying M reveals its favorite direction
                     (the dominant eigenvector)
"""

import math


def dot(u: list[float], v: list[float]) -> float:
    """Sum of coordinate-wise products. Geometrically: |u||v|cos(angle) —
    how much the two vectors agree in direction."""
    return sum(a * b for a, b in zip(u, v))


def matvec(M: list[list[float]], v: list[float]) -> list[float]:
    """Apply the matrix to a vector. Each output coordinate is the dot
    product of one row with v — the new position of v after M warps space."""
    return [dot(row, v) for row in M]


def matmul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """Compose two transformations: (A @ B) v means 'apply B, then A'.
    Entry (i, j) = row i of A · column j of B."""
    cols_B = transpose(B)
    return [[dot(row, col) for col in cols_B] for row in A]


def transpose(M: list[list[float]]) -> list[list[float]]:
    """Flip rows and columns: rows become columns."""
    return [list(col) for col in zip(*M)]


def det2(M: list[list[float]]) -> float:
    """Determinant of a 2x2: ad - bc.
    The (signed) area of the parallelogram that the unit square becomes."""
    return M[0][0] * M[1][1] - M[0][1] * M[1][0]


def inv2(M: list[list[float]]) -> list[list[float]]:
    """Inverse of a 2x2: swap the diagonal, negate the off-diagonal,
    divide by the determinant. If det = 0 the matrix squashed space
    onto a line — information was destroyed and there is no undo."""
    d = det2(M)
    if abs(d) < 1e-12:
        raise ValueError("singular matrix: det = 0, no inverse exists")
    a, b = M[0]
    c, e = M[1]
    return [[e / d, -b / d], [-c / d, a / d]]


def power_iteration(M: list[list[float]], steps: int = 100) -> tuple[float, list[float]]:
    """Find the dominant eigenvector: the direction M only stretches.

    Apply M over and over to any starting vector. The component along the
    largest eigenvalue's direction grows fastest, so after enough rounds
    that direction is all that's left. Normalize each step to stop the
    numbers exploding. Returns (eigenvalue, unit eigenvector).
    """
    v = [1.0, 0.7]  # any vector not perpendicular to the answer works
    for _ in range(steps):
        w = matvec(M, v)
        norm = math.sqrt(dot(w, w))
        v = [x / norm for x in w]
    # Rayleigh quotient: with v fixed, the eigenvalue is v·Mv / v·v
    lam = dot(v, matvec(M, v)) / dot(v, v)
    return lam, v


def fmt(M: list[list[float]]) -> str:
    return "  ".join(f"[{row[0]:+.4f} {row[1]:+.4f}]" for row in M)


if __name__ == "__main__":
    # 1. A shear pushes the top of space sideways; x-axis stays put.
    shear = [[1.0, 0.5], [0.0, 1.0]]
    print("— shear M = [[1, 0.5], [0, 1]] applied to vectors —")
    for v in ([1.0, 0.0], [0.0, 1.0], [1.0, 1.0]):
        print(f"  M @ {v} = {matvec(shear, v)}")
    print(f"  det = {det2(shear):.4f}  (shears preserve area)\n")

    # 2. The inverse undoes the transformation: M @ M^-1 must be identity.
    M = [[2.0, 1.0], [1.0, 3.0]]
    M_inv = inv2(M)
    I = matmul(M, M_inv)
    print(f"— inverse check for M = {fmt(M)} —")
    print(f"  M^-1      = {fmt(M_inv)}")
    print(f"  M @ M^-1  = {fmt(I)}")
    ok = all(abs(I[i][j] - (1.0 if i == j else 0.0)) < 1e-9 for i in range(2) for j in range(2))
    print(f"  identity? {ok}\n")

    # 3. Power iteration on a symmetric matrix, then verify M v ≈ λ v.
    print("— power iteration on the same symmetric M —")
    lam, v = power_iteration(M)
    Mv = matvec(M, v)
    lv = [lam * x for x in v]
    err = math.sqrt(sum((a - b) ** 2 for a, b in zip(Mv, lv)))
    print(f"  eigenvalue  λ ≈ {lam:.6f}")
    print(f"  eigenvector v ≈ [{v[0]:+.6f}, {v[1]:+.6f}]")
    print(f"  M v = [{Mv[0]:+.6f}, {Mv[1]:+.6f}]")
    print(f"  λ v = [{lv[0]:+.6f}, {lv[1]:+.6f}]")
    print(f"  ‖M v − λ v‖ = {err:.2e}  →  M only stretches v, never rotates it")
    # exact answer for [[2,1],[1,3]] is λ = (5 + √5)/2
    print(f"  exact λ = (5+√5)/2 = {(5 + math.sqrt(5)) / 2:.6f}")
