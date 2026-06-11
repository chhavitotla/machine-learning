"""PCA from scratch in 2D — pure standard library, zero dependencies.

Run it:  python3 implementation.py

The whole algorithm:
  1. center the data (subtract the mean)
  2. build the 2x2 covariance matrix
  3. eigendecompose it — closed form for 2x2, no iteration needed
  4. the eigenvectors are the principal components; the eigenvalues are
     the variance captured along each one

Projecting onto the top component keeps the direction of maximum variance,
which is exactly the projection with minimum squared reconstruction error.
"""

import math
import random


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def covariance_matrix(points: list) -> tuple[float, float, float]:
    """Returns (cxx, cxy, cyy) of the symmetric 2x2 covariance matrix.

    cxx = var(x), cyy = var(y), cxy = cov(x, y). Symmetric, so 3 numbers.
    """
    mx = mean([p[0] for p in points])
    my = mean([p[1] for p in points])
    n = len(points)
    cxx = sum((p[0] - mx) ** 2 for p in points) / n
    cyy = sum((p[1] - my) ** 2 for p in points) / n
    cxy = sum((p[0] - mx) * (p[1] - my) for p in points) / n
    return cxx, cxy, cyy


def eig2x2(cxx: float, cxy: float, cyy: float):
    """Exact eigendecomposition of a symmetric 2x2 matrix.

    Eigenvalues solve the characteristic polynomial:
        lambda = (trace ± sqrt(trace^2 - 4*det)) / 2
    The eigenvector for lambda satisfies (cxx - lambda)*vx + cxy*vy = 0,
    so v = (cxy, lambda - cxx) — unless cxy = 0, when the matrix is already
    diagonal and the axes themselves are the eigenvectors.

    Returns ((l1, v1), (l2, v2)) with l1 >= l2 and unit-length vectors.
    """
    tr, det = cxx + cyy, cxx * cyy - cxy * cxy
    gap = math.sqrt(max(tr * tr - 4 * det, 0.0))
    l1, l2 = (tr + gap) / 2, (tr - gap) / 2

    def unit_vec(lam):
        if abs(cxy) > 1e-12:
            vx, vy = cxy, lam - cxx
        else:                                  # already diagonal
            vx, vy = (1.0, 0.0) if cxx >= cyy else (0.0, 1.0)
            if lam == min(cxx, cyy):
                vx, vy = vy, vx
        norm = math.hypot(vx, vy)
        return (vx / norm, vy / norm)

    return (l1, unit_vec(l1)), (l2, unit_vec(l2))


def project_reconstruct(points: list, v: tuple, mx: float, my: float) -> list:
    """Project each centered point onto direction v, then map back to 2D.

    reconstruction = mean + (point-mean · v) * v  — the closest point on the
    PC1 line. What's thrown away is each point's coordinate along PC2.
    """
    out = []
    for x, y in points:
        t = (x - mx) * v[0] + (y - my) * v[1]   # scalar coordinate along v
        out.append((mx + t * v[0], my + t * v[1]))
    return out


if __name__ == "__main__":
    random.seed(7)

    # Correlated 2D gaussian: y leans on x, so variance concentrates
    # along a diagonal direction that neither raw axis captures.
    points = []
    for _ in range(300):
        x = random.gauss(0, 2.0)
        y = 0.6 * x + random.gauss(0, 0.8)
        points.append((x, y))

    cxx, cxy, cyy = covariance_matrix(points)
    (l1, v1), (l2, v2) = eig2x2(cxx, cxy, cyy)
    total = l1 + l2

    print(f"covariance matrix:  [[{cxx:6.3f} {cxy:6.3f}]")
    print(f"                     [{cxy:6.3f} {cyy:6.3f}]]")
    print(f"eigenvalues:        l1 = {l1:.3f}   l2 = {l2:.3f}")
    print(f"PC1 direction:      ({v1[0]:+.3f}, {v1[1]:+.3f})")
    print(f"explained variance: PC1 {100 * l1 / total:.1f}%   PC2 {100 * l2 / total:.1f}%")

    # Verify: reconstructing from PC1 alone must lose exactly the variance
    # that lives along PC2 — i.e. mean squared error == l2.
    mx = mean([p[0] for p in points])
    my = mean([p[1] for p in points])
    recon = project_reconstruct(points, v1, mx, my)
    mse = mean([(x - rx) ** 2 + (y - ry) ** 2
                for (x, y), (rx, ry) in zip(points, recon)])
    print(f"\nreconstruction MSE from 1 component: {mse:.4f}")
    print(f"discarded eigenvalue l2:             {l2:.4f}")
    assert abs(mse - l2) < 1e-9, "reconstruction error must equal l2 exactly"
    print("match — the error of keeping PC1 is exactly the variance along PC2.")
