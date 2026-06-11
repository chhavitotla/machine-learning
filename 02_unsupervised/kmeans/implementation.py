"""K-means clustering from scratch — pure standard library, zero dependencies.

Run it:  python3 implementation.py

The algorithm alternates two phases until nothing changes:
  ASSIGN: give every point to its nearest centroid
  UPDATE: move every centroid to the mean of its points

Each phase can only lower the inertia (within-cluster sum of squared
distances), so the loop always converges — just not always to the best
answer, which is why we restart from several random inits and keep the best.
"""

import random


def dist2(a: tuple, b: tuple) -> float:
    """Squared euclidean distance — no sqrt needed for comparisons."""
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def assign(points: list, centroids: list) -> list[int]:
    """Phase A: label each point with the index of its nearest centroid."""
    return [min(range(len(centroids)), key=lambda k: dist2(p, centroids[k]))
            for p in points]


def update(points: list, labels: list[int], k: int, old: list) -> list:
    """Phase B: move each centroid to the mean of its assigned points.

    An empty cluster keeps its old centroid rather than crashing —
    the simplest of several standard fixes.
    """
    centroids = []
    for j in range(k):
        members = [p for p, lab in zip(points, labels) if lab == j]
        if members:
            centroids.append((sum(p[0] for p in members) / len(members),
                              sum(p[1] for p in members) / len(members)))
        else:
            centroids.append(old[j])
    return centroids


def inertia(points: list, labels: list[int], centroids: list) -> float:
    """Within-cluster sum of squares — the quantity k-means minimizes."""
    return sum(dist2(p, centroids[lab]) for p, lab in zip(points, labels))


def kmeans_once(points: list, k: int, max_iter: int = 100, verbose: bool = False):
    """One run from a random init: alternate assign/update until labels freeze."""
    centroids = random.sample(points, k)   # plain random init: k actual data points
    labels = assign(points, centroids)
    for it in range(max_iter):
        centroids = update(points, labels, k, centroids)
        new_labels = assign(points, centroids)
        if verbose:
            print(f"  iter {it:2d}   inertia {inertia(points, new_labels, centroids):8.3f}")
        if new_labels == labels:           # no point changed cluster -> converged
            break
        labels = new_labels
    return centroids, labels, inertia(points, labels, centroids)


def kmeans(points: list, k: int, restarts: int = 3, verbose: bool = False):
    """Best-of-N restarts: k-means only finds a local optimum, so run it a few
    times from different random inits and keep the run with the lowest inertia."""
    best = None
    for r in range(restarts):
        if verbose:
            print(f"restart {r + 1}:")
        result = kmeans_once(points, k, verbose=verbose)
        if best is None or result[2] < best[2]:
            best = result
    return best


if __name__ == "__main__":
    random.seed(7)

    # Three synthetic gaussian blobs with known centers.
    TRUE_CENTERS = [(2.0, 2.0), (8.0, 3.0), (5.0, 8.0)]
    points = []
    for cx, cy in TRUE_CENTERS:
        points += [(random.gauss(cx, 0.6), random.gauss(cy, 0.6)) for _ in range(60)]
    random.shuffle(points)

    print("ground-truth blob centers:", [(f"{x:.1f}", f"{y:.1f}") for x, y in TRUE_CENTERS])
    print()
    centroids, labels, J = kmeans(points, k=3, restarts=3, verbose=True)

    print(f"\nbest inertia: {J:.3f}")
    print("recovered centers (sorted by x):")
    for cx, cy in sorted(centroids):
        print(f"  ({cx:.3f}, {cy:.3f})")
    print("\nEach recovered center lands within noise of a true blob center —")
    print("the assign/update loop found the structure with no labels at all.")
