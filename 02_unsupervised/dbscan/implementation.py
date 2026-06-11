"""DBSCAN from scratch — pure standard library, zero dependencies.

Run it:  python3 implementation.py

The idea: a cluster is a region where points are packed densely.
  CORE point:   has >= min_samples neighbors within radius eps
  BORDER point: within eps of a core point, but not core itself
  NOISE:        neither — it belongs to nothing

Clusters grow by chaining: start at any core point, claim its neighbors,
and keep expanding through every core point you reach (a plain BFS queue).
Because the chain follows *density* rather than distance-to-a-center,
DBSCAN traces out arbitrary shapes that k-means can never represent.
"""

import math
import random

NOISE = -1


def dbscan(points: list, eps: float, min_samples: int) -> list[int]:
    """Returns a label per point: 0,1,2,... for clusters, -1 for noise.

    Brute-force neighbor search — O(n^2), honest and plenty fast here.
    """
    n = len(points)

    def neighbors(i):
        return [j for j in range(n)
                if math.dist(points[i], points[j]) <= eps]

    labels = [None] * n          # None = not yet visited
    cluster = 0
    for i in range(n):
        if labels[i] is not None:
            continue
        nbrs = neighbors(i)
        if len(nbrs) < min_samples:
            labels[i] = NOISE    # may be promoted to border later
            continue
        # i is a core point: flood-fill a new cluster outward from it.
        labels[i] = cluster
        queue = list(nbrs)
        while queue:
            j = queue.pop(0)
            if labels[j] == NOISE:
                labels[j] = cluster          # noise within reach -> border
            if labels[j] is not None:
                continue
            labels[j] = cluster
            j_nbrs = neighbors(j)
            if len(j_nbrs) >= min_samples:   # j is core too: chain continues
                queue.extend(j_nbrs)
            # border points join the cluster but do NOT extend it
        cluster += 1
    return labels


def kmeans(points: list, k: int, iters: int = 50) -> list[int]:
    """Tiny k-means, just for the comparison below."""
    centroids = random.sample(points, k)
    labels = [0] * len(points)
    for _ in range(iters):
        labels = [min(range(k), key=lambda c: math.dist(p, centroids[c]))
                  for p in points]
        for c in range(k):
            members = [p for p, lab in zip(points, labels) if lab == c]
            if members:
                centroids[c] = (sum(p[0] for p in members) / len(members),
                                sum(p[1] for p in members) / len(members))
    return labels


if __name__ == "__main__":
    random.seed(7)

    # Two interleaved crescent moons — the classic shape k-means cannot split.
    points, truth = [], []
    for _ in range(120):
        t = random.uniform(0, math.pi)
        points.append((math.cos(t) + random.gauss(0, 0.07),
                       math.sin(t) + random.gauss(0, 0.07)))
        truth.append(0)
    for _ in range(120):
        t = random.uniform(0, math.pi)
        points.append((1.0 - math.cos(t) + random.gauss(0, 0.07),
                       0.5 - math.sin(t) + random.gauss(0, 0.07)))
        truth.append(1)

    def accuracy(labels):
        """Best-of-two label matching for a 2-cluster problem."""
        a = sum(l == t for l, t in zip(labels, truth))
        b = sum((1 - l if l in (0, 1) else l) == t for l, t in zip(labels, truth))
        return max(a, b) / len(truth)

    db = dbscan(points, eps=0.22, min_samples=4)
    n_clusters = len(set(l for l in db if l != NOISE))
    n_noise = sum(l == NOISE for l in db)
    print(f"DBSCAN:  {n_clusters} clusters, {n_noise} noise points, "
          f"accuracy vs truth {100 * accuracy(db):.1f}%")

    km = kmeans(points, k=2)
    print(f"k-means: 2 clusters (forced),               "
          f"accuracy vs truth {100 * accuracy(km):.1f}%")

    print("\nDBSCAN chains through each crescent's density and recovers both")
    print("moons; k-means slices the plane with a straight boundary and")
    print("mislabels the moons' interleaved tips.")
