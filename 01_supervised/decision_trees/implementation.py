"""Decision tree (CART) from scratch — pure standard library, zero dependencies.

Run it:  python3 implementation.py

The algorithm, in one breath:
  1. At a node, try every threshold on every feature as a yes/no split.
  2. Score each split by how much it reduces Gini impurity (weighted by
     how many points land on each side) — the "information gain".
  3. Take the best split, recurse left and right, stop when pure or deep.

Gini impurity of a node:  G = 1 - sum(p_k^2) over classes k.
  G = 0   -> all one class (pure)
  G = 0.5 -> a 50/50 coin flip (maximally confused, two classes)
It's the probability that two random draws from the node disagree.
"""


def gini(labels: list[int]) -> float:
    """1 - sum of squared class proportions. Zero means pure."""
    n = len(labels)
    if n == 0:
        return 0.0
    p1 = sum(labels) / n
    return 1.0 - p1 * p1 - (1 - p1) * (1 - p1)


def best_split(X: list[list[float]], y: list[int]):
    """Exhaustive scan: every feature, every midpoint between sorted values.

    Returns (feature, threshold, gain) or None if nothing helps.
    The gain is parent Gini minus the size-weighted child Ginis.
    """
    n = len(y)
    parent = gini(y)
    best = None  # (gain, feature, threshold)
    for f in range(len(X[0])):
        values = sorted(set(row[f] for row in X))
        for lo, hi in zip(values, values[1:]):
            t = (lo + hi) / 2                       # split between neighbors
            left = [y[i] for i in range(n) if X[i][f] <= t]
            right = [y[i] for i in range(n) if X[i][f] > t]
            weighted = (len(left) * gini(left) + len(right) * gini(right)) / n
            gain = parent - weighted
            if best is None or gain > best[0]:
                best = (gain, f, t)
    if best is None or best[0] <= 1e-12:
        return None
    return best[1], best[2], best[0]


def build(X: list[list[float]], y: list[int], depth: int = 0, max_depth: int = 4) -> dict:
    """Recursive CART: split greedily until pure, depth-capped, or no gain.

    Greedy means each split is locally best — the tree never reconsiders.
    That's why two mediocre splits that combine brilliantly (e.g. XOR)
    can be missed.
    """
    majority = 1 if sum(y) * 2 >= len(y) else 0
    if depth >= max_depth or gini(y) == 0.0 or (split := best_split(X, y)) is None:
        return {"leaf": True, "label": majority, "n": len(y)}
    f, t, gain = split
    L = [i for i in range(len(y)) if X[i][f] <= t]
    R = [i for i in range(len(y)) if X[i][f] > t]
    return {
        "leaf": False, "feature": f, "threshold": t, "gain": gain, "n": len(y),
        "left": build([X[i] for i in L], [y[i] for i in L], depth + 1, max_depth),
        "right": build([X[i] for i in R], [y[i] for i in R], depth + 1, max_depth),
    }


def predict(node: dict, x: list[float]) -> int:
    """Walk the tree: one comparison per level, land in a leaf."""
    while not node["leaf"]:
        node = node["left"] if x[node["feature"]] <= node["threshold"] else node["right"]
    return node["label"]


def print_tree(node: dict, names: list[str], indent: str = "") -> None:
    """The tree IS the explanation — print it as nested if/else rules."""
    if node["leaf"]:
        print(f"{indent}-> predict class {node['label']}  ({node['n']} samples)")
        return
    name, t = names[node["feature"]], node["threshold"]
    print(f"{indent}if {name} <= {t:.2f}:  (gain {node['gain']:.3f}, {node['n']} samples)")
    print_tree(node["left"], names, indent + "    ")
    print(f"{indent}else:  # {name} > {t:.2f}")
    print_tree(node["right"], names, indent + "    ")


if __name__ == "__main__":
    # Tiny hardcoded dataset: should we play tennis?
    # Features: [temperature °C, humidity %].  Label: 1 = play, 0 = stay in.
    # The pattern: play when it's mild AND not too humid — an axis-aligned
    # rectangle, exactly the shape trees carve naturally.
    FEATURES = ["temp", "humidity"]
    X = [
        [12, 40], [14, 55], [16, 50], [18, 45], [20, 60], [22, 55],  # mild, dry
        [24, 50], [26, 65], [21, 40], [19, 70],                      # mild-ish
        [31, 60], [33, 45], [35, 70], [30, 80], [34, 85],            # too hot
        [5, 50], [7, 65], [3, 40], [8, 75], [6, 90],                 # too cold
        [20, 92], [24, 95], [18, 88], [22, 91],                      # too humid
    ]
    y = [1, 1, 1, 1, 1, 1,
         1, 1, 1, 1,
         0, 0, 0, 0, 0,
         0, 0, 0, 0, 0,
         0, 0, 0, 0]

    tree = build(X, y, max_depth=3)

    print("— learned decision rules —\n")
    print_tree(tree, FEATURES)

    correct = sum(1 for xi, yi in zip(X, y) if predict(tree, xi) == yi)
    print(f"\ntraining accuracy: {correct}/{len(y)} = {correct / len(y):.1%}")
    print("\nEach split was the single greediest Gini reduction available —")
    print("no global plan, just the best question to ask right now.")
