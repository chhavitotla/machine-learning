"""Random forest from scratch — pure standard library, zero dependencies.

Run it:  python3 implementation.py

A random forest is many overfit decision trees that disagree, averaged:
  1. BOOTSTRAP: each tree trains on n points sampled WITH replacement,
     so each sees a slightly different world (~63% unique points).
  2. FEATURE RANDOMNESS: each split considers only a random subset of
     features, so the trees can't all ask the same questions.
  3. MAJORITY VOTE: each tree is high-variance and wrong in its own way;
     the errors are (partly) independent, so voting cancels them out.

Bias stays roughly the same as one tree's; variance divides down with the
number of (de-correlated) trees. That's the entire trick.
"""

import random

# ---------- a minimal decision tree (same CART as ../decision_trees) ----------

def gini(labels):
    """1 - sum of squared class proportions; 0 = pure, 0.5 = coin flip."""
    if not labels:
        return 0.0
    p1 = sum(labels) / len(labels)
    return 1.0 - p1 * p1 - (1 - p1) * (1 - p1)


def best_split(X, y, feat_indices):
    """Best (feature, threshold) by Gini gain — but only among feat_indices.

    Restricting the features per split is what de-correlates the trees:
    a forest where every tree roots on the same dominant feature is just
    one tree photocopied.
    """
    parent, n, best = gini(y), len(y), None
    for f in feat_indices:
        values = sorted(set(row[f] for row in X))
        for lo, hi in zip(values, values[1:]):
            t = (lo + hi) / 2
            left = [y[i] for i in range(n) if X[i][f] <= t]
            right = [y[i] for i in range(n) if X[i][f] > t]
            gain = parent - (len(left) * gini(left) + len(right) * gini(right)) / n
            if best is None or gain > best[0]:
                best = (gain, f, t)
    return None if (best is None or best[0] <= 1e-12) else best


def build(X, y, rng, max_depth, n_feats, depth=0):
    """Grow greedily; sample a fresh random feature subset at every split."""
    majority = 1 if sum(y) * 2 >= len(y) else 0
    if depth >= max_depth or gini(y) == 0.0:
        return {"leaf": True, "label": majority}
    feats = rng.sample(range(len(X[0])), n_feats)
    split = best_split(X, y, feats)
    if split is None:
        return {"leaf": True, "label": majority}
    _, f, t = split
    L = [i for i in range(len(y)) if X[i][f] <= t]
    R = [i for i in range(len(y)) if X[i][f] > t]
    return {"leaf": False, "feature": f, "threshold": t,
            "left": build([X[i] for i in L], [y[i] for i in L], rng, max_depth, n_feats, depth + 1),
            "right": build([X[i] for i in R], [y[i] for i in R], rng, max_depth, n_feats, depth + 1)}


def tree_predict(node, x):
    while not node["leaf"]:
        node = node["left"] if x[node["feature"]] <= node["threshold"] else node["right"]
    return node["label"]


# ---------- the forest: bootstrap + random features + majority vote ----------

def fit_forest(X, y, rng, n_trees=30, max_depth=8, n_feats=1):
    """Each tree: a bootstrap resample of the data, random features per split."""
    forest = []
    n = len(y)
    for _ in range(n_trees):
        idx = [rng.randrange(n) for _ in range(n)]      # sample WITH replacement
        Xb = [X[i] for i in idx]
        yb = [y[i] for i in idx]
        forest.append(build(Xb, yb, rng, max_depth, n_feats))
    return forest


def forest_predict(forest, x):
    """Democracy: every tree votes, majority wins."""
    votes = sum(tree_predict(t, x) for t in forest)
    return 1 if votes * 2 >= len(forest) else 0


def accuracy(predict_fn, X, y):
    return sum(1 for xi, yi in zip(X, y) if predict_fn(xi) == yi) / len(y)


if __name__ == "__main__":
    rng = random.Random(7)

    # Ground truth: class 1 inside a circle — a shape axis-aligned cuts can
    # only staircase around, so trees must go deep, so they overfit, so
    # averaging has real variance to remove.
    def true_label(x1, x2):
        return 1 if (x1 - 0.5) ** 2 + (x2 - 0.5) ** 2 < 0.09 else 0

    def make(n, label_noise):
        X, y = [], []
        for _ in range(n):
            x1, x2 = rng.random(), rng.random()
            lbl = true_label(x1, x2)
            if rng.random() < label_noise:
                lbl = 1 - lbl                       # mislabeled point: pure noise
            X.append([x1, x2])
            y.append(lbl)
        return X, y

    X_train, y_train = make(300, label_noise=0.15)  # noisy training labels
    X_test, y_test = make(400, label_noise=0.0)     # clean test = the truth

    # one deep tree, seeing all features: free to memorize every noisy label
    single = build(X_train, y_train, rng, max_depth=12, n_feats=2)
    acc_tree = accuracy(lambda x: tree_predict(single, x), X_test, y_test)

    forest = fit_forest(X_train, y_train, rng, n_trees=30, max_depth=8, n_feats=1)
    acc_forest = accuracy(lambda x: forest_predict(forest, x), X_test, y_test)

    print("trained on 300 points with 15% of labels flipped at random\n")
    print(f"single deep tree   test accuracy: {acc_tree:.1%}")
    print(f"forest of 30 trees test accuracy: {acc_forest:.1%}")
    print(f"\nforest wins by {(acc_forest - acc_tree) * 100:+.1f} points.")
    print("Each tree memorized different noise (different bootstrap, different")
    print("features) — the vote keeps the shared signal and cancels the rest.")
