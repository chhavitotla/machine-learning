"""The perceptron (Rosenblatt, 1958) from scratch — pure stdlib, zero dependencies.

Run it:  python3 implementation.py

The model:   y_hat = sign(w·x + b)
The rule:    if a point is misclassified:  w += alpha·y·x,  b += alpha·y
             (correct points change nothing)

That's the whole algorithm. If the data is linearly separable it provably
converges in finitely many mistakes (Novikoff, 1962). If it isn't — XOR —
it cycles forever, which is exactly the limitation (Minsky & Papert, 1969)
that stalled neural networks until hidden layers + backprop fixed it:
a hidden layer lets the network BUILD the feature (like x1 XOR x2) that
makes the problem linearly separable in the new space.
"""


def predict(w, b, x):
    """The entire model: which side of the line is x on?"""
    return 1 if w[0] * x[0] + w[1] * x[1] + b >= 0 else -1


def train(X, y, alpha=1.0, max_epochs=100):
    """The 1958 learning rule. Returns (w, b, epochs_used, converged)."""
    w, b = [0.0, 0.0], 0.0
    for epoch in range(1, max_epochs + 1):
        mistakes = 0
        for xi, yi in zip(X, y):
            if predict(w, b, xi) != yi:
                # Nudge the boundary TOWARD classifying this point correctly:
                # adding y·x rotates/shifts w so that w·x moves in y's direction.
                w[0] += alpha * yi * xi[0]
                w[1] += alpha * yi * xi[1]
                b += alpha * yi
                mistakes += 1
        if mistakes == 0:
            return w, b, epoch, True       # a clean epoch = converged, provably done
    return w, b, max_epochs, False


def demo(name, table):
    X = [t[0] for t in table]
    y = [t[1] for t in table]
    w, b, epochs, ok = train(X, y)
    acc = sum(predict(w, b, xi) == yi for xi, yi in zip(X, y)) / len(X)
    status = f"converged in {epochs} epochs" if ok else f"NO convergence after {epochs} epochs"
    print(f"{name:4s}  {status:32s}  w = [{w[0]:+.1f}, {w[1]:+.1f}]  "
          f"b = {b:+.1f}  accuracy = {acc:.0%}")
    return ok


if __name__ == "__main__":
    # Truth tables as 2D points with labels in {-1, +1}
    AND = [([0, 0], -1), ([0, 1], -1), ([1, 0], -1), ([1, 1], +1)]
    OR  = [([0, 0], -1), ([0, 1], +1), ([1, 0], +1), ([1, 1], +1)]
    XOR = [([0, 0], -1), ([0, 1], +1), ([1, 0], +1), ([1, 1], -1)]

    print("linearly separable problems — the convergence theorem applies:")
    demo("AND", AND)
    demo("OR", OR)

    print("\nnot linearly separable — no line can ever split XOR:")
    ok = demo("XOR", XOR)
    if not ok:
        print(
            "\nThe weights cycle forever: every update that fixes one corner of\n"
            "the XOR square breaks another. No single line separates the\n"
            "classes, so zero mistakes is unreachable. THIS is why we need\n"
            "hidden layers: a 2-neuron hidden layer can compute (x1 OR x2) and\n"
            "(x1 NAND x2), and XOR = AND of those — linearly separable in the\n"
            "hidden space. Stacking perceptrons = a neural network."
        )
