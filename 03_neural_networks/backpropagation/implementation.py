"""Backpropagation from scratch — pure standard library, zero dependencies.

Run it:  python3 implementation.py

A fully-connected network with sigmoid activations, trained on XOR — the
canonical "you need a hidden layer for this" problem. Every equation from
the README appears here with the same symbols:

    forward:   z[l] = W[l] @ a[l-1] + b[l],   a[l] = sigmoid(z[l])
    output:    delta[L] = (a[L] - y) * sigmoid'(z[L])
    backward:  delta[l] = (W[l+1].T @ delta[l+1]) * sigmoid'(z[l])
    gradients: dL/dW[l] = delta[l] @ a[l-1].T,   dL/db[l] = delta[l]

No numpy: "matrices" are lists of lists, and the three helper functions at
the top are the only linear algebra we need.
"""

import math
import random

# ---------------------------------------------------------------------------
# Minimal linear algebra on lists (vectors) and lists-of-lists (matrices)
# ---------------------------------------------------------------------------

def matvec(M: list[list[float]], v: list[float]) -> list[float]:
    """M @ v"""
    return [sum(M_ij * v_j for M_ij, v_j in zip(row, v)) for row in M]


def matvec_T(M: list[list[float]], v: list[float]) -> list[float]:
    """M.T @ v — used in the backward pass: blame flows back along the
    same connections the activations flowed forward, hence the transpose."""
    return [sum(M[i][j] * v[i] for i in range(len(M))) for j in range(len(M[0]))]


def outer(u: list[float], v: list[float]) -> list[list[float]]:
    """u @ v.T — the gradient of a weight matrix is an outer product:
    (blame of the neuron it feeds) x (activation of the neuron it leaves)."""
    return [[u_i * v_j for v_j in v] for u_i in u]


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def dsigmoid_from_a(a: float) -> float:
    """sigma'(z) written in terms of the activation: a * (1 - a).
    Note it peaks at 0.25 — every layer of sigmoid multiplies the gradient
    by at most 0.25, which is the vanishing-gradient problem in one line."""
    return a * (1.0 - a)


# ---------------------------------------------------------------------------
# The network
# ---------------------------------------------------------------------------

class MLP:
    def __init__(self, sizes: list[int], seed: int = 0):
        """sizes e.g. [2, 4, 1]: 2 inputs, one hidden layer of 4, 1 output.

        Weights are initialized small and RANDOM — initialize them all equal
        and every neuron in a layer gets identical gradients forever (the
        symmetry problem), collapsing the layer to one effective neuron.
        """
        rng = random.Random(seed)
        self.sizes = sizes
        self.W = [
            [[rng.gauss(0, 1) / math.sqrt(n_in) for _ in range(n_in)]
             for _ in range(n_out)]
            for n_in, n_out in zip(sizes[:-1], sizes[1:])
        ]
        self.b = [[0.0] * n_out for n_out in sizes[1:]]

    def forward(self, x: list[float]) -> list[list[float]]:
        """Forward pass. Returns the activations of EVERY layer, because the
        backward pass needs them (a weight's gradient uses the activation
        that flowed through it)."""
        activations = [x]
        for W_l, b_l in zip(self.W, self.b):
            z = [zj + bj for zj, bj in zip(matvec(W_l, activations[-1]), b_l)]
            activations.append([sigmoid(zj) for zj in z])
        return activations

    def backward(
        self, activations: list[list[float]], y: list[float]
    ) -> tuple[list, list]:
        """The algorithm. One pass, output to input, reusing each layer's
        delta — this caching is why backprop costs O(edges), not O(paths)."""
        # Blame at the output: how wrong, scaled by how responsive.
        delta = [
            (a - t) * dsigmoid_from_a(a)
            for a, t in zip(activations[-1], y)
        ]

        grads_W = [None] * len(self.W)
        grads_b = [None] * len(self.b)

        for l in range(len(self.W) - 1, -1, -1):
            # Gradients for this layer's parameters.
            grads_W[l] = outer(delta, activations[l])
            grads_b[l] = delta[:]
            if l > 0:
                # Propagate blame one layer back: transpose-multiply,
                # then scale by the local slope of the activation.
                upstream = matvec_T(self.W[l], delta)
                delta = [
                    u * dsigmoid_from_a(a)
                    for u, a in zip(upstream, activations[l])
                ]
        return grads_W, grads_b

    def train_step(self, x: list[float], y: list[float], alpha: float) -> float:
        """Forward, backward, update. Returns the loss before the update."""
        activations = self.forward(x)
        loss = 0.5 * sum((a - t) ** 2 for a, t in zip(activations[-1], y))
        grads_W, grads_b = self.backward(activations, y)
        for l in range(len(self.W)):
            for j in range(len(self.W[l])):
                for i in range(len(self.W[l][j])):
                    self.W[l][j][i] -= alpha * grads_W[l][j][i]
                self.b[l][j] -= alpha * grads_b[l][j]
        return loss


if __name__ == "__main__":
    # XOR: not linearly separable, so a perceptron CANNOT learn it —
    # the hidden layer plus backprop is the whole point.
    data = [
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.0, 1.0], [0.0]),
    ]

    net = MLP([2, 4, 1], seed=1)
    rng = random.Random(2)

    for epoch in range(10001):
        x, y = rng.choice(data)
        loss = net.train_step(x, y, alpha=2.0)
        if epoch % 1000 == 0:
            total = sum(
                0.5 * (net.forward(x)[-1][0] - y[0]) ** 2 for x, y in data
            )
            print(f"epoch {epoch:5d}   total loss {total:.6f}")

    print("\nlearned XOR:")
    for x, y in data:
        pred = net.forward(x)[-1][0]
        print(f"  {x} -> {pred:.3f}   (target {y[0]:.0f})")
