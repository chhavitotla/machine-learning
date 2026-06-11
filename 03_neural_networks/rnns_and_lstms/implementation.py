"""RNN cell vs LSTM cell, forward pass only — pure standard library.

Run it:  python3 implementation.py

The point of this file is ONE number per timestep: the size of the local
backward-pass factor.  When you backpropagate through time, the gradient at
step t is a PRODUCT of these per-step factors:

    RNN:   dh_t/dh_{t-1} = diag(1 - h_t^2) . W_hh     (a full matrix)
    LSTM:  dc_t/dc_{t-1} ~= diag(f_t)                  (the cell highway)

If the RNN factor's norm is below 1 (it usually is — tanh saturates and
W_hh is small), the product collapses exponentially: vanishing gradients.
The LSTM's highway factor is just the forget gate, which the network can
hold near 1 — so the product survives.  The demo prints both products
side by side over the same string.
"""

import math
import random

H = 8                      # hidden units
TEXT = "hello world"


def matrix(rows, cols, scale, rnd):
    return [[rnd.gauss(0, scale) for _ in range(cols)] for _ in range(rows)]


def matvec(M, v):
    return [sum(w * x for w, x in zip(row, v)) for row in M]


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


def norm(v):
    return math.sqrt(sum(x * x for x in v))


def spectral_norm(M, iters=50):
    """Largest singular value via power iteration on M^T M.

    This is the honest 'how much can this matrix stretch a vector' number —
    exactly what matters when the backward pass multiplies by it repeatedly.
    """
    n = len(M[0])
    v = [1.0] * n
    for _ in range(iters):
        u = matvec(M, v)                                   # u = M v
        v = [sum(M[i][j] * u[i] for i in range(len(M))) for j in range(n)]
        nv = norm(v) or 1.0
        v = [x / nv for x in v]
    return norm(matvec(M, v))


class RNNCell:
    """h_t = tanh(W_xh x_t + W_hh h_{t-1} + b)"""

    def __init__(self, vocab, rnd):
        self.Wxh = matrix(H, vocab, 0.3, rnd)
        self.Whh = matrix(H, H, 0.08, rnd)   # small: typical init, and the villain
        self.b = [0.0] * H

    def step(self, x, h):
        z = [a + c + b for a, c, b in zip(matvec(self.Wxh, x), matvec(self.Whh, h), self.b)]
        h_new = [math.tanh(zi) for zi in z]
        # Local backward factor: J = diag(1 - h^2) . W_hh.
        # tanh' = 1 - h^2 is <= 1 (0 when saturated), so J only shrinks W_hh.
        J = [[(1 - h_new[i] ** 2) * self.Whh[i][j] for j in range(H)] for i in range(H)]
        return h_new, spectral_norm(J)


class LSTMCell:
    """Gates read [x_t, h_{t-1}]; the cell state c is an additive highway:
        c_t = f_t * c_{t-1} + i_t * g_t       h_t = o_t * tanh(c_t)
    """

    def __init__(self, vocab, rnd):
        d = vocab + H
        self.Wf, self.Wi, self.Wo, self.Wg = (matrix(H, d, 0.3, rnd) for _ in range(4))
        # Forget bias initialized HIGH (the classic trick): start by remembering.
        # The other gates' biases are zero, so they're simply omitted below.
        self.bf = [3.0] * H

    def step(self, x, h, c):
        xh = x + h
        f = [sigmoid(z + b) for z, b in zip(matvec(self.Wf, xh), self.bf)]
        i = [sigmoid(z) for z in matvec(self.Wi, xh)]
        o = [sigmoid(z) for z in matvec(self.Wo, xh)]
        g = [math.tanh(z) for z in matvec(self.Wg, xh)]
        c_new = [ff * cc + ii * gg for ff, cc, ii, gg in zip(f, c, i, g)]
        h_new = [oo * math.tanh(cc) for oo, cc in zip(o, c_new)]
        # Dominant backward factor along the cell highway: dc_t/dc_{t-1} = diag(f).
        # A diagonal matrix's spectral norm is its largest entry — a gate value,
        # which the network can hold near 1 to preserve gradient.
        return h_new, c_new, max(f)


if __name__ == "__main__":
    rnd = random.Random(42)                      # seeded: same weights every run
    chars = sorted(set(TEXT))
    onehot = {ch: [1.0 if k == idx else 0.0 for k in range(len(chars))]
              for idx, ch in enumerate(chars)}

    rnn = RNNCell(len(chars), rnd)
    lstm = LSTMCell(len(chars), rnd)

    h_r = [0.0] * H                              # RNN hidden state
    h_l, c_l = [0.0] * H, [0.0] * H              # LSTM hidden + cell state
    prod_r = prod_l = 1.0                        # running gradient-factor products

    print(f'feeding "{TEXT}" one character at a time\n')
    print("step char | rnn ||h||  factor  grad-product | lstm ||h||  forget  grad-product")
    print("-" * 79)
    for t, ch in enumerate(TEXT):
        x = onehot[ch]
        h_r, jr = rnn.step(x, h_r)
        h_l, c_l, jl = lstm.step(x, h_l, c_l)
        prod_r *= jr                             # what reaches step 0 from step t
        prod_l *= jl
        print(f"{t:4d}  '{ch}' |   {norm(h_r):6.3f}  {jr:6.3f}  {prod_r:12.2e} "
              f"|    {norm(h_l):6.3f}  {jl:6.3f}  {prod_l:12.2e}")

    print("-" * 79)
    print(f"\nafter {len(TEXT)} steps, the gradient surviving back to step 0:")
    print(f"  RNN : {prod_r:.2e}   (each step multiplied by ~{jr:.2f} -> exponential decay)")
    print(f"  LSTM: {prod_l:.2e}   (forget gates near 1 keep the highway open)")
    print("\nSame string, same depth. The RNN's product collapses toward zero —")
    print("early characters become invisible to learning. The LSTM's cell state")
    print("is additive, so its factor is a gate the network controls, not a")
    print("squashed matrix product. That single design change is the LSTM.")
