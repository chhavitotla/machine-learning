"""SGD, Momentum, and Adam from scratch — pure stdlib, zero dependencies.

Run it:  python3 implementation.py

Each optimizer is ~10 lines: the exact update rules from the papers,
implemented over plain lists of floats. They race on the Rosenbrock
function — the classic banana-shaped valley that punishes naive descent.
"""

import math


# ---------------------------------------------------------------------------
# The optimizers. Each exposes .step(params, grads) -> new params.
# ---------------------------------------------------------------------------

class SGD:
    """theta <- theta - alpha * grad. The whole algorithm."""

    def __init__(self, alpha: float):
        self.alpha = alpha

    def step(self, params: list[float], grads: list[float]) -> list[float]:
        return [p - self.alpha * g for p, g in zip(params, grads)]


class Momentum:
    """v <- beta*v + grad;  theta <- theta - alpha*v.

    The velocity buffer is a low-pass filter over gradients: components
    that flip sign every step (ravine walls) cancel; components that
    persist (the valley floor) compound up to 1/(1-beta) = 10x.
    """

    def __init__(self, alpha: float, beta: float = 0.9):
        self.alpha, self.beta = alpha, beta
        self.v: list[float] | None = None

    def step(self, params: list[float], grads: list[float]) -> list[float]:
        if self.v is None:
            self.v = [0.0] * len(params)
        self.v = [self.beta * v + g for v, g in zip(self.v, grads)]
        return [p - self.alpha * v for p, v in zip(params, self.v)]


class Adam:
    """Per-parameter adaptive steps: theta <- theta - alpha * m_hat / (sqrt(v_hat) + eps).

    m tracks the mean gradient (momentum), v tracks the mean SQUARED
    gradient (scale). Dividing by sqrt(v) means a parameter with
    consistently large gradients takes small relative steps and a
    parameter with tiny gradients still moves at full speed.
    """

    def __init__(self, alpha: float, b1: float = 0.9, b2: float = 0.999,
                 eps: float = 1e-8):
        self.alpha, self.b1, self.b2, self.eps = alpha, b1, b2, eps
        self.m: list[float] | None = None
        self.v: list[float] | None = None
        self.t = 0

    def step(self, params: list[float], grads: list[float]) -> list[float]:
        if self.m is None:
            self.m = [0.0] * len(params)
            self.v = [0.0] * len(params)
        self.t += 1
        self.m = [self.b1 * m + (1 - self.b1) * g for m, g in zip(self.m, grads)]
        self.v = [self.b2 * v + (1 - self.b2) * g * g for v, g in zip(self.v, grads)]
        # Bias correction: m and v start at zero, so for the first ~1/(1-beta)
        # steps they underestimate the true moments; dividing by (1 - beta^t)
        # restores the correct scale. Skip this and early steps are huge.
        out = []
        for p, m, v in zip(params, self.m, self.v):
            m_hat = m / (1 - self.b1 ** self.t)
            v_hat = v / (1 - self.b2 ** self.t)
            out.append(p - self.alpha * m_hat / (math.sqrt(v_hat) + self.eps))
        return out


# ---------------------------------------------------------------------------
# The arena: Rosenbrock's banana valley, minimum at (1, 1).
# ---------------------------------------------------------------------------

def rosenbrock(p: list[float]) -> float:
    x, y = p
    return (1 - x) ** 2 + 100 * (y - x * x) ** 2


def rosenbrock_grad(p: list[float]) -> list[float]:
    x, y = p
    return [
        -2 * (1 - x) - 400 * x * (y - x * x),
        200 * (y - x * x),
    ]


if __name__ == "__main__":
    START = [-1.5, 2.0]
    STEPS = 20_000

    racers = {
        "SGD": SGD(alpha=1e-3),
        "Momentum": Momentum(alpha=1e-4),   # momentum amplifies ~10x, so alpha/10
        "Adam": Adam(alpha=2e-2),
    }

    print(f"start {START}, target (1, 1), budget {STEPS} steps, "
          f"finish line: loss < 1e-6\n")
    print(f"{'optimizer':<10} {'steps to converge':>18} {'final point':>22}")
    for name, opt in racers.items():
        p = START[:]
        steps_needed = None
        for t in range(1, STEPS + 1):
            p = opt.step(p, rosenbrock_grad(p))
            if steps_needed is None and rosenbrock(p) < 1e-6:
                steps_needed = t
                break
        shown = f"{steps_needed}" if steps_needed else f"> {STEPS} (DNF)"
        print(f"{name:<10} {shown:>18}     ({p[0]:+.4f}, {p[1]:+.4f})")

    print(
        "\nSame valley, same start — different step counts to the finish."
        "\nNote each optimizer needed a DIFFERENT alpha to behave at all:"
        "\nmake them equal and SGD diverges or Adam crawls. Robustness to"
        "\nthe learning rate is Adam's real selling point."
    )
