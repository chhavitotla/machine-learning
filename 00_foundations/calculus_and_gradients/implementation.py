"""Calculus for ML from scratch — pure standard library, zero dependencies.

Run it:  python3 implementation.py

The two ideas that power all of training:
    derivative:        f'(x) = limit of (f(x+h) - f(x-h)) / 2h as h -> 0
                       "which way is downhill, and how steep?"
    gradient descent:  x -= alpha * f'(x), repeated
                       "take a small step downhill, forever"
"""


def f(x: float) -> float:
    """The function we'll study: f(x) = x^4 - 3x^2 + x + 4.
    A quartic with a local minimum AND a global minimum — so descent
    can genuinely land in the wrong valley depending on where it starts."""
    return x**4 - 3 * x**2 + x + 4


def df_analytic(x: float) -> float:
    """The exact derivative, by the power rule: d/dx of x^n is n*x^(n-1)."""
    return 4 * x**3 - 6 * x + 1


def df_numeric(g, x: float, h: float = 1e-5) -> float:
    """Central difference: slope of the secant through x-h and x+h.

    Better than the one-sided (f(x+h)-f(x))/h because the symmetric
    secant's error shrinks like h^2 instead of h — the linear error
    terms on each side cancel.
    """
    return (g(x + h) - g(x - h)) / (2 * h)


def gradient_descent(g, dg, x0: float, alpha: float = 0.05, steps: int = 60) -> list[float]:
    """Walk downhill: at each point, the derivative says which way is up,
    so step the other way, scaled by the learning rate alpha."""
    xs = [x0]
    x = x0
    for _ in range(steps):
        x = x - alpha * dg(x)
        xs.append(x)
    return xs


if __name__ == "__main__":
    # 1. Numerical vs analytic derivative: they should agree to ~10 digits.
    print("— central difference vs the power rule, f(x) = x^4 - 3x^2 + x + 4 —")
    print(f"  {'x':>6}  {'numeric df/dx':>16}  {'analytic df/dx':>16}  {'|error|':>10}")
    for x in (-2.0, -0.5, 0.0, 0.7, 1.5):
        num = df_numeric(f, x)
        ana = df_analytic(x)
        print(f"  {x:>6.2f}  {num:>16.8f}  {ana:>16.8f}  {abs(num - ana):>10.2e}")

    # 2. Shrinking h: watch the secant slope converge to the tangent slope.
    print("\n— secant → tangent at x = 1.5 as h shrinks —")
    target = df_analytic(1.5)
    for h in (1.0, 0.1, 0.01, 0.001):
        sec = (f(1.5 + h) - f(1.5)) / h          # one-sided secant, like the demo
        print(f"  h = {h:<7} secant slope = {sec:>10.6f}   (tangent = {target:.6f})")

    # 3. Gradient descent from two different starts: two different valleys.
    print("\n— gradient descent trace, alpha = 0.05 —")
    for x0 in (2.0, -0.2):
        trace = gradient_descent(f, df_analytic, x0)
        shown = trace[:5] + ["..."] + [trace[-1]]
        path = " → ".join(t if isinstance(t, str) else f"{t:.4f}" for t in shown)
        print(f"  start x0 = {x0:+.1f}:  {path}")
        print(f"     landed at x = {trace[-1]:+.6f},  f(x) = {f(trace[-1]):.6f},"
              f"  f'(x) = {df_analytic(trace[-1]):+.2e}")
    print("\n  Same function, same algorithm, different start → different valley.")
    print("  The right basin (global min) is deeper; descent only sees local slope.")
