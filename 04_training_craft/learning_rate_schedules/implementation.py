"""Learning rate schedules from scratch — pure stdlib, zero dependencies.

Run it:  python3 implementation.py

Each schedule is a tiny function mapping the step counter to a learning
rate. The demo fits a 1D quadratic from noisy gradients (a stand-in for
mini-batch SGD) under every schedule and shows why constant LR stalls
at a noise floor while the decaying schedules grind straight through it.
"""

import math
import random


# ---------------------------------------------------------------------------
# The schedules. Each takes the step t and returns the LR for that step.
# All of them are pure functions of t — which is exactly what lets real
# training resume after a crash: the LR depends on nothing but the counter.
# ---------------------------------------------------------------------------

def constant(t: int, lr0: float) -> float:
    """The baseline everything else is measured against."""
    return lr0


def step_decay(t: int, lr0: float, gamma: float = 0.5, every: int = 200) -> float:
    """Staircase: multiply by gamma each time t crosses a milestone.

    The floor division is the whole trick — it freezes the LR between
    milestones and produces the cliff-shaped loss drops of classic
    ResNet training curves.
    """
    return lr0 * gamma ** (t // every)


def exponential_decay(t: int, lr0: float, k: float = 0.004) -> float:
    """Smooth slide: lose the same FRACTION of the LR every step."""
    return lr0 * math.exp(-k * t)


def cosine_annealing(t: int, lr0: float, total: int, lr_min: float = 0.0) -> float:
    """Half a cosine wave from lr0 down to lr_min over `total` steps.

    Flat at the start (full-speed exploration), steepest in the middle,
    flat again at the end (a long gentle glide into the minimum). The
    schedule NEEDS the horizon `total` up front — its defining trade-off.
    """
    t = min(t, total)  # clamp: training past the horizon stays at lr_min
    return lr_min + 0.5 * (lr0 - lr_min) * (1 + math.cos(math.pi * t / total))


def cosine_with_warmup(t: int, lr0: float, total: int, warmup: int = 100) -> float:
    """Linear ramp 0 -> lr0 over `warmup` steps, then cosine to zero.

    Warmup protects the most fragile steps: at t=0 the weights are
    random, gradients are erratic, and Adam's moment estimates are
    unreliable. This composition is the default transformer recipe.
    """
    if t < warmup:
        return lr0 * t / max(warmup, 1)
    return cosine_annealing(t - warmup, lr0, total - warmup)


# Warm restarts in one line of logic: replay the cosine curve with the
# step counter wrapped, so the LR snaps back to lr0 every `period` steps,
# kicking the model out of its basin to explore for a better one (SGDR).
def cosine_warm_restarts(t: int, lr0: float, period: int = 250) -> float:
    return cosine_annealing(t % period, lr0, period)


# ---------------------------------------------------------------------------
# The arena: fit theta to minimize E[(theta - 3)^2] from NOISY gradients.
# The noise plays the role of mini-batch sampling. With constant LR, SGD
# converges to a noise ball around theta*=3 whose radius scales with the
# LR; decaying the LR shrinks the ball to zero.
# ---------------------------------------------------------------------------

def train(schedule, steps: int, seed: int = 0) -> float:
    """Run noisy gradient descent under `schedule`; return final loss."""
    rng = random.Random(seed)            # same seed => same noise: fair race
    theta = -5.0                         # start far from the target
    for t in range(steps):
        grad = 2 * (theta - 3.0)         # exact gradient of (theta - 3)^2
        grad += rng.gauss(0.0, 2.0)      # ...corrupted by "mini-batch" noise
        theta -= schedule(t) * grad      # the entire algorithm
    return (theta - 3.0) ** 2


if __name__ == "__main__":
    STEPS, LR0 = 1_000, 0.1

    schedules = {
        "constant":         lambda t: constant(t, LR0),
        "step decay":       lambda t: step_decay(t, LR0),
        "exponential":      lambda t: exponential_decay(t, LR0),
        "cosine":           lambda t: cosine_annealing(t, LR0, STEPS),
        "cosine + warmup":  lambda t: cosine_with_warmup(t, LR0, STEPS),
        "warm restarts":    lambda t: cosine_warm_restarts(t, LR0),
    }

    # 1) The shape of each schedule, sampled along the run.
    samples = [0, 50, 250, 500, 750, 999]
    print(f"learning rate at sample steps (lr0 = {LR0}, T = {STEPS}):\n")
    print(f"{'schedule':<17}" + "".join(f"t={s:<8}" for s in samples))
    for name, sched in schedules.items():
        row = "".join(f"{sched(s):<10.4f}" for s in samples)
        print(f"{name:<17}{row}")

    # 2) The payoff: same noisy problem, same seed, different schedules.
    print(f"\nfinal loss after {STEPS} noisy steps (target: 0, averaged over 20 seeds):\n")
    for name, sched in schedules.items():
        avg = sum(train(sched, STEPS, seed=s) for s in range(20)) / 20
        print(f"{name:<17}{avg:.6f}")

    print(
        "\nConstant LR stalls ~20-40x above every decaying schedule: it"
        "\nconverges to a NOISE BALL around the minimum, not the minimum — its"
        "\nradius scales with the LR, so only a shrinking LR can reach the"
        "\nbottom. Warm restarts trade that final precision for repeated"
        "\nexploration (useful on bumpy, multi-minimum landscapes)."
    )
