"""Probability for ML from scratch — pure standard library, zero dependencies.

Run it:  python3 implementation.py

Three pillars:
    Law of Large Numbers:   averages of samples converge to the true mean
    Central Limit Theorem:  averages of ANYTHING pile up into a bell curve
    Bayes' rule:            P(A|B) = P(B|A) P(A) / P(B) — evidence updates belief
"""

import math
import random


def coin_flips(n: int, p: float) -> list[int]:
    """n flips of a coin that lands heads with probability p."""
    return [1 if random.random() < p else 0 for _ in range(n)]


def running_means(flips: list[int]) -> list[float]:
    """Mean of the first k flips, for each k — the LLN in motion."""
    means, total = [], 0
    for k, x in enumerate(flips, start=1):
        total += x
        means.append(total / k)
    return means


def ascii_histogram(values: list[float], bins: int = 15, width: int = 40) -> str:
    """Bucket the values and draw each bucket as a bar of #'s."""
    lo, hi = min(values), max(values)
    counts = [0] * bins
    for v in values:
        i = min(int((v - lo) / (hi - lo + 1e-12) * bins), bins - 1)
        counts[i] += 1
    peak = max(counts)
    lines = []
    for i, c in enumerate(counts):
        center = lo + (i + 0.5) * (hi - lo) / bins
        bar = "#" * round(c / peak * width)
        lines.append(f"  {center:6.3f} | {bar} {c}")
    return "\n".join(lines)


def bayes_posterior(prevalence: float, sensitivity: float, specificity: float) -> float:
    """P(disease | positive test), exactly, via Bayes' rule.

    P(D|+) = P(+|D) P(D) / P(+)
    where P(+) counts BOTH ways to test positive:
        true positives:  P(+|D) P(D)        = sensitivity * prevalence
        false positives: P(+|¬D) P(¬D)      = (1 - specificity) * (1 - prevalence)
    Base-rate neglect is forgetting that the second line can dwarf the first
    when the disease is rare.
    """
    true_pos = sensitivity * prevalence
    false_pos = (1 - specificity) * (1 - prevalence)
    return true_pos / (true_pos + false_pos)


if __name__ == "__main__":
    random.seed(7)

    # 1. Law of Large Numbers: the running mean homes in on p.
    p = 0.7
    flips = coin_flips(100_000, p)
    means = running_means(flips)
    print(f"— law of large numbers: coin with p = {p} —")
    for k in (10, 100, 1_000, 10_000, 100_000):
        print(f"  after {k:>7,} flips:  running mean = {means[k - 1]:.4f}"
              f"   (off by {abs(means[k - 1] - p):.4f})")

    # 2. Central Limit Theorem: means of small samples form a bell curve,
    #    even though a single flip is as un-bell-shaped as data gets.
    n_per_sample, n_samples = 30, 5_000
    sample_means = [sum(coin_flips(n_per_sample, p)) / n_per_sample
                    for _ in range(n_samples)]
    print(f"\n— central limit theorem: {n_samples:,} means of {n_per_sample}-flip samples —")
    print(ascii_histogram(sample_means))
    mu = sum(sample_means) / n_samples
    sd = math.sqrt(sum((m - mu) ** 2 for m in sample_means) / n_samples)
    print(f"  empirical mean {mu:.4f} (theory: p = {p})")
    print(f"  empirical sd   {sd:.4f} (theory: sqrt(p(1-p)/n) = "
          f"{math.sqrt(p * (1 - p) / n_per_sample):.4f})")

    # 3. Bayes: a 99%-accurate test for a 1-in-1000 disease.
    prev, sens, spec = 0.001, 0.99, 0.99
    exact = bayes_posterior(prev, sens, spec)
    print(f"\n— bayes' rule: prevalence {prev:.1%}, sensitivity {sens:.0%}, "
          f"specificity {spec:.0%} —")
    print(f"  exact P(disease | positive) = {exact:.4f}")

    # simulate a million people and count, to confirm the formula
    N = 1_000_000
    sick_pos = healthy_pos = 0
    for _ in range(N):
        sick = random.random() < prev
        positive = random.random() < (sens if sick else 1 - spec)
        if positive:
            if sick: sick_pos += 1
            else:    healthy_pos += 1
    sim = sick_pos / (sick_pos + healthy_pos)
    print(f"  simulated ({N:,} people)   = {sim:.4f}"
          f"   ({sick_pos:,} true positives vs {healthy_pos:,} false positives)")
    print("  A positive result still means you're probably healthy —")
    print("  the rare disease's false positives outnumber its true ones.")
