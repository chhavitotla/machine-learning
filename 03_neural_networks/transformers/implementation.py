"""Scaled dot-product & multi-head attention from scratch — pure stdlib.

Run it:  python3 implementation.py

Implements, with plain lists and math:

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
    MultiHead(X) = Concat(head_1 ... head_h) W^O

and demonstrates on a toy sentence that a refers-back query feature makes
"it" attend to "animal" — coreference falling out of dot products.
"""

import math

# ---------------------------------------------------------------------------
# Minimal matrix helpers (rows are vectors)
# ---------------------------------------------------------------------------

def matmul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    return [
        [sum(a * b for a, b in zip(row, col)) for col in zip(*B)]
        for row in A
    ]


def transpose(A: list[list[float]]) -> list[list[float]]:
    return [list(col) for col in zip(*A)]


def softmax(row: list[float]) -> list[float]:
    """Softmax with the max subtracted first — without this, exp() overflows
    on large scores. Same output, stable arithmetic."""
    m = max(row)
    exps = [math.exp(s - m) for s in row]
    total = sum(exps)
    return [e / total for e in exps]


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

def scaled_dot_product_attention(
    Q: list[list[float]],
    K: list[list[float]],
    V: list[list[float]],
) -> tuple[list[list[float]], list[list[float]]]:
    """The core equation. Returns (output, attention_weights).

    Q: (n, d_k)  one query per token  — "what am I looking for?"
    K: (n, d_k)  one key per token    — "what do I advertise?"
    V: (n, d_v)  one value per token  — "what do I contribute if chosen?"
    """
    d_k = len(Q[0])
    # Scores: every query against every key. This n×n matrix is the O(n²)
    # that makes long contexts expensive.
    scores = matmul(Q, transpose(K))

    # Divide by sqrt(d_k): dot products of d_k-dim vectors have variance
    # ~d_k, and softmax over large scores saturates to one-hot, killing
    # gradients. Scaling keeps the variance ~1.
    scaled = [[s / math.sqrt(d_k) for s in row] for row in scores]

    # Each row becomes a probability distribution: a fixed attention
    # budget the token spends across the sequence.
    weights = [softmax(row) for row in scaled]

    # Output: each token's new representation is a weighted blend of
    # everyone's values.
    return matmul(weights, V), weights


def multi_head_attention(
    X: list[list[float]],
    heads: list[dict],
    W_O: list[list[float]],
) -> list[list[float]]:
    """Run h independent attentions and mix the results.

    Each head has its own W_Q, W_K, W_V — that independence is the point:
    one head can track syntax while another tracks coreference.
    """
    head_outputs = []
    for h in heads:
        Q = matmul(X, h["W_Q"])
        K = matmul(X, h["W_K"])
        V = matmul(X, h["W_V"])
        out, _ = scaled_dot_product_attention(Q, K, V)
        head_outputs.append(out)

    # Concat along the feature axis, then project back with W_O.
    concat = [sum((out[i] for out in head_outputs), []) for i in range(len(X))]
    return matmul(concat, W_O)


# ---------------------------------------------------------------------------
# Demo: coreference from dot products
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tokens = ["The", "animal", "didn't", "cross", "the", "street",
              "because", "it", "was", "too", "tired"]

    # Hand-crafted embeddings so the demo is inspectable. Feature meaning:
    # [is-noun, is-animate, is-verb, is-function-word, refers-back]
    X = [
        [0.0, 0.0, 0.0, 1.0, 0.0],   # The
        [1.0, 1.0, 0.0, 0.0, 0.0],   # animal
        [0.0, 0.0, 1.0, 1.0, 0.0],   # didn't
        [0.0, 0.0, 1.0, 0.0, 0.0],   # cross
        [0.0, 0.0, 0.0, 1.0, 0.0],   # the
        [1.0, 0.0, 0.0, 0.0, 0.0],   # street
        [0.0, 0.0, 0.0, 1.0, 0.0],   # because
        [0.3, 0.3, 0.0, 1.0, 1.0],   # it — weakly nominal, strongly seeking
        [0.0, 0.0, 1.0, 1.0, 0.0],   # was
        [0.0, 0.0, 0.0, 0.0, 0.0],   # too
        [0.0, 0.0, 0.0, 0.0, 0.0],   # tired
    ]

    def diag(gains: list[float]) -> list[list[float]]:
        """Diagonal projection — each feature simply amplified or muted.
        Real W_Q/W_K are dense and learned; diagonal keeps this readable."""
        return [
            [gains[i] if i == j else 0.0 for j in range(len(gains))]
            for i in range(len(gains))
        ]

    # A "coreference head": queries seek [noun, animate, refers-back];
    # keys advertise [noun, animate] but NOT refers-back — a pronoun hunts
    # for an antecedent without advertising itself as one. Q ≠ K projections
    # are exactly what makes that asymmetry possible.
    W_Q = diag([2.0, 5.0, 0.6, 0.4, 8.0])
    W_K = diag([2.0, 3.0, 0.3, 0.2, 0.0])
    W_V = diag([1.0, 1.0, 1.0, 1.0, 1.0])

    Q = matmul(X, W_Q)
    K = matmul(X, W_K)
    V = matmul(X, W_V)
    _, weights = scaled_dot_product_attention(Q, K, V)

    it = tokens.index("it")
    print(f'Where "{tokens[it]}" sends its attention:\n')
    ranked = sorted(zip(tokens, weights[it]), key=lambda p: -p[1])
    for tok, w in ranked[:5]:
        print(f"  {tok:10s} {w:6.1%}  {'#' * int(w * 40)}")

    print("\n'it' resolves to 'animal' — no rules, no parser, just")
    print("softmax(q.k/sqrt(d_k)) over feature vectors. Training learns the")
    print("projections; the mechanism is what you just ran.")
