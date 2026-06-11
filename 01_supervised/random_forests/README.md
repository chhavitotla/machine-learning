# Random Forests

## The Intuition

One decision tree is a nervous expert: brilliant on the data it saw, but change ten rows and it rewrites its entire flowchart. A random forest hires a hundred of these nervous experts, deliberately makes each one *more* erratic, and averages them — and the average is calm. Two tricks manufacture the disagreement that makes averaging work. First, each tree trains on a bootstrap sample (draw n rows *with replacement*, so every tree sees a different remix of the data, missing about a third of it). Second, at every split a tree may only consider a random handful of features — so even trees that agree the best question is "income?" are often forbidden from asking it and must find another path. Each tree overfits its own remix in its own direction, but the *errors* point different ways and cancel in the vote, while the signal — which every tree sees — survives. You trade the single readable flowchart for a model that's hard to beat on tabular data, and the rows each tree never saw act as a free built-in test set.

## The Math

Bagging (bootstrap aggregating):

$$D_b = \{(x_{i_1}, y_{i_1}), \ldots, (x_{i_n}, y_{i_n})\}, \qquad i_j \sim \text{Uniform}\{1, \ldots, n\}$$

- $D_b$ — the training set for tree $b$: $n$ indices drawn with replacement
- $n$ — the original dataset size; each $D_b$ misses $\approx 36.8\%$ of rows, since $P(\text{row excluded}) = (1 - \tfrac{1}{n})^n \to e^{-1}$

Every tree gets its own resampled remix of the data, so no two trees train on quite the same evidence.

The prediction (majority vote / averaging):

$$\hat{y}(x) = \text{mode}\{T_1(x), \ldots, T_B(x)\} \qquad \text{or} \qquad \hat{y}(x) = \frac{1}{B}\sum_{b=1}^{B} T_b(x)$$

- $T_b(x)$ — the prediction of tree $b$ at input $x$
- $B$ — the number of trees; mode for classification, mean for regression

Each tree votes and the forest goes with the crowd — averaging $B$ models with variance $\sigma^2$ and pairwise correlation $\rho$ gives variance $\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$, so more trees and *less correlated* trees both shrink it.

Feature subsampling (the "random" in random forest):

$$\text{at each split: consider only } F \subset \{1, \ldots, d\}, \quad |F| = m \approx \sqrt{d}$$

- $d$ — total number of features; $m$ — the random subset size per split ($\sqrt{d}$ for classification, $d/3$ for regression, by convention)

Hiding features at each split forces trees to disagree — it attacks the $\rho$ term bagging alone can't touch, since otherwise every tree puts the same dominant feature at the root.

Out-of-bag (OOB) error:

$$\text{OOB} = \frac{1}{n}\sum_{i=1}^{n} \mathbb{1}\left[ \text{mode}\{T_b(x_i) : i \notin D_b\} \neq y_i \right]$$

- $\{T_b : i \notin D_b\}$ — only trees that *never trained on* point $i$ (about $0.368 B$ of them) get to vote on it

Score every point using only the trees that never saw it: an honest generalization estimate for free — no held-out set, no cross-validation loop.

## Open the visualization

`open visualize.html in any browser` — grow trees one at a time over noisy 2D data (sliders for forest size and max depth) and watch faint individual-tree boundaries chase flipped-label noise while the bold ensemble boundary stays smooth, with an OOB-accuracy chart filling in as the forest grows.

## When to use this

- **Tabular prediction with minimal tuning** — churn, credit risk, sensor-failure prediction: a forest with default hyperparameters is routinely within a few points of a heavily-tuned gradient-boosted model — the standard "strong baseline in an afternoon."
- **Small or messy datasets where overfitting is the enemy** — a few thousand rows of clinical or operational data: a single tree memorizes, a deep net starves, but averaging hundreds of decorrelated trees is one of the safest variance reducers there is.
- **Quick honest signal triage** — OOB error plus permutation importance tell you, with no validation pipeline, which of your 200 candidate features carry real signal and how predictable the target is at all.

## What breaks it

- **Latency- and memory-critical serving** — 500 trees of depth 20 means megabytes of model and hundreds of pointer-chasing traversals per prediction; on a tight budget you'll end up distilling it or shipping a smaller boosted model.
- **Extrapolation in regression** — every prediction is an average of training-leaf values, so a forest trained on house prices up to \$800K can *never* predict \$900K; on trending time series it silently flatlines at the edge of what it has seen.
- **Correlated trees from a dominant feature or leakage** — if one feature (or a leaked target proxy) is overwhelmingly strong, every tree builds around it despite subsampling; $\rho \to 1$, averaging cancels nothing, and you get one overfit tree at 500× the cost — with an OOB score that looks great until the leak vanishes in production.

## 5 Interview Questions

**1. Conceptual — "Bagging already gives every tree different data. Why also restrict features at each split?"**
Direct answer: because bootstrap samples overlap ~63%, trees trained on them stay highly correlated — they nearly all put the same strong feature at the root — and averaging correlated models barely reduces variance; restricting each split to $m \approx \sqrt{d}$ random features forces structurally different trees. Reason: ensemble variance is $\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$; adding trees only kills the second term, while feature subsampling is the only lever on $\rho$. Likely follow-up: *"What happens at $m = d$ and $m = 1$?"* — $m = d$ is plain bagging (correlated, lower-bias trees); $m = 1$ gives near-random splits (decorrelated, high-bias trees); $\sqrt{d}$ is the empirical sweet spot.

**2. Mathematical — "What fraction of the training data does each tree never see, and why does that matter?"**
Direct answer: $(1 - \frac{1}{n})^n \to e^{-1} \approx 36.8\%$ — each of the $n$ independent draws misses a given row with probability $1 - \frac{1}{n}$. Reason: that excluded third is what makes OOB evaluation valid — every point has, in expectation, $0.368B$ trees that are honest judges of it, so the forest carries its own test set. Likely follow-up: *"Is OOB error as good as cross-validation?"* — almost: it's nearly free and unbiased, but each OOB vote uses only ~37% of the forest, so it's slightly pessimistic for small $B$; at a few hundred trees the difference is negligible.

**3. Practical — "Your random forest's training accuracy is 100% and test accuracy is 76%. What do you actually tune?"**
Direct answer: don't panic about the 100% — trees are grown deep on purpose, so perfect training fit is normal for forests; tune what controls *tree correlation and depth*: `max_features` down, `min_samples_leaf` up (to ~1% of data), `max_depth` capped, `n_estimators` at a few hundred. Reason: forest generalization is governed by tree strength vs. correlation, not training error — a 24-point gap usually means too few trees, leaked features, or far too few samples per leaf for the noise level. Likely follow-up: *"Does adding more trees ever overfit?"* — no, test error converges monotonically (in expectation) as $B \to \infty$; more trees only cost compute, so $B$ is set by budget, not validation.

**4. Gotcha — "My forest's feature importance ranks a random noise column above several real features. How?"**
Direct answer: that's the known bias of impurity-based (Gini) importance — a continuous noise column offers thousands of candidate thresholds, so across deep trees it accumulates many small, chance impurity gains, outscoring, say, a binary feature with genuine signal. Reason: Gini importance sums *training-set* gains, which reward split opportunity, not generalization — and bagging doesn't average the bias away because every tree shares it. Likely follow-up: *"What do you use instead?"* — permutation importance on OOB or held-out data (shuffle a column, measure the score drop), which prices a feature by its contribution to *honest* predictions; the noise column drops to zero immediately.

**5. System design — "Design the model layer for a bank's loan default predictor: 80 mixed features, 200K rows, regulators want documentation, retraining is quarterly."**
Direct answer: a random forest — 300–500 trees, `min_samples_leaf` tuned by OOB, permutation importances and partial-dependence plots as the regulator-facing documentation, and OOB error tracked across retrains as a drift alarm. Reason: 200K tabular rows with mixed types is squarely forest territory — near-boosting accuracy with far less tuning to document, no preprocessing pipeline to validate, robust to messy financial data; the quarterly cadence makes serving latency a non-issue. Likely follow-up: *"The regulator demands reason codes per denial — now what?"* — per-prediction explanations via TreeSHAP (exact Shapley values for tree ensembles, fast enough to serve), giving a signed per-feature contribution for every decision — auditable reason codes without giving up the ensemble's accuracy.
