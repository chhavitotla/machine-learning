# Decision Trees

## The Intuition

A decision tree plays twenty questions with your data. "Is humidity above 87%? Is temperature below 9°? Then nobody's playing tennis." Training is just deciding *which questions to ask, in which order*: at every node the algorithm (CART) tries every possible threshold on every feature, measures how much each candidate cut would unmix the classes, and greedily commits to the best one — then repeats inside each half. Geometrically, every question is an axis-aligned cut, so the tree carves the feature space into rectangles, each rectangle voting its majority class. No gradients, no learning rate, no assumption that anything is linear — and the finished model is literally a flowchart a human can read. The catch: a tree greedy enough to keep cutting will happily fence off every single noisy point into its own tiny rectangle, which is why depth control and pruning matter more here than in almost any other model.

## The Math

The impurity (Gini):

$$G = 1 - \sum_{k} p_k^2$$

- $p_k$ — the fraction of points in the node belonging to class $k$
- $G = 0$ — the node is pure (all one class); $G = 0.5$ — a 50/50 coin flip (two classes)

Gini is the probability that two points drawn at random from the node disagree on class — a measure of how "mixed" the node still is.

The split criterion (information gain):

$$\Delta G = G_{parent} - \frac{n_L}{n} G_L - \frac{n_R}{n} G_R$$

- $G_L, G_R$ — impurity of the left/right children created by the candidate cut
- $n_L, n_R$ — how many points fall on each side; children are weighted by size

A split is good if the children are much purer than the parent, with big children counting more than tiny ones.

The algorithm (CART, greedy recursion):

$$\text{split}^* = \arg\max_{f,\, t} \; \Delta G(f, t)$$

- $f$ — which feature to ask about
- $t$ — the threshold; only midpoints between consecutive sorted values need checking

Scan all $(f, t)$ pairs, take the best, recurse into both children; stop when a node is pure, hits max depth, or no split gains anything. Each split costs $O(d \cdot n \log n)$ — no calculus anywhere, just counting.

The prediction:

$$\hat{y}(x) = \text{majority class of the leaf that } x \text{ falls into}$$

Follow the questions from the root; land in a rectangle; predict its majority. (For probabilities, use the leaf's class proportions.)

Note what's *absent*: no loss surface, no gradient, no convexity. The greed is the weakness too — a split that looks mediocre now but enables a brilliant one later (XOR-style interactions) will never be chosen.

## Open the visualization

`open visualize.html in any browser`

## When to use this

- **Decision-making that must be auditable** — loan denials, medical triage, fraud escalation rules: the model *is* a flowchart, so compliance can read the exact path behind every decision.
- **Messy heterogeneous tabular data** — mixed numeric/categorical features, missing values, wildly different scales: trees need no normalization, no one-hot gymnastics, no distributional assumptions.
- **As the building block of ensembles** — a single tree is rarely the final model, but random forests and gradient boosting (XGBoost, LightGBM) — which still win most tabular competitions — are nothing but hundreds of these.

## What breaks it

- **Overfitting by default** — grown to full depth, a tree happily gives every noisy point its own rectangle and hits 100% training accuracy while memorizing nothing generalizable; an unpruned, unconstrained tree is one of the highest-variance models there is.
- **Instability** — remove ten random rows and the root split can change, which rewrites the *entire* tree below it; two near-identical datasets can produce wildly different flowcharts, which undermines the interpretability story if anyone re-trains.
- **Smooth diagonal boundaries** — a tree can only make axis-aligned cuts, so the boundary $x_1 + x_2 > 1$ gets approximated by an ugly staircase of dozens of splits where logistic regression needs one line; same problem for any rotated or smooth structure.

## 5 Interview Questions

**1. Conceptual — "Why do trees overfit so easily, and what are the standard defenses?"**
Direct answer: because the hypothesis space is enormous and the algorithm is greedy-local — every additional split strictly fits training data better, down to one-point-per-leaf memorization with zero training error and terrible variance. The defenses: pre-pruning (max depth, min samples per leaf/split, min gain) and post-pruning (grow full, then collapse subtrees that don't survive cross-validation, e.g. cost-complexity pruning). Reason: a tree's capacity grows exponentially with depth, so unconstrained, its variance dominates; constraints trade a bit of bias for a lot of variance. Likely follow-up: *"Pre- vs post-pruning trade-off?"* — pre-pruning is cheap but can stop too early (a weak split sometimes enables a strong one below it); post-pruning sees the whole tree before judging, at extra compute cost.

**2. Mathematical — "A node has 10 positives and 10 negatives. A split sends 8+/2− left and 2+/8− right. Compute the Gini gain."**
Direct answer: parent Gini = $1 - 0.5^2 - 0.5^2 = 0.5$; each child has Gini $1 - 0.8^2 - 0.2^2 = 0.32$; weighted children = $0.5 \cdot 0.32 + 0.5 \cdot 0.32 = 0.32$; gain = $0.5 - 0.32 = 0.18$. Reason: it's pure proportion arithmetic — this is exactly the computation CART runs for every candidate threshold. Likely follow-up: *"Gini vs entropy?"* — entropy uses $-\sum p_k \log p_k$ (this split: $1.0 \to 0.722$); they pick the same split ~98% of the time, and Gini is preferred in practice because it skips the logarithm.

**3. Practical — "Your tree gets 99% training accuracy and 71% test accuracy. Walk me through your fix."**
Direct answer: classic variance gap — first check depth and leaf sizes (you'll likely find depth 20+ and single-sample leaves), then constrain: max_depth ≈ 3–8, min_samples_leaf ≈ 1–5% of data, tuned by cross-validation; if accuracy ceiling matters more than interpretability, switch to a random forest, which fixes variance by design. Reason: the 28-point gap is variance, not bias, so adding features or data transformations won't help — shrinking the hypothesis space (or averaging many trees) will. Likely follow-up: *"Train accuracy dropped to 85% after pruning — is that bad?"* — no, that's the point: you traded training fit for the test accuracy moving up; only the held-out number matters.

**4. Gotcha — "My tree's feature importances say `user_id` is the most important feature. Ship it?"**
Direct answer: absolutely not — high-cardinality features like IDs give the splitter thousands of candidate thresholds, so it can carve training data nearly perfectly on them by pure chance; the importance is an artifact of memorization, and the model will be useless on new users. Reason: impurity-based importance is biased toward features with many possible split points (continuous or high-cardinality), independent of real signal — a known flaw of Gini importance. Likely follow-up: *"How do you measure importance honestly?"* — permutation importance on a held-out set (shuffle one feature, measure the accuracy drop), which would expose `user_id` immediately, or drop the leaky feature entirely.

**5. System design — "Design the rules engine that decides whether to step-up-authenticate a login. Why might a decision tree beat a neural net here?"**
Direct answer: use a depth-constrained tree (or small tree ensemble) over features like device novelty, geo-velocity, failed-attempt counts, and time-of-day: it serves in microseconds with a handful of comparisons, exports directly to human-readable rules security analysts can review and override, and handles the mixed categorical/numeric features without preprocessing. Reason: this domain demands explainability (customers ask "why was I challenged?"), sub-millisecond latency at every login, and easy auditing of policy changes — all native strengths of trees, while a neural net's marginal accuracy wouldn't survive the ops and compliance cost. Likely follow-up: *"Where does this design strain?"* — adversaries drift; a static tree goes stale, so you retrain on rolling windows — and because small data changes can rewrite the tree, you diff the exported rules between versions and require human sign-off before deploying, keeping the auditability that justified the tree in the first place.
