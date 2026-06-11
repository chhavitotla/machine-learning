# ROC Curve & AUC

## The Intuition

A confusion matrix freezes one threshold; the ROC curve refuses to choose. Imagine sliding the threshold from impossibly strict (flag nothing) to absurdly lenient (flag everything), and at every stop recording two numbers: what fraction of real positives you're catching (TPR) and what fraction of negatives you're falsely flagging (FPR). Plot those pairs and you get the ROC curve — a portrait of the model at *every* operating point at once. A great model catches most positives before admitting any negatives, so its curve hugs the top-left corner; a useless one admits both in lockstep and traces the diagonal. AUC, the area under the curve, compresses the portrait to one number with a gorgeous interpretation: it is *exactly* the probability that a randomly chosen positive scores higher than a randomly chosen negative. AUC grades the model's ranking ability, threshold-free — which is both its power and its blind spot, because it also ignores class imbalance in a way that can flatter a model badly.

## The Math

The two rates, at threshold $t$:

$$\text{TPR}(t) = \frac{TP}{TP + FN} \qquad \text{FPR}(t) = \frac{FP}{FP + TN}$$

- $\text{TPR}$ — true positive rate (recall): conditions only on the *positive* class
- $\text{FPR}$ — false positive rate: conditions only on the *negative* class

Because each rate stays inside its own class, rescaling the class sizes changes neither — this is why ROC is invariant to class imbalance.

The ROC curve is the parametric path swept by the threshold:

$$\text{ROC} = \{(\text{FPR}(t), \text{TPR}(t)) : t \in (-\infty, \infty)\}$$

AUC by the trapezoid rule over the curve's points:

$$\text{AUC} = \sum_{k} (\text{FPR}_{k+1} - \text{FPR}_k) \cdot \frac{\text{TPR}_k + \text{TPR}_{k+1}}{2}$$

The probabilistic identity (Mann–Whitney):

$$\text{AUC} = P(s^+ > s^-) + \tfrac{1}{2} P(s^+ = s^-)$$

- $s^+$ — the score of a uniformly random positive
- $s^-$ — the score of a uniformly random negative

The geometric area and the pairwise-ranking probability are the *same number* — `implementation.py` verifies this to machine precision by comparing the trapezoid sum against brute-force counting over all positive–negative pairs.

The PR alternative replaces FPR with precision:

$$\text{precision}(t) = \frac{TP}{TP + FP}$$

Precision mixes both classes in its denominator, so it *does* feel imbalance — a random ranker's PR curve sits at the prevalence $\pi = P/(P+N)$, not at a fixed diagonal. When positives are rare, PR is the honest plot.

## Open the visualization

`open visualize.html in any browser`

## When to use this

- **Comparing models before a threshold is chosen** — AUC ranks candidate models on pure discrimination ability, independent of any operating point, so it's the right scoreboard during model selection and for tracking score quality across retrains.
- **When the operating threshold will vary or is unknown** — credit scoring, ad ranking, triage queues: the consumer of the scores picks their own cutoff, so you must certify the whole curve, not one point on it.
- **Communicating a model's headroom** — "AUC 0.92" tells you a good threshold *exists*; the curve shows exactly what recall is purchasable at each false-alarm budget, which is how you negotiate SLAs with stakeholders.

## What breaks it

- **Heavy class imbalance** — at 0.1% prevalence, an FPR of "only" 1% means ten false alarms for every true positive; ROC-AUC can read 0.95 while precision is 9%. The fraud team will discover this in production. Use PR curves when positives are rare.
- **Caring about one region of the curve** — AUC integrates over all thresholds, including absurd ones you'd never deploy; a model can win on AUC by dominating in the useless high-FPR region while losing where you actually operate. Use partial AUC or the metric at your operating point.
- **Treating AUC as calibration** — AUC only sees the *ordering* of scores; a model outputting 0.51 for every positive and 0.49 for every negative has AUC 1.0 and completely useless probabilities. If anyone multiplies your scores by dollar amounts, you need calibration (reliability curves, Brier score), not AUC.

## 5 Interview Questions

**1. Conceptual — "What does AUC = 0.85 actually mean?"**
Direct answer: pick a random positive and a random negative; 85% of the time the positive gets the higher score — AUC is exactly that pairwise ranking probability. Reason: the Mann–Whitney identity equates the area under the ROC curve with $P(s^+ > s^-)$; it's a statement about ranking quality, with no threshold and no class balance in it. Likely follow-up: *"What does AUC = 0.5 and AUC = 0.2 mean?"* — 0.5 is ranking by coin flip (the diagonal); 0.2 means the model ranks *backwards* better than chance — flip its sign and you have AUC 0.8, which usually signals a label or sign bug.

**2. Mathematical — "Why is the ROC curve monotonically non-decreasing?"**
Direct answer: as the threshold decreases, the set of flagged samples only grows, so TP and FP can only increase or stay flat — both coordinates move monotonically, and the curve can only step up or right. Reason: each distinct score, as it crosses the threshold, adds either one positive (step up by $1/P$) or one negative (step right by $1/N$); the curve is literally a staircase built from the sorted scores. Likely follow-up: *"So how do you compute the curve efficiently?"* — sort by score descending once ($O(n \log n)$) and accumulate the staircase in one pass; never re-count the confusion matrix per threshold.

**3. Practical — "Your fraud model has ROC-AUC 0.96 but the ops team says most alerts are junk. Explain."**
Direct answer: prevalence — with, say, 0.2% fraud, even a 2% FPR produces ~10 false alerts per true fraud, so precision is ~10% despite stellar AUC. Reason: FPR's denominator is the enormous negative class, so a "small" FPR is a huge absolute number of false alarms; AUC never sees this because it's imbalance-invariant by construction. Likely follow-up: *"What do you report instead?"* — PR-AUC or precision at the operating recall, plus expected alerts/day at the chosen threshold — numbers denominated in ops workload, not in rates.

**4. Gotcha — "Model A has higher AUC than model B. Is A the better model to deploy?"**
Direct answer: not necessarily — AUC averages over every threshold, but you deploy at one; B's curve may dominate A's exactly in the low-FPR region where you operate. Reason: ROC curves can cross, and a model can buy whole-curve area in operating regions you'll never use; deployment quality is a point (or short arc) on the curve, not the integral. Likely follow-up: *"How would you compare them properly?"* — plot both curves and compare TPR at your FPR budget (or partial AUC over the feasible FPR range), on data resembling production prevalence.

**5. System design — "Design the evaluation pipeline for a content-moderation classifier that human reviewers sit behind. ROC or PR?"**
Direct answer: PR — violating content is rare and the binding constraint is reviewer throughput, so the operative question is "of what we send to humans, how much is real?" (precision) at the recall policy requires; track PR-AUC for model selection and precision@operating-recall as the deploy gate. Reason: ROC's imbalance-blindness hides the review-queue cost: an FPR change from 1% → 2% looks negligible but doubles the queue when 99% of content is benign. Likely follow-up: *"Prevalence shifts week to week — what happens to your metrics?"* — recall and FPR are stable per-class, but precision moves with prevalence even with a frozen model; monitor prevalence separately and re-estimate the threshold so reviewer load stays within budget.
