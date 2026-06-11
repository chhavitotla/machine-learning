# Confusion Matrix (Precision, Recall, F1, Accuracy)

## The Intuition

A classifier doesn't really say "spam" or "not spam" — it emits a *score*, and you pick a threshold that turns scores into decisions. Once you decide, there are exactly four ways reality can respond: you flagged it and it was real (true positive), you flagged it and it wasn't (false positive), you passed on it correctly (true negative), or you passed on something real (false negative). The confusion matrix is just an honest tally of those four outcomes — and every classification metric you'll ever be asked about is a ratio of cells in that 2×2 table. The deep point: the four counts are not independent. Slide the threshold down and false negatives convert into true positives *and* true negatives convert into false positives, simultaneously. You never improve recall for free; you pay for it in precision. Choosing a threshold is choosing which mistake you'd rather make — a product decision wearing a math costume.

## The Math

The four counts at threshold $t$ (predict positive when score $\geq t$):

$$\text{TP}, \quad \text{FP}, \quad \text{TN}, \quad \text{FN}$$

- $\text{TP}$ — real positives you caught
- $\text{FP}$ — false alarms (negatives you flagged)
- $\text{TN}$ — negatives you correctly ignored
- $\text{FN}$ — real positives you missed

Precision — when you flag something, how often are you right:

$$\text{precision} = \frac{TP}{TP + FP}$$

Recall (sensitivity, TPR) — of all real positives, how many you caught:

$$\text{recall} = \frac{TP}{TP + FN}$$

F1 — the harmonic mean of the two, which punishes whichever is worse:

$$F_1 = \frac{2 \cdot \text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}$$

The harmonic (not arithmetic) mean matters: precision 1.0 with recall 0.01 gives $F_1 \approx 0.02$, not 0.5. You cannot game F1 by maxing one side.

Accuracy — fraction of all decisions that were correct:

$$\text{accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

Under class imbalance this is the liar of the family: with 1% positives, the constant model "always negative" scores 99% accuracy with zero recall.

Cost-sensitive thresholding — when mistakes have price tags $c_{FP}$ and $c_{FN}$, the right threshold minimizes:

$$\text{cost}(t) = c_{FP} \cdot FP(t) + c_{FN} \cdot FN(t)$$

There is no universally "correct" threshold — only the one your costs imply.

## Open the visualization

`open visualize.html in any browser`

## When to use this

- **Any deployed classifier with a decision attached** — fraud blocking, spam filtering, content moderation: the confusion matrix at your *production threshold* is the only honest report of what users actually experience.
- **Imbalanced problems** — disease screening, defect detection, churn: precision and recall stay meaningful when accuracy goes blind, because they each condition on a single row or column of the matrix.
- **Stakeholder conversations about trade-offs** — "we can catch 95% of fraud if we tolerate 3× the false alarms" is a confusion-matrix sentence; it turns a modeling knob into a business decision people can actually vote on.

## What breaks it

- **Reporting accuracy on imbalanced data** — a 99.2%-accurate cancer screener that misses 70% of tumors is a press release and a lawsuit in one number. Always check recall before celebrating accuracy.
- **Evaluating at the default threshold of 0.5** — 0.5 is an arbitrary library default, not a decision. If your FN costs 50× your FP, the right threshold may be 0.1, and every metric you quoted at 0.5 describes a system you should never ship.
- **Optimizing F1 when costs are asymmetric** — F1 weights precision and recall equally, which silently assumes a false alarm hurts exactly as much as a miss. In medical screening or fraud that assumption is wildly wrong; use the actual cost function or $F_\beta$.

## 5 Interview Questions

**1. Conceptual — "Your model is 99% accurate. Why might that be terrible?"**
Direct answer: if positives are 1% of the data, "always predict negative" also scores 99% — the model may have learned nothing while accuracy looks superb. Reason: accuracy averages over both classes weighted by their frequency, so the majority class drowns out the minority class you actually care about; recall on the positive class is what reveals it. Likely follow-up: *"So what would you report instead?"* — precision, recall, and F1 on the minority class, plus the full matrix at the production threshold; if a single number is demanded, PR-AUC.

**2. Mathematical — "Why is F1 the harmonic mean and not the arithmetic mean of precision and recall?"**
Direct answer: the harmonic mean collapses toward the smaller value, so F1 is only high when *both* precision and recall are high — arithmetic mean would award 0.5 to a degenerate model with precision 1.0 and recall ≈ 0. Reason: $2pr/(p+r) \leq \min$-dominated behavior makes F1 ungameable by sacrificing one side; an arithmetic mean is trivially gamed by predicting one positive correctly. Likely follow-up: *"What if recall matters more than precision?"* — use $F_\beta$ with $\beta > 1$ (e.g. $F_2$ weights recall higher); $\beta$ encodes how many units of precision you'd trade for one of recall.

**3. Practical — "Stakeholders say the fraud model raises too many false alarms. What do you change?"**
Direct answer: raise the decision threshold — precision rises as low-confidence flags are dropped — and present the new confusion matrix showing exactly how many frauds will now slip through. Reason: this needs no retraining; threshold choice is a post-hoc knob on the same scores, and the matrix makes the cost of the change legible (e.g. FP halves, FN grows by 12). Likely follow-up: *"And if they want fewer false alarms AND no extra misses?"* — that requires a better model (more separation between score distributions), not a better threshold; show them the two overlapping distributions to explain why.

**4. Gotcha — "Precision is 0.95 and recall is 0.95. Is the model good?"**
Direct answer: can't say — those numbers depend on the test set's class balance and threshold, and say nothing about whether the test set resembles production. Reason: precision is prevalence-dependent: the same classifier at the same threshold can drop from 0.95 to 0.30 precision when deployed where positives are 50× rarer, because FP comes from the (now much larger) negative pool. Likely follow-up: *"Which of the two metrics survives a prevalence shift?"* — recall (and FPR), because each conditions on a single true class; precision mixes the classes, so it shifts with the base rate.

**5. System design — "Design the alerting for a sepsis early-warning system in a hospital. How do you set the threshold?"**
Direct answer: start from the cost asymmetry — a missed sepsis case can be fatal while a false alert costs a nurse's check — so set the threshold to minimize $c_{FP} \cdot FP + c_{FN} \cdot FN$ with $c_{FN} \gg c_{FP}$, landing at high recall (say 90%+) and accepting modest precision. Reason: the threshold is a clinical-policy decision, so derive it with clinicians from explicit costs, validate the matrix on a held-out hospital, and monitor recall in production since prevalence drifts. Likely follow-up: *"Alert fatigue is now a problem — nurses ignore the alerts."* — that means the *effective* $c_{FP}$ was underestimated: a false alarm doesn't cost one check, it erodes trust in all future alerts; re-price the costs, add alert tiers, and treat precision as a first-class constraint.
