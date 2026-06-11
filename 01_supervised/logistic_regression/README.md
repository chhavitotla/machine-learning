# Logistic Regression

## The Intuition

Linear regression predicts a number; most real questions are yes/no — will this user churn, is this transaction fraud, is this email spam. The trick: keep the linear machinery ($w \cdot x + b$ gives a score), but squash the score through a sigmoid so it lands in $(0, 1)$ and reads as a *probability*. A score of 0 becomes 50/50; a big positive score becomes "almost surely class 1." Training nudges the weights so that confident wrong answers — saying 99% fraud on a legitimate purchase — hurt enormously, while hedged answers hurt a little. The boundary where the model says "exactly 50/50" is still a straight line, so logistic regression is a linear classifier wearing a probability costume. It is also, almost exactly, a single neuron — stack and compose these and you have a neural network.

## The Math

The model:

$$p = \sigma(w \cdot x + b) = \frac{1}{1 + e^{-(w \cdot x + b)}}$$

- $w \cdot x + b$ — the linear score (the *logit*): positive means leaning class 1
- $\sigma$ — the sigmoid: squashes any real number into $(0, 1)$
- $p$ — the predicted probability that $y = 1$

This says: compute a linear score, then read it as a probability.

The loss (binary cross-entropy):

$$L = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i \log p_i + (1 - y_i)\log(1 - p_i)\right]$$

- $y_i \in \{0, 1\}$ — the true class of point $i$
- $p_i$ — the model's probability for point $i$

This is negative log-likelihood: each point charges you $-\log$ of the probability you assigned to *its true class*. Confidently wrong ($p \to 0$ when $y = 1$) costs $\to \infty$.

The gradient:

$$\frac{\partial L}{\partial w} = \frac{1}{n}\sum_{i=1}^{n}(p_i - y_i)\,x_i \qquad \frac{\partial L}{\partial b} = \frac{1}{n}\sum_{i=1}^{n}(p_i - y_i)$$

- $(p_i - y_i)$ — prediction minus truth, the entire error signal

The $1/p$ from the log and the $p(1-p)$ from the sigmoid's derivative cancel exactly, leaving *(prediction − truth) × input* — the same shape as linear regression's gradient. This cancellation is why sigmoid + cross-entropy is the canonical pairing.

The update rule:

$$w \leftarrow w - \alpha \frac{\partial L}{\partial w} \qquad b \leftarrow b - \alpha \frac{\partial L}{\partial b}$$

- $\alpha$ — the learning rate

Cross-entropy with a sigmoid is *convex* in $(w, b)$: one bowl, one global minimum, no bad luck possible.

## Open the visualization

`open visualize.html in any browser`

## When to use this

- **Risk scoring with calibrated probabilities** — credit default, churn, fraud: anywhere downstream decisions need an actual probability ("decline above 3% default risk"), not just a label.
- **High-stakes models that must be explained** — in lending and healthcare, regulators and clinicians can audit "each prior late payment multiplies the odds of default by 1.4×"; that's a legal requirement, not a nicety.
- **The classification baseline, especially on wide sparse data** — text with TF-IDF or hashed features at millions of dimensions, where logistic regression trains in seconds and is brutally hard to beat.

## What breaks it

- **Nonlinear class boundaries** — if class 1 surrounds class 0 in a ring, no straight line separates them; the model converges to near-coin-flip probabilities everywhere and looks "calibrated" while being useless. You need feature crosses or a nonlinear model.
- **Perfectly separable data** — counterintuitively, *too easy* a problem breaks it: the loss keeps improving as $\|w\| \to \infty$, weights blow up, and every prediction saturates to 0 or 1. Regularization isn't optional here; it's what makes the optimum exist.
- **Class imbalance + accuracy-blind thresholds** — at 1:1000 fraud rates, the default 0.5 threshold predicts "legit" for everything and scores 99.9% accuracy while catching zero fraud. The probabilities may be fine; the threshold and the metric are the trap.

## 5 Interview Questions

**1. Conceptual — "Why can't you just use linear regression with a 0/1 target and threshold it?"**
Direct answer: you can fit it, but the outputs aren't probabilities (they go below 0 and above 1), and squared error punishes *confidently correct* points — a point predicted at 3.0 with label 1 drags the boundary as if it were an error. Reason: MSE on labels penalizes distance from exactly 0 or 1, so well-classified outliers distort the fit, while cross-entropy only ever rewards moving probability toward the true class. Likely follow-up: *"So what exactly does the sigmoid buy you?"* — outputs that are valid probabilities, a likelihood-based loss that is convex, and log-odds linearity, which makes coefficients interpretable as odds-ratio multipliers.

**2. Mathematical — "Derive the gradient of cross-entropy through the sigmoid."**
Direct answer: with $z = w \cdot x + b$ and $p = \sigma(z)$, use $\sigma'(z) = p(1-p)$; then $\partial L/\partial p = -y/p + (1-y)/(1-p)$, and multiplying by $\sigma'(z)$ collapses everything to $\partial L/\partial z = p - y$, hence $\partial L/\partial w = (p - y)x$. Reason: the $p(1-p)$ from the sigmoid cancels the denominators from the log terms — that cancellation is the whole point of the pairing. Likely follow-up: *"What goes wrong if you use MSE with a sigmoid instead?"* — the gradient keeps a $p(1-p)$ factor, which vanishes when the model is saturated-and-wrong ($p \approx 0$, $y = 1$), so learning stalls exactly when the error is largest; it's also non-convex.

**3. Practical — "Your fraud model outputs probabilities. How do you pick the threshold?"**
Direct answer: not 0.5 — pick it from the business cost matrix: if a missed fraud costs 100× a false alarm, sweep thresholds and choose the one minimizing expected cost (equivalently, optimize on the precision-recall curve), then validate the choice on held-out data. Reason: the model's job is calibrated probabilities; the threshold is a *business decision* layered on top, and the optimal point moves whenever costs or base rates move. Likely follow-up: *"How do you check the probabilities are trustworthy enough to do this?"* — a calibration curve (reliability diagram): bucket predictions and check that the "30% risk" bucket is actually fraudulent ~30% of the time.

**4. Gotcha — "My logistic regression hit 100% training accuracy and the weights keep growing every epoch. Great, right?"**
Direct answer: no — your data is linearly separable, and the MLE doesn't exist: scaling $w$ up forever keeps shrinking the loss toward zero, so gradient descent never converges and predictions saturate to overconfident 0s and 1s. Reason: cross-entropy strictly rewards more confidence on correctly-classified points, so with a perfect separating line the only way to improve is to inflate $\|w\|$. Likely follow-up: *"Fix?"* — L2 regularization, which adds a $\lambda\|w\|^2$ bowl so a finite optimum exists; this is why sklearn regularizes by default and why your "unregularized" coefficients from it surprised you.

**5. System design — "Design the ranking model for which posts appear in a feed. Where does logistic regression fit?"**
Direct answer: as the classic CTR model — predict $P(\text{click} \mid \text{user}, \text{post})$ with logistic regression over sparse crossed features (user × topic, hour × category), trained online with frequent updates, scores used to rank candidates. Reason: it serves in microseconds at billions of requests, handles millions of sparse features, supports per-feature online updates, and its probabilities plug directly into expected-value ranking (bid × pCTR in ads); this was the production architecture at Google and Facebook for years. Likely follow-up: *"Why did the industry move to deep models, and what survived?"* — manual feature crosses don't scale to discovering interactions; embeddings + DNNs learn them. What survived is the head: the final layer of nearly every CTR network is still a sigmoid trained with cross-entropy — logistic regression with learned features.
