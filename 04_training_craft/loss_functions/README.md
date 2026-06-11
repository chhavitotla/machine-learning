# Loss Functions — MSE, MAE, Huber, Cross-Entropy

## The Intuition

A loss function is how you tell the model what "wrong" costs — and the model will exploit your pricing scheme ruthlessly. MSE squares the error, so being off by 10 costs a hundred times more than being off by 1: the model becomes obsessed with its worst mistakes and will warp the whole fit to appease one bad data point. MAE charges the same rate per unit of error no matter how large, so it shrugs at outliers — but it also can't say "you're very close now, ease off," because its push is the same size whether you're off by 0.001 or by 1000. Huber splits the difference: act like MSE for small errors (smooth, gentle landing) and like MAE for big ones (outliers can't bully you). Cross-entropy is a different game entirely — for classification you're pricing *confidence*, and it charges astronomically for being confidently wrong, which is exactly the behavior you want stamped out. Pick the loss, and you've picked what the model fears.

## The Math

**MSE** (mean squared error):

$$L_{\text{MSE}} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2 \qquad \frac{\partial L}{\partial \hat{y}_i} = \frac{2}{n}(\hat{y}_i - y_i)$$

- $y_i$ — the true target for example $i$
- $\hat{y}_i$ — the model's prediction
- $n$ — number of examples

The penalty grows with the *square* of the error, so the gradient grows linearly — the farther off you are, the harder you get pushed, without limit.

**MAE** (mean absolute error):

$$L_{\text{MAE}} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i| \qquad \frac{\partial L}{\partial \hat{y}_i} = \frac{1}{n}\,\text{sign}(\hat{y}_i - y_i)$$

- $\text{sign}(\cdot)$ — $+1$ for positive error, $-1$ for negative (undefined exactly at zero)

Every error gets the same constant-magnitude push toward the target, which makes outliers powerless but also makes the final approach jittery — the gradient never shrinks as you converge.

**Huber:**

$$L_\delta(e) = \begin{cases} \frac{1}{2}e^2 & |e| \le \delta \\ \delta\left(|e| - \frac{1}{2}\delta\right) & |e| > \delta \end{cases} \qquad \frac{\partial L_\delta}{\partial e} = \begin{cases} e & |e| \le \delta \\ \delta \cdot \text{sign}(e) & |e| > \delta \end{cases}$$

- $e = \hat{y} - y$ — the residual
- $\delta$ — the crossover threshold: errors smaller than this are "inliers," larger are "outliers"

Quadratic near the target (smooth convergence like MSE), linear far away (bounded gradient like MAE) — equivalently, it's MSE with the gradient clipped at $\delta$.

**Binary cross-entropy:**

$$L_{\text{BCE}} = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i \log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i)\right]$$

- $y_i \in \{0, 1\}$ — the true class label
- $\hat{p}_i \in (0, 1)$ — the predicted probability of class 1, usually $\sigma(z_i)$ from a sigmoid over the raw score $z_i$

The cost is the negative log of the probability assigned to the truth — predicting 0.01 for a true positive costs $-\log(0.01) \approx 4.6$ while 0.99 costs $\approx 0.01$, so confident wrongness is punished exponentially harder than honest uncertainty; paired with a sigmoid, the gradient with respect to the raw score collapses to the beautifully simple $\hat{p}_i - y_i$. The multi-class version, **categorical cross-entropy**, is the same idea over a softmax: the negative log-probability assigned to the correct class.

## Open the visualization

`open visualize.html in any browser` — drag the prediction slider to watch all four loss curves and their gradients live, then drag the outlier point in the bottom panel to see an MSE-fit line chase it while the MAE-fit line refuses to budge.

## When to use this

- **MSE for clean regression where big misses are genuinely worse** — predicting energy demand for grid planning: a 10× error causes blackouts, not 10× inconvenience, so the quadratic penalty matches the real cost. Also the default for Gaussian-ish noise (MSE is its maximum-likelihood twin).
- **MAE (or Huber) when the data has fat tails or label errors** — predicting delivery times when 2% of records have a warehouse glitch logging 400-hour deliveries: under MSE those records dominate the loss and drag every prediction upward; MAE fits the typical order. Huber with a tuned $\delta$ adds smooth convergence — it's the standard for bounding-box regression in object detection for exactly this reason.
- **Cross-entropy for anything classification** — fraud detection, sentiment, image labels: you want calibrated probabilities and a loss that hammers confident mistakes. Never MSE on probabilities — its gradient *vanishes* exactly when the model is confidently wrong (see question 4).

## What breaks it

- **MSE + one corrupted label** — a single target mistyped as 1000 instead of 10 contributes $\sim 10^6$ to the loss; the model sacrifices accuracy on every clean point to shave error off the typo. In production this looks like a model that's mysteriously biased high, and the cause is three rows in the training set.
- **BCE with predictions that touch 0 or 1** — $\log(0) = -\infty$: one fully-saturated wrong prediction makes the loss NaN and the run dies. Real implementations clamp $\hat{p}$ to $[\epsilon, 1-\epsilon]$ or, better, fuse sigmoid+BCE into one numerically-stable op (`BCEWithLogitsLoss`) — using the unfused version is a classic silent bug.
- **Optimizing a loss that isn't your metric** — train on MSE, get judged on MAPE (percentage error), and the model over-prioritizes high-value targets where absolute errors are big but percentage errors are fine. The loss is a *proxy* for what you care about; when the proxy and the metric rank models differently, you ship the wrong model while your training curves look great.

## 5 Interview Questions

**1. Conceptual — "Why is MSE so much more sensitive to outliers than MAE?"**
Direct answer: squaring makes the penalty grow quadratically — an outlier 100× farther away contributes 10,000× more loss under MSE but only 100× more under MAE, so MSE's optimum gets pulled toward extreme points while MAE's barely moves. Reason: it falls out of the gradients — MSE's gradient is proportional to the error, so the outlier's vote on the parameters is proportionally huge, while MAE gives every point the same ±1 vote; equivalently, MSE's minimizer is the *mean* of the targets (outlier-sensitive), MAE's the *median* (outlier-robust). Likely follow-up: *"So when would you still want MSE despite outliers?"* — when the outliers are signal rather than noise (extreme load spikes you must not miss), or when the data is already clean and you want the smooth convergence of a shrinking gradient.

**2. Mathematical — "Derive the gradient of binary cross-entropy with a sigmoid, with respect to the raw logit."**
Direct answer: with $\hat{p} = \sigma(z)$ and $L = -[y\log\hat{p} + (1-y)\log(1-\hat{p})]$, chain rule gives $\partial L/\partial \hat{p} = -y/\hat{p} + (1-y)/(1-\hat{p})$ and $\partial \hat{p}/\partial z = \hat{p}(1-\hat{p})$; multiply and the $\hat{p}(1-\hat{p})$ factors cancel, leaving $\partial L/\partial z = \hat{p} - y$. Reason: that cancellation is the whole point — sigmoid alone has a vanishing derivative when saturated, but BCE's $1/\hat{p}$ blow-up exactly compensates, so a confidently-wrong prediction ($\hat{p}=0.99$, $y=0$) still gets a near-maximal gradient and can recover. Likely follow-up: *"Does the same happen with softmax + categorical cross-entropy?"* — yes, identically: the gradient with respect to the logits is $\hat{p} - y$ with one-hot $y$, which is why softmax and cross-entropy are always fused into one layer.

**3. Practical — "You're predicting house prices and your test MAE looks fine, but a handful of predictions are off by millions. What do you change?"**
Direct answer: first check whether those extreme cases are noise (bad records → clean or drop them) or signal (real mansions → the model needs better features for them); if you must reduce worst-case misses, move from MAE toward MSE or Huber with a large $\delta$, and consider predicting $\log(\text{price})$ so the loss operates on relative error. Reason: MAE explicitly told the optimizer that a \$3M miss is only 3000× as bad as a \$1k miss, so it happily traded away tail accuracy — the fix is to re-price large errors, not to tune the optimizer. Likely follow-up: *"Why does the log transform help so much here?"* — prices are multiplicative (roughly log-normal), so MSE on log-price penalizes *percentage* error: a \$50k miss on a \$200k home and a \$500k miss on a \$2M home cost the same, which usually matches what the business cares about.

**4. Gotcha — "Cross-entropy and MSE both go to zero at a perfect prediction — so for classification it doesn't really matter which I use, right?"**
Direct answer: it matters enormously — with a sigmoid output, MSE's gradient with respect to the logit is $(\hat{p}-y)\cdot\hat{p}(1-\hat{p})$, and that $\hat{p}(1-\hat{p})$ factor goes to zero whenever $\hat{p}$ saturates, *including when the model is confidently wrong*: a prediction of 0.999 for a true negative receives almost no corrective gradient and learning stalls. Reason: BCE is the loss whose gradient cancels the sigmoid's saturation (the maximum-likelihood loss for a Bernoulli output), while MSE assumes Gaussian noise that classification doesn't have — same zero point, completely different dynamics far from it. Likely follow-up: *"Then why did early neural nets use MSE for classification?"* — habit from regression and weaker theory; the move to cross-entropy was one of the quiet fixes (alongside ReLU and better init) that made deep classifiers trainable.

**5. System design — "You're building a demand-forecasting service for thousands of products, from steady sellers to viral spikes. How do you design the loss?"**
Direct answer: no single off-the-shelf loss survives this — combine (1) a scale-free formulation (predict log-demand, or use a quantile/pinball loss) so a 10-unit error on a 20-unit product isn't priced like one on a 20,000-unit product; (2) robustness to unpredictable spikes via Huber or quantile losses, while routing *predictable* spikes (promotions, holidays) into features; (3) asymmetry matching the business — if stockouts cost 5× what overstock costs, weight under-prediction 5× or train quantile regression at the 80th percentile instead of the median. Reason: the loss is the contract between the optimizer and the business — every mismatch between training loss and real cost (scale, tails, asymmetry) becomes a systematic bias in production. Likely follow-up: *"How do you validate the loss choice itself?"* — backtest each candidate and evaluate on the *business* metric (realized inventory cost, stockout rate), never the training loss; the winner is the loss whose optimum transfers, and that's an empirical question.
