# Gradient Boosting

## The Intuition

One small decision tree is a blunt instrument — it can only draw a step function. Gradient boosting's trick: don't ask one tree to be right, ask a *sequence* of trees to each fix what's still wrong. Start by predicting the mean. Compute the residuals — how far off you are at every point. Fit a tiny tree to *those residuals*, add it (shrunk by a learning rate ν) to your running prediction, and repeat. Each round, the new tree aims squarely at the region where the current ensemble is most wrong, so the errors get hammered down everywhere. The "gradient" in the name: for squared error, residuals literally *are* the negative gradient of the loss with respect to the predictions — so this is gradient descent, but the steps are trees, taken in function space instead of parameter space. It's the engine inside XGBoost and LightGBM, which won basically every tabular Kaggle competition of the 2010s.

## The Math

The model — an additive ensemble of $M$ small trees:

$$F_M(x) = F_0(x) + \nu \sum_{m=1}^{M} h_m(x)$$

- $F_0(x)$ — the initial prediction: just the mean of $y$ for squared error
- $h_m$ — the $m$-th weak learner, a shallow regression tree
- $\nu$ — the learning rate (shrinkage), typically 0.01–0.3

The general recipe — fit each tree to the *negative gradient* of the loss:

$$r_i^{(m)} = -\left.\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right|_{F = F_{m-1}}$$

- $r_i^{(m)}$ — the "pseudo-residual" of point $i$ at round $m$
- $L$ — any differentiable loss

For squared error $L = \frac{1}{2}(y - F(x))^2$, this collapses to the plain residual:

$$r_i^{(m)} = y_i - F_{m-1}(x_i)$$

— which is why "fit a tree to the leftovers" is doing gradient descent in disguise.

The update:

$$F_m(x) = F_{m-1}(x) + \nu \, h_m(x)$$

- each round takes a small step in function space along the tree that best approximates the gradient

Shrinkage $\nu$ and round count $M$ trade off: smaller $\nu$ needs more trees but generalizes better — many tiny corrections beat few big ones. Swap in a different loss (absolute error, log-loss, quantile) and only the pseudo-residual formula changes; that pluggability is the framework's superpower.

## Open the visualization

`open visualize.html in any browser`

## When to use this

- **Tabular data, full stop** — churn prediction, credit risk, click-through rate: with heterogeneous numeric + categorical features, tuned boosting (XGBoost/LightGBM/CatBoost) is still the strongest default, routinely beating neural nets.
- **Problems with non-obvious feature interactions** — trees discover splits like "high usage AND short tenure" automatically; no feature engineering of interaction terms needed.
- **When you need a custom objective** — quantile loss for delivery-time ranges, Poisson for counts, pairwise loss for ranking (this is how search-result rankers work): plug in any differentiable loss and the same machinery runs.

## What breaks it

- **Extrapolation** — trees can only predict values seen in leaves, so the ensemble outputs a flat line outside the training range. Train on 2020–2023 sales and the model will never predict above the historical max, no matter the trend.
- **Too many rounds without early stopping** — train error glides toward zero forever while holdout error quietly U-turns; ship the 500-round model that looked great on training data and you've shipped memorized noise. Always hold out a validation set and stop when it stalls.
- **Label noise and outliers with squared error** — each round, the biggest residuals attract the next tree, so a wrongly-labeled point gets an entire sequence of trees devoted to chasing it. Robust losses (Huber, absolute) or row subsampling blunt this.

## 5 Interview Questions

**1. Conceptual — "What's the actual difference between bagging (random forest) and boosting?"**
Direct answer: bagging trains many *independent* deep trees in parallel on bootstrapped data and averages them to cut variance; boosting trains shallow trees *sequentially*, each fitting the previous ensemble's errors, to cut bias. Reason: a random forest's trees never see each other's mistakes — diversity comes from randomness; boosting's whole mechanism is that tree $m$ exists only to fix tree $1..m{-}1$. Likely follow-up: *"Which overfits more easily?"* — boosting: it relentlessly drives training error down, so it needs ν, depth limits, and early stopping, while forests are remarkably hard to overfit by adding trees.

**2. Mathematical — "Why is it called *gradient* boosting? Where's the gradient?"**
Direct answer: each tree is fit to the negative gradient of the loss with respect to the current predictions, $r_i = -\partial L(y_i, F(x_i))/\partial F(x_i)$ — for squared error that's exactly the residual $y_i - F(x_i)$. Reason: think of the $n$ predictions as $n$ free parameters; the steepest-descent direction in that space is the vector of negative gradients, and the tree is the best function-shaped approximation of that direction — gradient descent in function space. Likely follow-up: *"What changes for classification?"* — use log-loss; pseudo-residuals become $y_i - p_i$ (label minus predicted probability), and the trees regress on those even though the task is classification.

**3. Practical — "Your boosted model overfits. Rank the knobs you'd turn."**
Direct answer: first early stopping on a validation set (cap rounds where holdout error bottoms out), then lower ν (with proportionally more rounds), then shallower trees (depth 3–6), then subsampling rows/columns per tree, then min-samples-per-leaf and L2 on leaf weights. Reason: rounds × ν controls total fitting capacity most directly; depth controls interaction order; subsampling decorrelates the trees, working like bagging on top of boosting. Likely follow-up: *"Why do small ν + many trees beat large ν + few trees?"* — each small step leaves more residual signal for later trees and averages out greedy split mistakes, acting as implicit regularization; empirically the holdout minimum is almost always lower.

**4. Gotcha — "Training error has been flat for 200 rounds, so more rounds are harmless, right?"**
Direct answer: wrong on both counts — train error in boosting essentially never goes flat (it keeps creeping down), and even when improvements look negligible, holdout error can be steadily *rising*; extra rounds are not free. Reason: each new tree still finds something residual to fit, and past the validation minimum that something is noise; the train curve cannot tell you when that happens. Likely follow-up: *"So what's the stopping criterion?"* — monitor validation loss and stop after it fails to improve for k consecutive rounds (early-stopping patience), then use the round count from the minimum.

**5. System design — "Design the ETA prediction system for a food-delivery app. Why boosting?"**
Direct answer: gradient-boosted trees on tabular features — distance, restaurant prep-time history, courier load, hour-of-week, weather — trained on completed deliveries, with quantile loss to ship a P50 and P90 ETA rather than a single number. Reason: the data is purely tabular with strong interactions (rain × dinner-rush), boosting handles mixed features without scaling, and the quantile objective drops straight into the same framework — "arrives by 7:45" is a P90 statement. Likely follow-up: *"Where does it fail in production?"* — extrapolation on unseen conditions (new city, record storm) where trees flat-line; mitigate with monotonic constraints (ETA must increase with distance), fallback heuristics, and frequent retraining on fresh data.
