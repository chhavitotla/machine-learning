# Regularization

## The Intuition

Give a flexible model enough freedom and it will do something embarrassing: instead of learning the pattern in your data, it will memorize the noise. A degree-15 polynomial through 26 noisy points can hit every single one — and be catastrophically wrong everywhere in between. Regularization is the fix, and the idea is almost insultingly simple: charge the model rent for complexity. Add a penalty to the loss that grows with the size of the weights, so the model only keeps a big weight if it earns its keep by reducing error more than it costs in penalty. L2 (ridge) charges rent on squared weights, which shrinks everything smoothly toward zero. L1 (lasso) charges on absolute values, which has a sharper personality: weights that don't pull their weight get evicted — set to *exactly* zero. Same principle, two different characters, and between them they cover most of what "don't overfit" means in practice.

## The Math

The regularized objective:

$$L(w) = \underbrace{\sum_{i=1}^{n}(y_i - \mathbf{x}_i^\top w)^2}_{\text{fit the data}} + \underbrace{\lambda \, \Omega(w)}_{\text{pay rent}}$$

- $\lambda$ — the regularization strength: how expensive complexity is
- $\Omega(w)$ — the penalty: $\|w\|_2^2 = \sum_j w_j^2$ for ridge, $\|w\|_1 = \sum_j |w_j|$ for lasso

Ridge has a closed form. Setting the gradient to zero:

$$w^* = (X^\top X + \lambda I)^{-1} X^\top y$$

- $X$ — the $n \times d$ feature matrix
- $\lambda I$ — the penalty showing up as a ridge down the diagonal (hence the name)

The $\lambda I$ does double duty: it shrinks the weights *and* guarantees the matrix is invertible even with perfectly correlated features.

Lasso has no closed form (the $|w_j|$ kink isn't differentiable at 0), but each coordinate alone has an exact solution — the soft-threshold:

$$w_j \leftarrow \frac{S(\rho_j, \lambda/2)}{z_j}, \qquad S(\rho, t) = \begin{cases} \rho - t & \rho > t \\ 0 & |\rho| \le t \\ \rho + t & \rho < -t \end{cases}$$

- $\rho_j$ — feature $j$'s correlation with the residual: what the weight "wants" to be
- $z_j = \sum_i x_{ij}^2$ — the normalizer
- $S$ — soft-thresholding: shrink toward zero, and *clip to exactly zero* inside $[-t, t]$

That clip-to-zero is the entire sparsity story: a feature whose evidence $\rho_j$ can't exceed the tax $\lambda/2$ gets weight exactly 0. Cycle through coordinates until nothing moves — that's coordinate descent.

Bayesian footnote worth knowing: ridge is MAP estimation with a Gaussian prior on weights; lasso with a Laplace prior. "Penalty" and "prior belief that weights are small" are the same math.

## Open the visualization

`open visualize.html in any browser`

## When to use this

- **Always, basically** — any model trained by loss minimization (linear, logistic, neural nets via weight decay) should ship with L2 unless you have a reason not to; it's one hyperparameter that buys robustness to noise and correlated features almost for free.
- **High-dimensional tabular data with suspected junk features** — lasso on 5,000 gene-expression features that zeroes all but 40 is doing feature selection and fitting in one move, and the surviving feature list is itself a deliverable.
- **Multicollinearity rescue** — when $X^\top X$ is near-singular (correlated features, more features than rows), ridge's $\lambda I$ makes the problem well-posed and the coefficients stable run-to-run.

## What breaks it

- **Unstandardized features** — the penalty charges all weights the same rent, but a feature measured in millimeters needs a weight 1000× larger than the same feature in meters. Skip standardization and λ silently punishes features for their units, not their uselessness.
- **Penalizing the intercept** — the bias term isn't complexity, it's where the data lives. Regularize it and at high λ your model gets dragged toward predicting 0 instead of the mean; every prediction inherits a systematic offset.
- **Lasso with correlated features** — given two near-duplicate features, lasso arbitrarily picks one and zeroes the other; tiny data changes flip the choice. If anyone reads the zeros as "this feature doesn't matter," they're being lied to. (Elastic net — L1 + L2 blended — is the standard fix.)

## 5 Interview Questions

**1. Conceptual — "Why does L1 produce exact zeros but L2 doesn't?"**
Direct answer: L2's gradient ($2\lambda w$) fades to nothing as $w \to 0$, so the push toward zero gets ever gentler and never finishes; L1's subgradient ($\pm\lambda$) stays constant-strength all the way in, so weights hit exactly zero and stay pinned there. Reason: geometrically, the L1 ball has corners on the axes and the loss contours tend to touch the constraint set at those corners, where coordinates are exactly zero; the L2 ball is round and touches at generic points. Likely follow-up: *"So when would you still prefer L2?"* — when most features genuinely matter a little (dense truth), when features are correlated (lasso picks arbitrarily), or inside neural nets, where sparsity in raw weights buys little and smooth shrinkage (weight decay) optimizes better.

**2. Mathematical — "Derive the ridge solution and explain what λI does to the eigenvalues."**
Direct answer: set $\nabla_w \left[\|y - Xw\|^2 + \lambda\|w\|^2\right] = -2X^\top(y - Xw) + 2\lambda w = 0$, giving $(X^\top X + \lambda I)w = X^\top y$. Reason: $X^\top X$ is PSD with eigenvalues $\sigma_i^2 \ge 0$; adding $\lambda I$ lifts each to $\sigma_i^2 + \lambda > 0$, so the inverse always exists, and each component of the solution gets shrunk by the factor $\sigma_i^2/(\sigma_i^2 + \lambda)$ — weak directions (small $\sigma_i$, where noise lives) are crushed hardest. Likely follow-up: *"What's the connection to the bias–variance tradeoff?"* — λ adds bias but cuts variance precisely in those ill-determined directions; test error is minimized at the λ where the marginal variance saved equals the marginal bias added.

**3. Practical — "How do you actually choose λ?"**
Direct answer: cross-validation over a log-spaced grid (e.g. $10^{-4}$ to $10^2$), picking the λ with the best held-out error — or the "1-SE rule": the largest λ within one standard error of the best, preferring the simpler model. Reason: λ trades training fit for generalization, so only held-out data can arbitrate; the curve of CV error vs λ is U-shaped and the grid must be log-spaced because λ's effect spans orders of magnitude. Likely follow-up: *"Anything special for lasso?"* — yes, use the regularization path: coordinate descent warm-started from the previous λ computes the whole path nearly for free, and you can watch the order in which features enter the model.

**4. Gotcha — "I applied ridge and my training error went up. Did I do something wrong?"**
Direct answer: no — that's the entire point; regularization *deliberately* sacrifices training fit, and the only number that can justify it is held-out error. Reason: the unregularized optimum is, by definition, the training-error minimizer, so any λ > 0 must raise training error; if test error also went up, λ is too large or the model was underfitting to begin with — regularizing an underfit model only digs the hole deeper. Likely follow-up: *"Your model overfits even at the best λ. Now what?"* — regularization isn't the only lever: get more data, cut features, add data augmentation, use early stopping (itself implicit regularization), or shrink the model class.

**5. System design — "You're building a churn model on 2,000 customer features, and the product team wants to know which features matter. Design the modeling approach."**
Direct answer: standardize everything, fit an elastic-net logistic regression with λ chosen by time-aware cross-validation, and hand the product team the sparse coefficient list — typically a few dozen surviving features with signs and magnitudes. Reason: L1 gives the interpretable shortlist the stakeholders actually asked for, the L2 component keeps correlated feature groups stable instead of arbitrarily zeroed, and a sparse linear model is auditable and cheap to serve. Likely follow-up: *"The PM reads a zeroed coefficient as 'tenure doesn't affect churn.' Is that right?"* — no: zero means "adds no predictive value *given the other features*," and with correlated features lasso's choice of which twin to keep is arbitrary — refit on bootstraps and report selection stability before letting anyone narrate the zeros.
