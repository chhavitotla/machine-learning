# Linear Regression

## The Intuition

You have a scatter of points and you suspect there's a straight-line relationship hiding in the noise — say, apartment size vs. rent. Linear regression draws that line, but the clever part is how it picks *which* line: it measures how far every point is from a candidate line, squares those distances (so misses in either direction count, and big misses count a lot), adds them up, and then nudges the line's slope and intercept to make that total smaller. Repeat the nudging until no nudge helps. That nudging process — gradient descent — is the same engine that trains GPT, just with two parameters instead of billions. If you deeply get this one, you get the skeleton of all of deep learning.

## The Math

The model:

$$\hat{y} = wx + b$$

- $\hat{y}$ — the predicted value
- $w$ — the slope (weight): how much $y$ changes per unit of $x$
- $b$ — the intercept (bias): the prediction when $x = 0$

This just says: predictions are a straight line through the data.

The loss (Mean Squared Error):

$$L(w, b) = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

- $n$ — number of data points
- $y_i$ — the true value of point $i$
- $\hat{y}_i$ — the model's prediction for point $i$

This measures how wrong the line is, on average, with big errors punished quadratically.

The gradients:

$$\frac{\partial L}{\partial w} = -\frac{2}{n}\sum_{i=1}^{n} x_i(y_i - \hat{y}_i) \qquad \frac{\partial L}{\partial b} = -\frac{2}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)$$

- $\frac{\partial L}{\partial w}$ — how much the loss changes if you nudge the slope
- $\frac{\partial L}{\partial b}$ — how much the loss changes if you nudge the intercept

These tell you which direction to tilt and shift the line to reduce the error.

The update rule:

$$w \leftarrow w - \alpha \frac{\partial L}{\partial w} \qquad b \leftarrow b - \alpha \frac{\partial L}{\partial b}$$

- $\alpha$ — the learning rate: how big a step to take

Walk downhill on the loss surface, one small step at a time.

There's also a closed-form solution (the *normal equation*), $\mathbf{w} = (X^TX)^{-1}X^Ty$, which solves it in one shot — but it costs $O(d^3)$ in the number of features, which is why gradient descent wins at scale.

## Open the visualization

`open visualize.html in any browser`

## When to use this

- **Pricing and forecasting baselines** — predicting house prices, sales volume, or delivery times where you need a model stakeholders can read off as "every extra bedroom adds $40k."
- **A/B test effect estimation** — regressing a metric on a treatment indicator plus covariates gives you the treatment effect with confidence intervals.
- **The first model on any new tabular problem** — it trains in milliseconds and gives you a floor; if your neural net can't beat linear regression, your features are the problem.

## What breaks it

- **Nonlinear relationships** — fit a line to seasonal sales data and it will confidently predict straight through the holiday spike, missing every peak. You'll ship a forecast that's wrong exactly when it matters.
- **Outliers** — because errors are *squared*, one data-entry typo (a $10M house logged as $10B) can drag the whole line toward it and corrupt every other prediction.
- **Correlated features (multicollinearity)** — with `sqft` and `num_rooms` both present, the model can assign +500 to one and −450 to the other; predictions stay fine but the coefficients become uninterpretable garbage, which is fatal if anyone reads them as effects.

## 5 Interview Questions

**1. Conceptual — "Why do we square the errors instead of taking absolute values?"**
Direct answer: squaring makes the loss smooth and differentiable everywhere, punishes large errors disproportionately, and corresponds to maximum-likelihood estimation under Gaussian noise. Reason: $|x|$ has no derivative at 0 and gives constant-magnitude gradients, so optimization is messier; squared error's gradient shrinks as you approach the optimum, giving natural convergence. Likely follow-up: *"When would you prefer absolute error?"* — when outliers are expected and you want robustness; MAE corresponds to predicting the conditional median rather than the mean, so it's the right call for, say, delivery-time predictions with occasional extreme delays.

**2. Mathematical — "Derive the gradient of MSE with respect to w."**
Direct answer: write $L = \frac{1}{n}\sum(y_i - wx_i - b)^2$, apply the chain rule: the outer derivative gives $2(y_i - wx_i - b)$, the inner derivative of the residual w.r.t. $w$ is $-x_i$, so $\partial L/\partial w = -\frac{2}{n}\sum x_i(y_i - \hat{y}_i)$. Reason: it's pure chain rule — derivative of square times derivative of inside. Likely follow-up: *"Set it to zero and solve"* — that's the normal equation; be ready to write $(X^TX)^{-1}X^Ty$ and note it requires $X^TX$ to be invertible, which fails under perfect multicollinearity.

**3. Practical — "Your linear regression has high training error AND high test error. What do you do?"**
Direct answer: that's underfitting (high bias), so add capacity — engineer nonlinear features (polynomials, interactions, log transforms), or move to a more expressive model. Reason: when train and test error are both high and close together, the model class can't represent the pattern; more data or regularization won't help because the problem isn't variance. Likely follow-up: *"And if training error is low but test error is high?"* — that's overfitting: regularize (ridge/lasso), simplify features, or get more data.

**4. Gotcha — "I added a feature and R² went up. Is my model better?"**
Direct answer: not necessarily — R² *never decreases* when you add a feature, even a column of random noise. Reason: OLS can always exploit a new degree of freedom to fit the training data at least as well; that's mechanical, not evidence of signal. Likely follow-up: *"So what do you look at instead?"* — adjusted R², which penalizes parameter count, or better, cross-validated error on held-out data, which is the only honest measure.

**5. System design — "Design a home-price estimator for a real-estate site. Would you use linear regression?"**
Direct answer: yes, as the v1 production model — features like size, location (target-encoded zip), age, and beds/baths into ridge regression, trained nightly, served behind a cache. Reason: it's interpretable (you can show users "why this estimate"), trivially cheap to serve at millions of requests, and gives a baseline that justifies any added complexity later. Likely follow-up: *"Where does it hit a wall?"* — interactions and nonlinearity (location × size effects), at which point you move to gradient boosting for accuracy but keep the linear model as a sanity-check shadow and for the explanation UI.
