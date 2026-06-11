# Support Vector Machine

## The Intuition

Lots of lines separate two clouds of points — which one should you trust? The SVM's answer: the one that leaves the **widest street** between the classes. A boundary that barely squeaks past the training points will misclassify the first new point that lands slightly off; a boundary in the middle of the widest possible gap has the most room for error. The beautiful twist is that only the points *on the edge of the street* matter — move or delete any other point and the boundary doesn't budge. Those edge points "support" the boundary, hence support vectors. Real data is rarely perfectly separable, so the *soft margin* version lets points violate the street for a price: the hyperparameter C sets how expensive a violation is, trading street width against training mistakes.

## The Math

The model:

$$f(\mathbf{x}) = \mathbf{w} \cdot \mathbf{x} + b, \qquad \hat{y} = \text{sign}(f(\mathbf{x}))$$

- $\mathbf{w}$ — the weight vector, perpendicular to the boundary
- $b$ — the offset of the boundary from the origin
- $\hat{y} \in \{-1, +1\}$ — the predicted class

The boundary is $f(\mathbf{x}) = 0$; the margin lines are $f(\mathbf{x}) = \pm 1$, and the distance between them is $2/\|\mathbf{w}\|$ — **shrinking $\|\mathbf{w}\|$ widens the street**.

The soft-margin objective:

$$\min_{\mathbf{w}, b} \;\; \frac{\lambda}{2}\|\mathbf{w}\|^2 + \frac{1}{n}\sum_{i=1}^{n} \max\!\big(0,\; 1 - y_i f(\mathbf{x}_i)\big)$$

- $\lambda$ — regularization strength (equivalently $\lambda = \frac{1}{nC}$; big C ⇔ small λ ⇔ violations expensive)
- $\max(0, 1 - y_i f(\mathbf{x}_i))$ — the **hinge loss**: zero once a point clears its margin line, then linear

The first term wants a wide street; the second charges for every point inside or on the wrong side of it.

The subgradient update (Pegasos):

$$\mathbf{w} \leftarrow \mathbf{w} - \eta_t \lambda \mathbf{w} + \eta_t \, y_i \mathbf{x}_i \,[\,y_i f(\mathbf{x}_i) < 1\,], \qquad \eta_t = \frac{1}{\lambda t}$$

- $\eta_t$ — a decaying step size
- $[\cdot]$ — the indicator: the hinge term only pushes when the point violates its margin

Points that clear the margin contribute *nothing* to the gradient — this is exactly why only support vectors define the boundary.

For nonlinear boundaries, the dual formulation depends on data only through inner products $\mathbf{x}_i \cdot \mathbf{x}_j$, so you can swap in a kernel $K(\mathbf{x}_i, \mathbf{x}_j)$ — e.g. the RBF kernel — and separate in an implicit high-dimensional space without ever computing coordinates there. That's the *kernel trick*.

## Open the visualization

`open visualize.html in any browser`

## When to use this

- **Small-to-medium tabular datasets with clear structure** — text classification with TF-IDF features, bioinformatics, fraud flags: thousands of rows where a max-margin boundary generalizes better than a brittle one.
- **High-dimensional, few-samples regimes (p ≫ n)** — gene expression with 20k features and 200 patients; the margin objective plus regularization resists overfitting where many models drown.
- **When you need a strong classifier with few knobs** — a linear SVM has essentially one hyperparameter (C); an RBF SVM has two (C, γ). Easy to cross-validate exhaustively.

## What breaks it

- **Large datasets** — kernel SVMs train in roughly $O(n^2)$–$O(n^3)$ and must keep support vectors around at inference; at a few hundred thousand rows training stalls and serving bloats. Gradient boosting or a linear model wins on throughput.
- **Noisy, heavily overlapping classes with C set high** — every stray point becomes a margin violation the optimizer contorts to dodge; the "maximum margin" boundary ends up memorizing noise. You'll see beautiful training accuracy and a test set in flames.
- **Unscaled features** — the margin is measured in raw Euclidean distance, so a feature ranging 0–10,000 utterly dominates one ranging 0–1. Forget to standardize and the SVM is effectively fitting a single feature.

## 5 Interview Questions

**1. Conceptual — "Why maximize the margin? Any separating line gets 100% training accuracy."**
Direct answer: training accuracy says nothing about new points; the max-margin boundary is the one most robust to perturbation, since every training point sits as far as possible from the decision surface. Reason: statistically, generalization bounds for linear classifiers depend on the margin, not the dimension — a fat margin acts like a capacity control even in huge feature spaces. Likely follow-up: *"Which points determine that boundary?"* — only the support vectors; the solution is a sparse combination of margin-touching points, and deleting any non-support point provably changes nothing.

**2. Mathematical — "Show that the distance between the margin lines is 2/‖w‖."**
Direct answer: the lines $\mathbf{w}\cdot\mathbf{x}+b=+1$ and $-1$ are parallel with normal $\mathbf{w}$; the distance from a point on one to the other plane is $|(+1)-(-1)|/\|\mathbf{w}\| = 2/\|\mathbf{w}\|$ by the point-to-plane formula. Reason: moving along the unit normal $\mathbf{w}/\|\mathbf{w}\|$ changes $f(\mathbf{x})$ at rate $\|\mathbf{w}\|$, so covering a gap of 2 in $f$ takes a distance of $2/\|\mathbf{w}\|$. Likely follow-up: *"So why minimize ‖w‖²?"* — maximizing $2/\|\mathbf{w}\|$ is equivalent to minimizing $\|\mathbf{w}\|$, and squaring it makes the objective convex and differentiable, giving a clean quadratic program.

**3. Practical — "Your SVM gets 99% train / 70% test accuracy. What do you tune first?"**
Direct answer: lower C (and if using RBF, lower γ) — the model is buying training accuracy with a narrow, contorted margin; cheaper violations force a wider, simpler boundary. Reason: C is exactly the overfitting dial: high C makes the hinge term dominate so the boundary chases individual points; low C lets the regularizer smooth it. Likely follow-up: *"And if both train and test accuracy are low?"* — that's underfitting: raise C, or move from a linear kernel to RBF because the boundary you need probably isn't linear.

**4. Gotcha — "Adding more training data far from the boundary will improve my SVM, right?"**
Direct answer: no — points that clear the margin have zero hinge loss and zero gradient; they don't move the solution at all. Reason: the boundary is determined solely by the support vectors, so new easy points are informationally dead weight (they do make training slower, though). Likely follow-up: *"What data would help?"* — points near the decision boundary; this is why active learning with SVMs queries the points closest to the current hyperplane.

**5. System design — "Build a spam filter for 10M emails/day. Would you use an SVM?"**
Direct answer: yes — a *linear* SVM on hashed n-gram/TF-IDF features, trained with Pegasos-style SGD, served as a single dot product per email. Reason: text is high-dimensional and close to linearly separable, linear SVMs are a proven top performer there, and inference is one sparse dot product — microseconds at any scale; an RBF kernel is off the table because serving cost scales with support-vector count. Likely follow-up: *"How do you handle drift as spammers adapt?"* — periodic or online retraining on fresh labels, hashing to keep the feature space stable, and monitoring the margin distribution: a shrinking average margin on incoming traffic is an early-warning signal before accuracy visibly drops.
