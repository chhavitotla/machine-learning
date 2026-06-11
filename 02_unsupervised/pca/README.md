# Principal Component Analysis

## The Intuition

A correlated cloud of points wastes coordinates. If height and weight move together, you don't really have two independent numbers per person — you have one big axis ("overall size") and one small one ("build, given size"). PCA finds those axes automatically: it rotates the coordinate system so the first axis points along the direction of maximum spread, the second along the most spread that's left, and so on. Keep the big axes, drop the small ones, and you've compressed the data while losing as little as mathematically possible — because the direction that *maximizes kept variance* is exactly the direction that *minimizes squared reconstruction error*. Those are the same problem viewed from opposite ends, and that equivalence is the heart of all dimensionality reduction.

## The Math

Center the data, then build the covariance matrix:

$$C = \frac{1}{n} X^T X \qquad \text{(X has the mean subtracted)}$$

- $X$ — the $n \times d$ centered data matrix
- $C$ — the $d \times d$ covariance matrix; entry $C_{jk}$ says how features $j$ and $k$ co-vary

The variance of the data projected onto a unit direction $v$ is:

$$\mathrm{Var}(Xv) = v^T C v$$

Maximizing this over unit vectors is solved by the eigendecomposition:

$$C v_k = \lambda_k v_k$$

- $v_k$ — the $k$-th principal component (an eigenvector, a direction)
- $\lambda_k$ — its eigenvalue: the variance of the data along $v_k$

The top eigenvector is the direction of maximum variance; eigenvectors of a symmetric matrix are orthogonal, so the components form a rotated coordinate system.

For 2D this is closed-form. With $C = \begin{pmatrix} a & b \\ b & c \end{pmatrix}$:

$$\lambda_{1,2} = \frac{(a+c) \pm \sqrt{(a+c)^2 - 4(ac - b^2)}}{2} \qquad v_1 \propto (b,\ \lambda_1 - a)$$

Explained variance ratio and reconstruction error:

$$\text{EVR}_1 = \frac{\lambda_1}{\lambda_1 + \lambda_2} \qquad \text{MSE}_{\text{1 component}} = \lambda_2$$

Keeping PC1 retains $\lambda_1$ of the variance; the squared error of reconstructing from PC1 alone is *exactly* the discarded eigenvalue. Variance maximization and error minimization are the same optimization.

## Open the visualization

`open visualize.html in any browser`

## When to use this

- **Visualization of high-dimensional data** — projecting embeddings, gene-expression profiles, or sensor logs to 2–3 components is the standard first look at whether structure (clusters, gradients, outliers) exists at all.
- **Decorrelation and compression before other models** — PCA whitening removes feature correlation that destabilizes linear models, and keeping 95% of variance often cuts dimensions 10× — cheaper training, less overfitting, smaller indexes.
- **Noise reduction** — if signal lives in a few strong directions and noise is isotropic, truncating small components is a denoiser; this is why PCA preprocessing often *improves* downstream accuracy, not just speed.

## What breaks it

- **Nonlinear structure** — a spiral or swiss-roll has its "true" 1D coordinate wrapped around in space; PCA can only rotate, so it reports two fat eigenvalues and compresses nothing. You need kernel PCA, UMAP, or an autoencoder.
- **Unscaled features** — variance is unit-dependent: a feature in millimeters has 10⁶× the variance of the same feature in meters, and PC1 will dutifully point along whichever column has the biggest units. Standardize first (correlation-matrix PCA) or the components are an artifact of your unit choices.
- **Variance ≠ relevance** — PCA is unsupervised: it keeps directions of large variance, but the label might depend on a tiny-variance direction (e.g., a small but decisive sensor offset). Projecting it away guarantees your classifier can never recover it; that's what supervised reduction (LDA) is for.

## 5 Interview Questions

**1. Conceptual — "Why does maximizing variance minimize reconstruction error?"**
Direct answer: for a centered point, the Pythagorean theorem splits its squared norm into (component along the projection direction)² + (perpendicular residual)²; total squared norm is fixed, so the direction keeping maximum variance is identically the one leaving minimum squared residual. Reason: $\sum_i \lVert x_i \rVert^2 = \mathrm{Var}_{\text{kept}} + \mathrm{MSE}_{\text{lost}}$ — one budget, two views. Likely follow-up: *"So what exactly is the reconstruction error of keeping the top k components?"* — the sum of the discarded eigenvalues, $\sum_{j>k} \lambda_j$.

**2. Mathematical — "Show that the variance-maximizing direction is the top eigenvector."**
Direct answer: maximize $v^T C v$ subject to $\lVert v \rVert = 1$ with a Lagrange multiplier: $\nabla_v (v^T C v - \lambda v^T v) = 2Cv - 2\lambda v = 0$, so $Cv = \lambda v$ — every stationary point is an eigenvector, and at an eigenvector the objective equals $\lambda$ itself, so the maximum is the largest eigenvalue's eigenvector. Reason: the constraint stops you from inflating variance by scaling $v$; the multiplier turns the constrained problem into the eigenvalue equation. Likely follow-up: *"Why are the components orthogonal?"* — $C$ is symmetric, and symmetric matrices have orthogonal eigenvectors (spectral theorem).

**3. Practical — "How many components do you keep?"**
Direct answer: plot cumulative explained variance and keep enough for a threshold (90–95%) — but validate against the downstream task: cross-validate the end-to-end pipeline over a few candidate dimensionalities and let task metric decide. Reason: the variance threshold is a proxy; the real question is whether the discarded directions carried task-relevant signal, which only the downstream metric can answer. Likely follow-up: *"The scree plot has no elbow"* — that means variance is spread thin (near-isotropic data); PCA buys little, and you should question whether linear compression is the right tool at all.

**4. Gotcha — "My teammate ran PCA, then standardized the components, then trained — and separately standardized the raw features *after* fitting PCA on unstandardized data. Same thing?"**
Direct answer: no — PCA on unstandardized data is dominated by whichever features have the largest units, and no post-hoc scaling of the components undoes that, because the *directions* were already chosen wrong. Reason: scaling and PCA don't commute; covariance-PCA and correlation-PCA give genuinely different eigenvectors. Likely follow-up: *"When would you deliberately skip standardization?"* — when features share meaningful physical units (pixel intensities, spectral bands) and their natural variances *are* the signal.

**5. System design — "You serve 768-dim text embeddings to a vector database and costs are exploding. Walk me through using PCA."**
Direct answer: fit PCA on a representative sample (a few hundred thousand vectors is plenty), keep ~128 dims at 90%+ variance, store the projection matrix as a versioned artifact, and project at both index time and query time — recall typically drops a point or two while index memory and latency drop ~6×. Reason: embedding variance is heavily concentrated in the top directions, and PCA is a single matrix multiply at inference — negligible serving cost. Likely follow-up: *"What operational trap should you watch for?"* — train/serve skew: if the embedding model is updated, the old projection matrix is silently wrong for new vectors; version the PCA with the encoder and re-fit on model upgrades, and monitor recall against a held-out ground-truth set.
