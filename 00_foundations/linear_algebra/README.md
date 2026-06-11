# Linear Algebra

## The Intuition

Forget the spreadsheet-of-numbers picture of a matrix. A matrix is a *machine that moves space*. Feed it a vector — an arrow from the origin — and it hands back a new arrow. Feed it *every* point in the plane and the whole plane stretches, rotates, shears, or flips, all at once, all linearly: grid lines stay straight, parallel, and evenly spaced. A 2×2 matrix is fully described by where it sends just two arrows, $(1,0)$ and $(0,1)$ — those landing spots are literally its columns. Everything else follows for free. The determinant tells you how much area got scaled (and a negative sign means space got mirror-flipped); eigenvectors are the rare stubborn directions the matrix only stretches, never turns. This is the bedrock of ML because *every* layer of a neural network starts with exactly this: a matrix warping its input space, hunting for an orientation where the classes become separable.

## The Math

The transformation:

$$M\mathbf{v} = \begin{bmatrix} a & b \\ c & d \end{bmatrix}\begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} ax + by \\ cx + dy \end{bmatrix}$$

- $M$ — the 2×2 matrix: the transformation itself
- $\mathbf{v} = (x, y)$ — the input vector being moved
- $(a, c)$ — the first column: where the unit arrow $(1,0)$ lands
- $(b, d)$ — the second column: where the unit arrow $(0,1)$ lands

This says: the output is $x$ copies of where the first basis arrow went, plus $y$ copies of where the second one went — a recipe, not a calculation.

The determinant:

$$\det(M) = ad - bc$$

- $\det(M)$ — the signed factor by which $M$ scales area

The unit square becomes a parallelogram with area $|ad - bc|$; if the sign is negative, the transformation flipped space like a pancake, and if it's zero, the whole plane got crushed onto a line (or a point) and can never be un-crushed.

The inverse:

$$M^{-1} = \frac{1}{ad - bc}\begin{bmatrix} d & -b \\ -c & a \end{bmatrix}, \qquad M M^{-1} = I$$

- $M^{-1}$ — the transformation that exactly undoes $M$
- $I$ — the identity matrix: the do-nothing transformation

Swap the diagonal, negate the off-diagonal, divide by the determinant — and note the division: when $\det(M) = 0$ there is no inverse, because you can't undo a collapse.

Eigenvectors and eigenvalues:

$$M\mathbf{v} = \lambda \mathbf{v}$$

- $\mathbf{v}$ — an eigenvector: a direction $M$ does not rotate
- $\lambda$ — its eigenvalue: the factor by which that direction gets stretched

For a 2×2, solve the characteristic polynomial $\lambda^2 - (a+d)\lambda + \det(M) = 0$; when its discriminant is negative the eigenvalues are complex, which geometrically means the matrix rotates *everything* and no direction survives unturned.

## Open the visualization

`open visualize.html in any browser`

## When to use this

- **Every neural network layer** — `Wx + b` is a matrix warping the input space; understanding "matrices move space" is understanding what depth actually buys you (a stack of warps that untangles the data).
- **PCA and dimensionality reduction** — the eigenvectors of the covariance matrix are the directions your data actually varies along; projecting onto the top few compresses 1000 features into 10 with minimal loss.
- **Embeddings and similarity search** — dot products measure directional agreement, which is why "cosine similarity between embedding vectors" powers every recommendation and retrieval system you've used today.

## What breaks it

- **Singular and near-singular matrices** — if $\det \approx 0$, inverting amplifies tiny numerical noise into enormous garbage; this is exactly why the normal equation $(X^TX)^{-1}X^Ty$ explodes when features are nearly collinear.
- **Assuming linearity where there is none** — a matrix keeps grid lines straight and parallel, so no matrix alone can separate concentric circles; that's why networks must interleave nonlinearities between the matrix layers.
- **Repeated multiplication amplifying or killing signal** — applying a matrix many times grows everything along the largest eigenvalue and crushes the rest; in recurrent networks this is literally the exploding/vanishing gradient problem, eigenvalues wearing a trench coat.

## 5 Interview Questions

**1. Conceptual — "What does a matrix *do*, geometrically?"**
Direct answer: it's a linear transformation of space — every vector gets moved such that grid lines remain straight, parallel, and evenly spaced, and the origin stays put; the columns are exactly where the basis vectors land. Reason: any vector is a combination of basis vectors, and linearity means the output is the same combination of the *transformed* basis vectors, so two landing spots determine everything. Likely follow-up: *"So what does matrix multiplication mean?"* — composition of transformations: $AB$ means "apply $B$, then $A$," which is why it's associative but famously not commutative (shear-then-rotate ≠ rotate-then-shear).

**2. Mathematical — "Derive the condition for a 2×2 matrix to have real eigenvalues."**
Direct answer: eigenvalues solve $\det(M - \lambda I) = 0$, which for a 2×2 expands to $\lambda^2 - (a+d)\lambda + (ad - bc) = 0$; the roots are real iff the discriminant $(a+d)^2 - 4(ad-bc) \geq 0$, i.e. $\text{tr}(M)^2 \geq 4\det(M)$. Reason: it's just the quadratic formula applied to the characteristic polynomial — trace is the sum of eigenvalues, determinant is their product. Likely follow-up: *"Name a matrix with no real eigenvectors"* — any rotation by an angle that isn't 0° or 180°: it turns every direction, so no vector keeps its line, and the eigenvalues come out complex.

**3. Practical — "Your model has a 10,000×10,000 weight matrix. How do you find its dominant eigenvector without an eigensolver?"**
Direct answer: power iteration — start with a random vector, repeatedly multiply by the matrix and normalize; it converges to the dominant eigenvector at a rate of $|\lambda_2/\lambda_1|$ per step, costing only matrix-vector products. Reason: writing the start vector in the eigenbasis, each multiplication scales each component by its eigenvalue, so the largest one wins exponentially fast. Likely follow-up: *"Where is this used in production?"* — spectral norm regularization in GANs, PageRank (the web's dominant eigenvector), and estimating the Lipschitz constant of network layers.

**4. Gotcha — "det(M) = 0. Can you still solve Mx = y?"**
Direct answer: not uniquely, and possibly not at all — the matrix collapses the plane onto a line, so $y$ either lies off that line (no solution) or on it (infinitely many solutions); what you can do is pick the minimum-norm solution via the pseudoinverse. Reason: a determinant of zero means two input directions map to the same output, so information was destroyed and there's no function that maps back. Likely follow-up: *"How does this show up in ML?"* — perfectly collinear features make $X^TX$ singular, which is precisely why ridge regression adds $\lambda I$: it nudges every eigenvalue away from zero so the inverse exists and is stable.

**5. System design — "Design the similarity-search core of a recommendation system."**
Direct answer: embed users and items as vectors so that the dot product (or cosine) measures affinity; precompute item embeddings as a matrix, and serving a user is one matrix-vector product followed by a top-k — with approximate nearest-neighbor indexing (HNSW/IVF) once the catalog outgrows brute force. Reason: dot products are the cheapest meaningful similarity, they batch into hardware-friendly matrix multiplies, and the geometry (similar things point the same way) is exactly what embedding training optimizes. Likely follow-up: *"The item matrix is 100M × 768 and doesn't fit in RAM — now what?"* — compress with quantization or project to a lower-dimensional space using the top principal components (eigenvectors of the covariance again), trading a little recall for a 10× memory cut.
