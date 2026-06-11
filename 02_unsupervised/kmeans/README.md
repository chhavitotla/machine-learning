# K-Means Clustering

## The Intuition

You have a cloud of unlabeled points and a hunch that it's secretly several groups — customer types, song genres, cell populations. K-means finds the groups with a beautifully dumb two-step dance: drop K "centroids" anywhere, then (A) let every point join its nearest centroid and (B) let every centroid move to the center of its flock. Repeat. Each step can only make the clusters tighter, so the dance always settles — usually in a handful of iterations. The catch is that "settles" means a *local* optimum: where the centroids start decides where they end, which is why real systems run several random starts and keep the best. There's no teacher anywhere in this loop — the structure was in the distances all along.

## The Math

The objective (inertia, or within-cluster sum of squares):

$$J = \sum_{i=1}^{n} \lVert x_i - \mu_{c(i)} \rVert^2$$

- $x_i$ — data point $i$
- $c(i)$ — the cluster point $i$ is assigned to
- $\mu_k$ — the centroid of cluster $k$

This measures total squared distance from every point to its cluster's center — smaller means tighter clusters.

The assign step:

$$c(i) \leftarrow \arg\min_k \lVert x_i - \mu_k \rVert^2$$

With centroids frozen, the best assignment for each point is simply its nearest centroid — anything else would add distance.

The update step:

$$\mu_k \leftarrow \frac{1}{|C_k|} \sum_{i \in C_k} x_i$$

- $C_k$ — the set of points currently assigned to cluster $k$

With assignments frozen, the mean is the unique point minimizing the sum of squared distances to the members (set the gradient of $\sum_i \lVert x_i - \mu \rVert^2$ to zero and the mean falls out).

Each phase weakly decreases $J$ and there are finitely many assignments, so the loop converges — to a local optimum. Initialization (random restarts, or k-means++, which seeds centroids far apart with probability proportional to $d^2$) decides which one.

## Open the visualization

`open visualize.html in any browser`

## When to use this

- **Customer segmentation** — millions of users described by behavior features, compressed into K personas that marketing can name and target; k-means is the default first cut because it scales linearly in points, clusters, and dimensions.
- **Vector quantization and codebooks** — color palette reduction in images, product quantization for approximate nearest-neighbor search, and KV-cache/embedding compression all use k-means centroids as a learned dictionary.
- **Preprocessing for other models** — cluster IDs or distances-to-centroids as features, or k-means to pick representative samples (coresets) when labeling budget is tight.

## What breaks it

- **Non-spherical clusters** — k-means partitions the plane with straight Voronoi boundaries, so two crescent moons or concentric rings get sliced through the middle no matter how obvious the shape is to your eye. Density-based methods (DBSCAN) exist precisely for this.
- **The wrong K** — ask for 5 clusters in 3-blob data and k-means will cheerfully manufacture 5; it always returns exactly K, real or not. The elbow plot or silhouette score is a heuristic, not an oracle, and stakeholders will reify whatever number you pick.
- **Unscaled features and outliers** — a feature measured in dollars (range 10⁵) drowns one measured in years (range 10¹), so clusters form along income alone; and because the objective squares distances, a single far-flung outlier can hijack a centroid all to itself. Standardize features and consider k-medoids when outliers are endemic.

## 5 Interview Questions

**1. Conceptual — "Why is k-means guaranteed to converge, and to what?"**
Direct answer: both phases monotonically decrease the inertia — assignment picks the nearest centroid (can't increase distance), and the mean minimizes squared distance for a fixed assignment — and there are finitely many possible assignments, so the loop must reach a fixed point. Reason: it's coordinate descent on $J(c, \mu)$, alternating exact minimization over $c$ and over $\mu$. Likely follow-up: *"Converges to the global optimum?"* — no, only local; the global problem is NP-hard, which is why we use random restarts or k-means++ seeding.

**2. Mathematical — "Show that the update step should use the mean and not, say, the median."**
Direct answer: minimize $f(\mu) = \sum_{i \in C_k} \lVert x_i - \mu \rVert^2$; the gradient is $-2\sum_i (x_i - \mu)$, which is zero exactly when $\mu = \frac{1}{|C_k|}\sum_i x_i$ — the mean is the unique minimizer of squared euclidean distance. Reason: squared L2 loss pairs with the mean the same way L1 pairs with the median. Likely follow-up: *"So what algorithm do you get with L1?"* — k-medians (or k-medoids for arbitrary metrics), which is more robust to outliers precisely because the median ignores extreme values.

**3. Practical — "How do you choose K in production?"**
Direct answer: combine the elbow on inertia with silhouette score, but let the downstream use case cap it — if marketing can act on at most 6 segments, K > 6 is academic. Reason: inertia always decreases with K (K = n gives zero), so you're looking for diminishing returns, and silhouette tells you whether clusters are actually separated rather than arbitrary slices. Likely follow-up: *"The elbow is ambiguous — now what?"* — check stability: rerun with different seeds and subsamples; a real K produces the same clusters repeatedly (high adjusted Rand index between runs), a fake one doesn't.

**4. Gotcha — "I ran k-means twice and got completely different clusters. Is the implementation buggy?"**
Direct answer: probably not — k-means is initialization-dependent and only finds local optima, so different random seeds legitimately give different partitions. Reason: the loss surface has many basins; a bad init can strand two centroids in one true blob and split it while merging two others. Likely follow-up: *"How do you make it reproducible and better?"* — fix the seed for reproducibility, and for quality use k-means++ initialization plus n_init=10 restarts keeping the lowest inertia, which is exactly scikit-learn's default behavior.

**5. System design — "Design the clustering stage of a news-app pipeline that groups today's articles into stories."**
Direct answer: embed articles (sentence-transformer vectors), reduce to ~50–100 dims, then run minibatch k-means with K estimated from yesterday's story count, refreshing centroids incrementally as articles stream in. Reason: minibatch k-means handles streaming volume cheaply, embeddings make "same story" approximately spherical in cosine space (normalize vectors so euclidean ≈ cosine), and centroids give you a free story summary — the articles nearest each center. Likely follow-up: *"Stories aren't fixed in number and breaking news appears mid-day"* — exactly; handle it with a distance threshold for "new cluster" spawning (or switch to an online/density method), and merge clusters whose centroids drift within a similarity threshold.
