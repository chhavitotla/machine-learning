# DBSCAN

## The Intuition

K-means asks "where are the centers?" — DBSCAN asks a better question for lumpy data: "where is it *crowded*?" Pick a radius and a headcount. Any point with at least that many neighbors inside its radius is a "core" point — it's standing somewhere dense. Now flood-fill: start at a core point, absorb everything within its radius, and keep expanding through every core point you absorb, the way a rumor spreads through a packed room but dies at the empty hallway. When the flood stalls, that's one cluster; find an untouched core point and start the next. Absorbed-but-not-crowded points sit on the cluster's rim; points the flood never reaches are declared noise and left alone. That last part is quietly radical: DBSCAN is allowed to say "this point belongs to nothing," and it never asks how many clusters to expect. Whatever shapes the dense regions trace — crescents, rings, spirals — the flood follows, because it only takes radius-sized local steps and never assumes anything is round.

## The Math

The ε-neighborhood:

$$N_\varepsilon(p) = \{\, q \in D : \lVert p - q \rVert \le \varepsilon \,\}$$

- $p, q$ — points in the dataset $D$
- $\varepsilon$ (eps) — the radius defining "near"

A point's neighborhood is simply everything within distance ε of it — the ball the flood-fill can step across.

Core, border, and noise:

$$p \text{ is core} \iff |N_\varepsilon(p)| \ge \textit{minPts}$$

- $\textit{minPts}$ — the headcount required to call a spot "dense" (the point itself counts)
- border — not core, but inside some core point's neighborhood
- noise — neither core nor border: reachable from nothing dense

Every point gets exactly one of three labels: crowded (core), on the rim of a crowd (border), or alone (noise).

Density-reachability:

$$q \text{ is density-reachable from } p \iff \exists\, p = p_1, p_2, \ldots, p_k = q : \; p_{i+1} \in N_\varepsilon(p_i) \text{ and } p_i \text{ is core for } i < k$$

- $p_1, \ldots, p_k$ — a chain of radius-sized hops, where every hop *launches from* a core point

A cluster is a maximal set of points connected by chains of core points — only crowded points may extend the chain, which is exactly why the flood dies at sparse gaps; border points can be absorbed but never recruit (that's the asymmetry: core reaches border, never the reverse).

The algorithm:

$$\text{for each unvisited core } p: \text{ seed a cluster, expand via a queue, recursing through every core neighbor}$$

- runtime — $O(n \log n)$ with a spatial index (KD-tree/R-tree) for neighborhood queries, $O(n^2)$ without

One sweep, one flood-fill per cluster; no objective, no iterations, no K — ε and minPts fully determine the answer.

## Open the visualization

`open visualize.html in any browser` — step or run the flood-fill live over switchable datasets (two moons, blobs + noise, ring-in-a-ring) with ε and minPts sliders, click-to-add points, core/border/noise drawn distinctly, and live counters for clusters, noise, visited points, and the expansion queue.

## When to use this

- **Geospatial hotspot detection** — crime concentrations, ride-share demand zones, outbreak foci: geographic clusters follow streets and rivers rather than circles, their count is unknown in advance, and isolated events *should* be ignored as noise.
- **Anomaly detection as a by-product** — in fraud or intrusion pipelines, the noise label is the product: normal behavior forms dense regions, and whatever DBSCAN refuses to cluster is precisely the weird stuff worth a human's attention.
- **Cleaning spatial sensor data** — lidar point clouds, GPS traces, astronomical surveys: separating dense objects from scattered background returns is exactly the core-vs-noise distinction, computed in one pass with no K to guess.

## What breaks it

- **Clusters of very different densities** — one global ε can't serve both: tight enough for the dense cluster, it shatters the sparse one into noise; loose enough for the sparse one, it fuses the dense ones. Often *no* ε works — which is what OPTICS and HDBSCAN were built to fix.
- **High dimensions** — distances concentrate as dimension grows, so everything sits at roughly the same distance from everything else; the dense/sparse contrast DBSCAN feeds on evaporates, ε becomes unsettable, and neighborhood queries lose their index speedup. Above ~10–20 dims, reduce first.
- **ε set by folklore instead of the data** — results are brutally sensitive to ε (the k-distance elbow plot is the honest way to pick it), and unscaled features corrupt the distance underneath it: an ε tuned on raw dollars-and-years data measures income alone, and tiny ε shifts can flip thousands of points between cluster and noise.

## 5 Interview Questions

**1. Conceptual — "Why can DBSCAN find the two-moons clusters when k-means provably can't?"**
Direct answer: k-means assigns each point to its nearest centroid, partitioning space with straight Voronoi boundaries — so any non-convex cluster (a crescent wraps around the other's centroid) gets sliced; DBSCAN never compares to a center, it chains radius-ε hops through crowded points, tracing whatever shape the dense region has. Reason: k-means optimizes a global objective built on cluster *means*, baking in convexity; DBSCAN's cluster definition is purely local connectivity — no objective, no shape prior. Likely follow-up: *"What does DBSCAN give up in exchange?"* — predictability: no objective to monitor, results hinge on ε/minPts, border points are order-dependent, and new points can't be assigned without re-running or nearest-core heuristics.

**2. Mathematical — "Define core, border, and noise precisely, and explain why density-reachability is asymmetric."**
Direct answer: $p$ is core iff $|N_\varepsilon(p)| \ge \textit{minPts}$; border iff not core but $p \in N_\varepsilon(q)$ for some core $q$; noise otherwise. Reachability is asymmetric because each hop must launch *from a core point*: a core reaches a border point in its neighborhood, but the border point — lacking minPts neighbors — can never extend a chain back. Reason: only crowded points may recruit, which is exactly what stops clusters from leaking through a thin bridge of sparse points. Likely follow-up: *"So is cluster membership well-defined?"* — for core points yes (density-*connectedness* via a shared core ancestor is symmetric); border points near two clusters go to whichever expansion arrives first, a known and accepted order-dependence.

**3. Practical — "How do you actually choose ε and minPts on a real dataset?"**
Direct answer: set minPts first from dimension and noise tolerance (rule of thumb: minPts ≈ 2·dims, higher for noisier data), then plot every point's distance to its minPts-th nearest neighbor, sorted — the k-distance plot — and set ε at the elbow where distances bend upward. Reason: points inside true clusters have small k-distances and noise points have large ones, so the elbow is the empirical dense/sparse boundary for your data; guessing ε in raw units has no such grounding. Likely follow-up: *"The elbow is mushy or there are two elbows?"* — that's the signature of multi-density data: run DBSCAN per density regime, or switch to HDBSCAN, which sweeps all ε values and extracts the stablest clusters automatically.

**4. Gotcha — "I standardized my features and DBSCAN now returns one giant cluster instead of the five I had. Bug?"**
Direct answer: no — ε is a distance in feature units, and standardizing rescaled every distance, so your old ε is now far too large relative to the data's spread; neighborhoods balloon, everything becomes core, and the flood-fill connects the whole dataset. Reason: ε and the feature scaling jointly define density, so any change to scaling silently invalidates a tuned ε. Likely follow-up: *"So should I not standardize?"* — you almost always should (otherwise the largest-range feature dominates the metric); the rule is scale *first*, then tune ε via the k-distance plot on the scaled data — never reuse an ε across preprocessing changes.

**5. System design — "Design the clustering stage for a ride-share app that detects surge-demand hotspots from live GPS pings."**
Direct answer: maintain a sliding window (last ~10 minutes) of pickup-request coordinates per city, index them in a spatial grid or R-tree, and run DBSCAN every minute or two with ε ≈ a few hundred meters and minPts set to the request volume that justifies a surge; clusters become surge polygons via the hull of member points, and noise pings are correctly ignored as scattered demand. Reason: hotspot count is unknown and varies hourly (k-means' K would be a constant lie), hotspots are street-shaped rather than circular, and isolated requests must not trigger surges — all three are DBSCAN's native strengths, and the spatial index keeps each run near $O(n \log n)$. Likely follow-up: *"Downtown is 50× denser than the suburbs — one ε can't serve both."* — correct; shard by zone with per-zone (ε, minPts) calibrated from historical k-distance plots, or set minPts proportional to local baseline demand — or run HDBSCAN if compute allows, trading speed for density-adaptivity.
