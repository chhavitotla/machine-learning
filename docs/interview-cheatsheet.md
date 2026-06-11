# The ML Interview Cheatsheet

One page. Read it on the train. Every claim here is expanded with a visualization in the topic folders.

## Algorithm table

| Algorithm | Best for | Fails when | Key hyperparams | One-line intuition |
|---|---|---|---|---|
| **Linear regression** | Interpretable baselines, effect estimation | Nonlinear patterns, outliers (squared loss), correlated features | learning rate; L1/L2 strength | Best straight line through the data, "best" = least squared error |
| **Logistic regression** | Calibrated probabilities, high-dim sparse data (text) | Nonlinear boundaries; perfectly separable data (weights blow up unregularized) | C (inverse reg. strength) | Linear regression squashed through a sigmoid into a probability |
| **Decision tree** | Mixed feature types, need a human-readable model | Deep trees memorize noise; tiny data changes restructure the whole tree | max_depth, min_samples_leaf | A learned flowchart of if/else questions |
| **Random forest** | Strong tabular default with near-zero tuning | Need extrapolation beyond training range; tight memory/latency budgets | n_estimators, max_features | Many decorrelated trees out-voting each other's mistakes |
| **Gradient boosting (XGBoost/LightGBM)** | Winning on tabular data, competitions, ranking | Heavy label noise (it will fit it); needs careful tuning vs. forests | learning_rate, n_estimators, max_depth | Each new tree predicts the errors of the ensemble so far |
| **SVM** | Medium-sized, high-dimensional, clean data | Large datasets (kernel = O(n²)+); noisy overlapping classes; no native probabilities | C, kernel, gamma | The widest-margin boundary, defined only by the borderline points |
| **K-Means** | Fast segmentation when clusters are roundish and K is known | Non-spherical clusters, unequal densities, unknown K, outliers | K, n_init | Drop K pins, let each drift to the center of its crowd |
| **DBSCAN** | Arbitrary-shape clusters, automatic outlier detection | Clusters of very different densities; high dimensions (distances degrade) | eps, min_samples | Clusters are dense neighborhoods; loners are noise |
| **PCA** | Decorrelation, compression, visualization, denoising | Nonlinear structure (manifolds); when rare-but-important variance gets cut | n_components | Rotate to the axes where the data actually varies, drop the quiet ones |
| **Feed-forward net (MLP)** | Learned interactions on dense mid-size data | Small data (overfits), tabular data (boosting usually wins) | width, depth, lr, dropout | Stacked linear maps + nonlinearities = universal function approximator |
| **CNN** | Images, audio, anything with local spatial structure | Global/long-range relations; rotations it never saw (no built-in invariance) | filters, kernel size, stride | Slide small pattern detectors; stack them into a hierarchy |
| **RNN / LSTM** | Streaming/online sequences, modest-length dependencies | Long sequences (vanishing signal), no parallel training over time | hidden size, seq length | A loop with memory: read input + own past, write new state |
| **Transformer** | Language, code, any sequence at scale; transfer learning | Small data from scratch; O(n²) cost in context length | layers, heads, d_model, context len | Every token attends to every other token in one step |

## The 20 questions that actually come up

**1. Bias-variance tradeoff — explain it.**
Bias is error from a model too simple to represent the truth; variance is error from fitting noise that changes with each training sample. Total error = bias² + variance + irreducible noise, so you tune capacity/regularization to minimize the sum, not either term alone.

**2. Overfitting vs. underfitting — how do you tell which one you have?**
Compare train and validation error: low train + high val = overfitting; high train ≈ high val = underfitting. The fix is opposite in each case (regularize/simplify vs. add capacity/features), which is why you diagnose before treating.

**3. L1 vs. L2 regularization?**
L1 (lasso) penalizes |w| and drives weights exactly to zero — built-in feature selection; L2 (ridge) penalizes w² and shrinks all weights smoothly toward zero. Geometric reason: the L1 ball has corners on the axes, and optima land on corners.

**4. Precision vs. recall — when do you optimize for which?**
Precision = of my positive predictions, how many were right (optimize when false positives are costly — spam filters, fraud accusations); recall = of the true positives, how many I caught (optimize when misses are costly — cancer screening, safety checks). State the costs of each error type first; the metric choice falls out.

**5. What does AUC actually measure?**
The probability that a randomly chosen positive example gets a higher score than a randomly chosen negative one — threshold-free ranking quality. It can mislead under heavy class imbalance, where precision-recall curves are more informative.

**6. Why accuracy fails on imbalanced data, and what to use?**
With 99% negatives, "always predict negative" scores 99% accuracy while catching nothing. Use precision/recall/F1 or PR-AUC, and consider resampling or class weights in training.

**7. How does gradient descent work, and what does the learning rate do?**
Compute the gradient (direction of steepest loss increase), step the opposite way, repeat. Learning rate is the step size: too small = slow convergence, too large = oscillation or divergence; it's the single most important hyperparameter.

**8. SGD vs. Adam — when each?**
Adam adapts a per-parameter step size from gradient history, making it robust to learning-rate choice and great for transformers/sparse gradients; tuned SGD+momentum sometimes generalizes slightly better in vision and is cheaper in memory. Default to AdamW; switch to SGD when you have the budget to tune and will train a CNN once and serve it forever.

**9. What is backpropagation, in two sentences?**
It's the chain rule applied backwards through the network, computing the loss gradient for every weight in one backward pass by reusing each layer's error signal. The reuse (dynamic programming) is the point — cost is proportional to network size, not to the number of paths.

**10. Vanishing/exploding gradients — cause and fixes?**
The gradient at early layers is a product of ~depth many Jacobians; norms consistently below 1 shrink it to zero, above 1 blow it up. Fixes: ReLU-family activations, He/Xavier init, residual connections, normalization layers, gradient clipping (and gating, for RNNs).

**11. Why do we need activation functions?**
Without them, stacked linear layers collapse to a single linear map — a 100-layer network with no nonlinearity is just one matrix. The nonlinearity is what lets depth create new representational power.

**12. What does dropout do, and why does it work?**
Randomly zeroing units during training prevents co-adaptation — no neuron can rely on a specific partner, so the network learns redundant, robust features. It's roughly an implicit ensemble of exponentially many subnetworks sharing weights.

**13. Batch norm — what problem does it solve?**
Activations drifting in scale/distribution mid-training make every layer chase a moving target; normalizing each batch to zero mean/unit variance (then learnable rescale) keeps layer inputs stable, allowing higher learning rates and faster convergence. At inference it uses running statistics, not batch statistics — a classic source of train/serve bugs.

**14. How does attention work, in three sentences?**
Each token emits a query ("what I'm looking for"), a key ("what I contain"), and a value ("what I'll contribute"). Each query is dotted against all keys, scaled by √d_k, softmaxed into weights summing to 1. The token's new representation is the weight-blended sum of all values — content-based routing, learned end to end.

**15. Why did transformers beat RNNs?**
Constant path length between any two positions (no signal decay across distance) and full parallelism over the sequence during training. RNNs process serially and squeeze history through a fixed-size state; transformers pay O(n²) compute for the privilege of not doing that.

**16. What is cross-validation and when is a plain split not enough?**
K-fold: train K times, each fold taking a turn as validation, average the scores — every point gets validated exactly once. Needed when data is small (a single split is noisy) or when you're comparing models/hyperparameters and need error bars; use time-based splits for temporal data, grouped splits when rows share an entity.

**17. What is data leakage and what's the classic example?**
Information from outside the training timeframe/fold sneaking into features, producing great offline metrics and a dead model in production. Classics: normalizing using statistics computed on the full dataset before splitting, or features that encode the future (e.g., "account closed date" when predicting churn).

**18. How do you handle class imbalance?**
In order of preference: get more minority data; reweight the loss (cheap, no information thrown away); resample (SMOTE/undersampling); move the decision threshold using the actual cost ratio. And evaluate with PR metrics — see #6 — or the "fix" is invisible.

**19. Generative vs. discriminative models?**
Discriminative models learn p(y|x) directly — the boundary (logistic regression, standard nets); generative models learn p(x, y) or p(x) — the data itself (Naive Bayes, VAEs, LLMs). Generative can synthesize data and handle missing inputs; discriminative usually wins at pure classification accuracy.

**20. Walk me through an ML system design (the 60-second skeleton).**
Clarify the product goal and turn it into an ML objective + metric (online and offline); then data (sources, labels, freshness, leakage), features, model choice (baseline first, justify complexity), training pipeline (splits, retraining cadence), serving (latency, throughput, fallback), and monitoring (drift, feedback loops, A/B testing). Saying "baseline first" and "monitoring" out loud is most of the senior signal.

---

*Each row of the table links to a folder in this repo with the math, an interactive visualization, and a from-scratch implementation. If you have 60 minutes, the [README](../README.md) has the ordered route.*
