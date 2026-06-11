# Feature Engineering

Models are interchangeable; features are where systems are actually won and lost. This is also where interview deep-dives go when the interviewer wants to find out if you've shipped anything — every section below ends in a production failure mode, because that's the level the questions get asked at.

## Numerical transforms

**Standardization (z-score) and min-max scaling.** Linear models, SVMs, neural nets, and anything distance-based (k-means, kNN) need features on comparable scales, or the feature measured in cents dominates the one measured in years purely by unit choice. Tree models don't care — splits are rank-based — so don't burn interview time scaling features for XGBoost. The rule that matters: **fit the scaler on the training split only**, then apply those frozen parameters to validation, test, and production. Fitting on the full dataset is the most common leakage bug in existence (bug #1 below).

**Log transform.** For anything spanning orders of magnitude with a long right tail — income, transaction amounts, view counts, time-since-event. `log(1+x)` turns "this user has 4M followers" from an outlier that dominates the gradient into a point three units right of the median. Use it when the *ratio* matters more than the difference: $10 → $100 should feel like $1,000 → $10,000. Don't use it on features that are already roughly symmetric, or on anything that can be negative.

**Binning.** Chop a continuous feature into buckets (age → 18–25, 26–35, ...) and one-hot the buckets. You're spending resolution to buy two things: a linear model can now learn non-monotonic effects (risk that's high for the young *and* the old), and the feature becomes robust to outliers. Quantile bins beat equal-width bins on skewed data — equal-width on income puts 99% of users in bucket one. Skip binning for trees; they bin internally and better.

## Categorical encodings

The decision is driven almost entirely by **cardinality** — how many distinct values.

**One-hot** — up to ~50 categories. Honest, no assumptions, every model handles it. Beyond that, your feature matrix bloats and rare categories get one positive example each, which is just noise wearing a column.

**Target encoding** — ~50 to ~100k categories (zip codes, merchant IDs). Replace each category with the mean label for that category: zip 94110 → 0.23 historical conversion rate. One dense, highly informative column.

Here's the trap, properly: if a row's own label contributes to the encoding of its own category, you have leaked the label into a feature. The pathological case makes it obvious — a category with one row encodes as *exactly its own label*, and the model learns "predict the encoding" and posts a spectacular validation AUC. Then production arrives, labels aren't available at prediction time (of course — they're what you're predicting), and the model is blind. The fixes: compute encodings out-of-fold (each fold's encodings come from the other folds), or use only data from *before* each row's timestamp, and always blend rare categories toward the global mean (smoothing), because a merchant with 3 transactions has no business having its own confident statistic.

**Embeddings** — 100k+ categories (user IDs, item IDs, query strings), and only when you have enough interaction data to train them. Learn a dense vector per category as part of the model; similar categories end up nearby, which is something one-hot can structurally never give you. Cost: you need a deep-learning serving stack, an embedding table that can run to gigabytes, and a story for new IDs that have no vector yet (the cold-start default row).

**Hashing** — the budget escape hatch at any cardinality: hash the category into a fixed number of buckets. Collisions are the price; no vocabulary file to maintain is the prize. Good for fast-growing vocabularies where maintaining an ID map is its own outage risk.

## Text and sequences

For non-deep-learning systems: TF-IDF over word or character n-grams into logistic regression remains a brutally strong baseline for classification — spam, ticket routing, query categorization — and serves in microseconds. Character n-grams survive typos and code-switching better than word tokens.

When meaning matters more than keywords ("my card got eaten" ≈ "ATM retained my card"), move to pretrained sentence embeddings: encode the text into a ~384–768-dim vector and feed it downstream as features. For behavioral sequences (last N items viewed, last N transactions), the baseline is windowed aggregates over the sequence (below); the upgrade is feeding the sequence itself into a sequence model. Say the baseline first.

## Time features and windowed aggregates

Raw timestamps are useless; what's predictive is **position in a cycle** and **recency**. Hour-of-day and day-of-week, encoded as sin/cos pairs so 23:00 and 01:00 are neighbors instead of opposite ends of a line. Time-since-last-event (log-transformed — see above) is consistently a top-five feature in churn, fraud, and ranking systems.

Windowed aggregates are the workhorse of production tabular ML: count/sum/mean of an entity's events over the last 1 hour, 24 hours, 7 days, 30 days. "Transactions in the last hour vs. user's 30-day daily average" is a fraud feature; "sessions in the last 7 days vs. the prior 30" is a churn feature. Two things must be true: the window must end **strictly before prediction time** (not "today", which silently includes the future during batch training — bug #2), and the aggregation must be computed by the same logic online and offline (the skew section below).

## Feature crosses

A linear model can learn "Android users convert less" and "Japan converts more" but not "Android users *in Japan* convert more" — that's an interaction, and a cross feature (`country × platform` as one categorical) hands it the interaction explicitly. Crosses are how linear models stayed competitive in ads ranking for a decade. The cost is multiplicative cardinality: crossing two 1,000-value features makes a million-value feature, so crosses get hashed (Vowpal Wabbit's whole trick) or you let a GBM/neural net find interactions on its own. In an interview: propose crosses for the linear baseline, note that the v2 model learns them implicitly.

## Missing values, honestly

The dirty secret: **missingness is usually signal, not noise.** Income is missing because the user declined to share it. A sensor reading is missing because the device was off. "Not missing at random" is the norm in product data, and mean-imputing destroys exactly that signal while inventing a fake observation.

The honest playbook:

1. **Add a `was_missing` indicator column** alongside any imputation, always. It's free and it preserves the signal.
2. **Impute boringly** — median for numericals, a literal `"MISSING"` category for categoricals. Fancy imputation (model-based) adds a model to maintain and rarely beats median + indicator.
3. **Or let the trees handle it** — XGBoost/LightGBM route missing values to whichever side of each split reduces loss, which is effectively a learned per-split indicator.
4. **Ask why it's missing.** If a feature is missing in training but will always be present in production (or the reverse), the training distribution is lying to you and no imputation fixes that.

## Train/serve skew, and why feature stores exist

The training pipeline computes `user_7d_txn_count` in a nightly Spark job from the warehouse. The serving path computes "the same" feature in a Java service from a Redis counter. They differ — timezone of the window boundary, whether refunds count, null vs. zero for new users — and now the model was trained on one function and serves on another. Offline AUC 0.85, online performance mysteriously mediocre, and nothing crashes, which is what makes skew the most expensive bug class in production ML: **it fails silently.**

A **feature store** exists to make the skew structurally impossible: define each feature's computation *once*, and the store materializes it both ways — batch tables for training, a low-latency online lookup (typically <10ms) for serving — plus point-in-time-correct training joins, so each training row sees feature values **as they stood at that row's timestamp**, not as they stand today. That point-in-time join is the unglamorous feature that prevents half the leakage bugs below. The interview phrasing: "I'd define features once in a feature store so training and serving share the definition, and use point-in-time joins so training rows can't see the future."

Cheap alternative if "feature store" sounds too heavy for the problem: log the feature vector at serving time and train on the logged vectors. What the model saw is what the model trains on, by construction.

## The five classic feature bugs

Each with the production symptom — because that's how you actually meet them.

**1. Leakage via global statistics.** Scaler, vocabulary, or target encoding fit on the *full* dataset before splitting, so test rows influenced the transform applied to training rows. **Symptom:** validation metrics a few points better than they should be, then a quiet, permanent gap when the model meets genuinely new data. Often survives for months because nothing crashes — the model is just consistently worse than its offline numbers promised, and the team blames "drift."

**2. Future information.** A feature whose value is computed *after* the moment of prediction: a "30-day activity" window that includes the prediction day, a `days_to_delivery` field populated post-delivery, account fields backfilled by a batch job. **Symptom:** offline metrics that look too good to be true (AUC 0.97 on a hard problem — be suspicious, not proud), then a production model that performs near-randomly, because at serving time the future hasn't happened yet and the feature is null or stale.

**3. Label proxies.** A feature that is causally downstream of the label: `customer_service_calls_about_cancellation` when predicting churn, `chargeback_flag` when predicting fraud, `time_on_page` when predicting click. The model learns to read the answer key. **Symptom:** one feature with absurd importance (60%+ of gain) and a model that is effectively that single column; in production the feature is empty at prediction time and the model collapses. Audit rule: top-importance features get asked "could a human know this *before* the outcome?"

**4. Drifting vocabularies.** The categorical world moves — new merchants, new content categories, renamed product SKUs, a client app version string that changes format — but the training-time vocabulary is frozen. Every new value maps to `UNK` or, worse, to a recycled integer ID that used to mean something else. **Symptom:** slow performance decay between retrains, sharp drops after upstream releases, and a climbing `UNK` rate that nobody alerts on. Monitor the fraction of out-of-vocabulary values per feature; alert when it climbs.

**5. Inconsistent nulls.** Training data encodes missing as `NaN`; the serving path sends `0`, `-1`, `""`, or omits the key entirely, and the deserializer defaults it. The model learned "NaN → route right at this split" and now never takes that path; meanwhile `0` looks like a legitimate observed zero. **Symptom:** a model that's fine on average but badly wrong on exactly the segment with missing data (new users, old app versions) — visible only in per-segment metrics, invisible in the topline. The defense is a feature-validation layer that compares the serving null/zero/distribution profile per feature against the training profile, and pages when they diverge.

---

*The template ([ml_system_design_template.md](ml_system_design_template.md)) tells you when to say all this; the scenarios ([interview_scenarios.md](interview_scenarios.md)) show these bugs appearing as deep-dive follow-ups.*
