# The ML System Design Template

Print this. The structure works on any prompt — recommender, fraud, search, churn, anything. The timings assume a 45-minute interview; if yours is 60, stretch modeling and serving, never clarification.

**The clock:**

| Minutes | Phase |
|---|---|
| 0–5 | Clarify the problem |
| 5–10 | Metrics (online + offline) |
| 10–20 | Data, labels, features |
| 20–30 | Modeling (baseline → v2) |
| 30–40 | Serving + monitoring |
| 40–45 | Wrap: risks, iteration plan |

One meta-rule before the sections: **narrate your trade-offs.** "I'm choosing X over Y because Z, and here's what would change my mind" is the sentence shape of every strong answer below. Silent correct decisions score lower than spoken ones.

---

## 1. Clarify the problem (5 min)

Do not design anything yet. Ask, out loud:

- "What's the business goal in one sentence — what number does the company want to move?"
- "What's the scale? Users, items, requests per second?"
- "What's the latency tolerance — is this inline in a request, or can it be batch?"
- "What exists today — a heuristic, a v1 model, nothing? What's it bad at?"
- "Which mistake is worse here — false positive or false negative — and what does each cost?"
- "Any constraints I should know — privacy, regulation, explainability, budget?"

Then say back the problem as an ML statement: *"So: predict P(click | user, item, context) for ~10M DAU at ~5k QPS, ranking 500 candidates in under 150ms, optimizing for sessions per user, not raw clicks."* Getting the interviewer to nod at that sentence is the checkpoint for this phase.

> **Weak:** starts drawing the architecture within 60 seconds.
> **Strong:** "Before I pick anything — which error is more expensive, and what's the latency budget?"

**Senior signals:** asking about the *cost asymmetry* of errors; asking what the current system is so you're designing the delta, not greenfield.

## 2. Metrics (5 min)

Two metrics, always, and the gap between them:

- **Offline** (what training optimizes / how you compare models pre-launch): AUC or log loss for classification, NDCG/MRR for ranking, recall@k for retrieval.
- **Online** (what the A/B test is judged on): retention, dollars of fraud prevented, conversion, time-to-resolution.
- Say where they diverge: "Offline CTR AUC can improve while we ship clickbait — so the launch gate is the online retention metric, not the offline one."
- Name the **guardrail metrics**: latency p99, complaint rate, false-positive rate on the protected segment — things v2 must not regress even while the target metric improves.

> **Weak:** "I'd use accuracy." (On imbalanced data this is an instant down-level.)
> **Strong:** "Offline I'd track PR-AUC since positives are ~0.1%; online, dollars of fraud caught per dollar of declined-legitimate revenue — and the threshold comes from that cost ratio, not from 0.5."

**Senior signals:** naming the offline–online gap unprompted; deriving the classification threshold from error costs instead of defaulting to 0.5.

## 3. Data, labels, features (10 min)

Questions to ask out loud:

- "Where do labels come from — explicit (reports, ratings) or implicit (clicks, purchases)?"
- "What's the label delay?" (Clicks: seconds. Chargebacks: 30–90 days. Churn: a billing cycle. This determines your retraining cadence and your validation design.)
- "What's the feedback loop — does the current system decide what gets observed?" (Logged data only contains outcomes for items the old system chose to show. Train naively on it and you inherit and amplify its biases.)
- "How will I split? Time-based for anything temporal — random splits leak the future."

Then features, in four groups — user, item, context, interaction — picking 2–3 concrete examples per group rather than reciting twenty. Flag, unprompted: "every feature here must be knowable at prediction time and computed identically online and offline — I'd put these in a feature store with point-in-time joins." (Full treatment: [feature_engineering.md](feature_engineering.md).)

> **Weak:** "I'd collect a lot of features and let the model figure it out."
> **Strong:** "Labels are chargebacks at 60-day delay, so today's training data is two months stale — I'd train on mature labels but monitor on early proxies like declined-auth rate."

**Senior signals:** "here's my leakage check — for each feature: could we have known this value at the moment of prediction?"; raising the feedback loop before the interviewer does; "I'd log the served feature vectors so training data matches serving by construction."

## 4. Modeling (10 min)

The order is non-negotiable: **baseline first.**

- **v0 — the no-ML baseline:** popularity ranking, a rules list, "most recent first." Say why: it ships in days, it forces the logging/eval/serving plumbing to exist, and it's the floor every model must beat to justify its complexity.
- **v1 — the simple model:** logistic regression or gradient boosting on the features from section 3. For most tabular problems this captures 80%+ of the achievable lift.
- **v2 — the justified upgrade:** two-tower for retrieval, a deep ranker, a fine-tuned LLM — *named together with the evidence that would justify it*: "if GBM beats LR by less than a point of AUC, we keep LR and its 2ms CPU inference."
- State how models compete: backtest on the most recent time slice, then shadow, then A/B. Offline wins are necessary, not sufficient.

> **Weak:** "I'd fine-tune a transformer on the interaction data." (First sentence, no baseline — the classic mid-level tell.)
> **Strong:** "Logistic regression first — it sets the floor and debugs the pipeline. The model is the easiest part of this system to swap later; the data design isn't."

**Senior signals:** "baseline first"; tying model choice to the latency budget from section 1 ("a 200ms ranker doesn't fit a 100ms budget no matter how good its AUC is"); naming the *evidence threshold* for added complexity.

## 5. Serving + monitoring (10 min)

**Serving — design backwards from the latency budget:**

- State the budget and shape the architecture to it: ranking 1,000 items in 100ms means a cheap retrieval stage (ANN lookup, ~10ms) feeding an expensive ranking stage on the top ~200.
- Batch vs. real-time scoring: churn scores can be computed nightly and cached; fraud must be inline. Choose per-problem and say why.
- Features at serving: online store lookups (<10ms), with the staleness trade-off ("the 7-day aggregate can be an hour stale; the 5-minute velocity counter cannot").
- **The fallback, always:** "if the model service times out, serve the popularity baseline / auto-approve under $50 with async review. Never serve nothing."
- The rollout path: **offline backtest → shadow deploy (score live traffic, log, don't act) → 1% A/B → ramp.**

**Monitoring — three layers:**

1. **System:** latency p99, error rate, feature-store staleness.
2. **Prediction:** score distribution drift, per-feature null rates and out-of-vocabulary rates — your earliest warning, because labels lag.
3. **Business:** the online metric, **per segment** (new users, each platform) — topline averages hide segment failures.

Plus the feedback-loop control: keep a small exploration slice (1–5% randomized or epsilon-greedy) so the model keeps seeing outcomes for things it wouldn't have chosen, and future training data isn't purely its own echo.

> **Weak:** "Deploy it behind an API and retrain monthly."
> **Strong:** "I'd shadow-deploy before ramping — score real traffic for two weeks without acting on it, compare score distributions to backtest, and only then take 1% of decisions."

**Senior signals:** "I'd shadow-deploy before ramping"; "score drift is my early-warning metric because labels arrive 60 days late"; naming the degraded-mode fallback unprompted.

## 6. Wrap (5 min)

Close like someone who has shipped this and watched it break:

- **Top risks:** "The two things I'd worry about: label feedback loop and train/serve skew — here's the mitigation for each."
- **What I'd build first:** "Week 1–2: logging and the baseline. Week 3–4: v1 model and backtest. Week 5: shadow. Week 7: 1% A/B."
- **What I'd measure to know it's working,** and the explicit kill criterion: "if the A/B doesn't move the online metric in two weeks at 5% traffic, we hold the ramp and investigate skew first, model second."
- Offer the deep-dive: "Happy to go deeper on the feature pipeline, the cold-start story, or the retraining cadence — where's most useful?"

> **Weak:** trails off after describing the model.
> **Strong:** ends with risks, a sequenced plan, and a kill criterion — the shape of someone who's been on call for a model.

---

## The senior-signal phrases, collected

Say these where they're true, and mean them:

- "Baseline first — the model is the easiest part to swap later."
- "Here's my leakage check: is every feature knowable at prediction time?"
- "Offline metric and online metric, and here's where they diverge."
- "The threshold comes from the cost ratio of the two errors, not from 0.5."
- "I'd shadow-deploy before ramping."
- "Time-based splits — random splits leak the future."
- "What's the fallback when the model service is down?"
- "Score-distribution drift is my early warning, because labels are delayed."
- "I'd keep an exploration slice so we're not training on our own echo."
- "Per-segment metrics — the topline hides who the model is failing."

*Now watch it run five times: [interview_scenarios.md](interview_scenarios.md).*
