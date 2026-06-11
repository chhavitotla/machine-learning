# ML System Design

## What these interviews actually test

Not modeling. The interviewer already assumes you can train a classifier. What they're testing is whether you can take a vague product ask — "make the feed better", "stop fraud", "reduce churn" — and translate it into an ML problem with a defensible shape: a precise prediction target, a metric that actually tracks the business goal, a label strategy that survives contact with reality, and an architecture whose trade-offs you chose on purpose.

The failure mode they're screening for is the engineer who jumps straight to "I'd fine-tune a transformer." The pass signal is the engineer who asks what "better" means, names the metric tension out loud ("optimizing clicks will reward clickbait — do we have a dwell-time signal?"), proposes logistic regression as v1, and explains what they'd watch on a dashboard after launch. Every senior answer is a chain of *trade-off → decision → how I'd know if I was wrong*.

## The three docs in this folder

| Doc | What it's for | When to read it |
|---|---|---|
| [ml_system_design_template.md](ml_system_design_template.md) | The fill-in-the-blanks structure with 45-minute timing, the exact questions to ask out loud, and weak-vs-strong answer contrasts | Memorize the skeleton the night before. It works on any prompt. |
| [interview_scenarios.md](interview_scenarios.md) | Five fully worked designs — feed ranking, fraud, autocomplete, churn, LLM ticket triage — with real numbers and deep-dive follow-ups | Read 2–3 of them end to end so the template stops being abstract. |
| [feature_engineering.md](feature_engineering.md) | The deepest-deep-dive topic: encodings, leakage, train/serve skew, the five classic feature bugs | Read once carefully. Leakage questions are where candidates die. |

Suggested order: template → one scenario closest to your target company → feature engineering → remaining scenarios.

## The 60-second skeleton, expanded

This is question #20 in the [cheatsheet](../docs/interview-cheatsheet.md). Eight steps. Say them in order and you cannot get lost.

**1. Clarify the goal.** Before anything else, turn the product ask into one sentence with a verb and a number: "increase 30-day retention by ranking feed items by predicted engagement." Ask about scale (users, items, QPS), latency tolerance, and what the *business* loses when the model is wrong in each direction. Five minutes here saves you from designing the wrong system for forty.

**2. Metric — online and offline.** Pick the offline metric you'll optimize during training (AUC, NDCG, log loss) *and* the online metric the A/B test will be judged on (retention, revenue, fraud loss in dollars) — then say where they diverge. Offline AUC can go up while users get angrier; naming that gap unprompted is one of the strongest signals you can send.

**3. Data.** Where do labels come from, how delayed are they, and how biased? Clicks arrive in seconds; chargebacks take 60 days; churn takes a billing cycle. Logged data only shows outcomes for what the *previous* system chose to show — that feedback loop is the single most-asked follow-up in these interviews, so raise it before they do.

**4. Features.** Group them: user (history, demographics), item (content, age, popularity), context (time, device), and interaction (user × item history). For every feature, the question is "is this value knowable at prediction time, computed the same way in training and serving?" If not, it's a leakage bug or a skew bug waiting to be paged about. Details in [feature_engineering.md](feature_engineering.md).

**5. Model — baseline first.** Always propose the dumb thing first: popularity ranker, logistic regression, a rules list. It ships in a week, sets the floor, and forces the infrastructure (logging, features, evaluation) to exist before the clever model needs it. Then name v2 (gradient boosting, two-tower, fine-tuned LLM) and *what evidence would justify the upgrade* — "if the GBM beats LR by less than a point of AUC, we keep LR and its 2ms inference."

**6. Training pipeline.** How you split (time-based, never random, for anything temporal), how often you retrain (nightly? hourly? triggered by drift?), and how a new model proves itself before serving traffic — backtest on the most recent week, then shadow deploy. Mention that the splits must respect the label delay from step 3, or your validation set is lying to you.

**7. Serving.** State the latency budget and design backwards from it: a feed ranker gets ~100ms for 1,000 candidates, so you need a cheap retrieval stage before the expensive ranking stage; fraud scoring sits inline at ~50ms p99 because the payment is waiting. Always name the fallback — what happens when the model service is down? (Serve the popularity baseline. Never serve nothing.)

**8. Monitoring.** Three layers: system health (latency, error rate), prediction health (score distribution drift, feature null rates), and business health (the online metric, per-segment). Plus the feedback-loop check: a model trained on its own outputs narrows over time. Saying "I'd alert on the score distribution shifting, because labels are 60 days delayed and that's my earliest warning" is a senior closing line.

---

*Next: open the [template](ml_system_design_template.md) and run it against a prompt out loud, with a timer. The skeleton only becomes yours when you've spoken it.*
