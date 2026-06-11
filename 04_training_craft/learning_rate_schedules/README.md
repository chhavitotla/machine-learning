# Learning Rate Schedules — Warmup, Decay, Cosine

## The Intuition

Imagine parallel parking. A block away you drive at full speed — big movements, no precision needed, you just want to close the distance. The final few inches happen at a crawl, because at full speed you'd overshoot the spot every time, lurching back and forth forever. A learning rate schedule is exactly this: big steps early when you're far from the minimum, small careful steps when you're close — because the step size that covers ground fast is the same step size that bounces around the minimum and never settles into it. Warmup is the counterintuitive part: for the first few hundred steps you *start slow and speed up*. Early gradients are huge and erratic (random weights, Adam's statistics not yet stabilized), so a full-size first step can fling the weights somewhere the rest of training never recovers from. Ramp up gently, cruise at full speed, brake smoothly into the minimum — that shape is behind nearly every modern training run.

## The Math

**Step decay:**

$$\alpha_t = \alpha_0 \cdot \gamma^{\lfloor t / s \rfloor}$$

- $\alpha_t$ — learning rate at step $t$
- $\alpha_0$ — initial (base) learning rate
- $\gamma$ — decay factor, typically 0.1–0.5
- $s$ — how many steps between drops
- $\lfloor \cdot \rfloor$ — floor (integer division)

Hold the learning rate constant, then slash it by $\gamma$ every $s$ steps — a staircase heading down.

**Exponential decay:**

$$\alpha_t = \alpha_0 \cdot e^{-kt}$$

- $k$ — decay rate: larger $k$ means faster shrinking
- $t$ — current step

Shrink by the same *fraction* every step, so the curve is a smooth slide instead of a staircase.

**Cosine annealing:**

$$\alpha_t = \alpha_{min} + \frac{1}{2}(\alpha_0 - \alpha_{min})\left(1 + \cos\frac{\pi t}{T}\right)$$

- $\alpha_{min}$ — floor learning rate, often 0 or $\alpha_0 / 100$
- $T$ — total number of training steps

Follow half a cosine wave from $\alpha_0$ to $\alpha_{min}$: gentle at the start, fastest in the middle, gentle at the end — the slow final glide is why models "settle in" so cleanly. *Warm restarts* (SGDR) replay the curve — snap back to $\alpha_0$ at the bottom — kicking the model out of its basin to hunt for a better one.

**Linear warmup** (composed with any decay above):

$$\alpha_t = \alpha_0 \cdot \frac{t}{T_w} \quad \text{for } t < T_w$$

- $T_w$ — number of warmup steps (commonly a few hundred to a few thousand)

Ramp linearly from 0 to $\alpha_0$ over the first $T_w$ steps, then hand off to the decay schedule; warmup + cosine is the default recipe for transformers.

## Open the visualization

`open visualize.html in any browser` — watch the LR curve trace out live while a ball descends a bumpy 1D loss landscape under that schedule, racing a constant-LR ghost that bounces around the minimum forever while cosine decay settles in.

## When to use this

- **Warmup + cosine decay for any transformer or Adam-trained model** — Adam's second-moment estimate is garbage for the first ~1000 steps, and a full-size step on garbage statistics can permanently damage training; ramp up over 500–2000 steps, cosine to ~10% of peak — the recipe in GPT, LLaMA, BERT, and essentially every LLM paper.
- **Step decay for classic vision training with SGD** — drop LR 10× at fixed milestones (epochs 30/60/90 on ImageNet); the loss visibly drops at each cliff, decades of baselines are built on it, and the LR at any epoch is obvious, making debugging trivial.
- **Warm restarts when training is cheap and you want an ensemble for free** — each cosine cycle ends in a different good minimum; snapshot the weights at every trough and average their predictions (snapshot ensembles) for ensemble accuracy on one budget.

## What breaks it

- **Decaying too early or too fast** — the model lands in the first mediocre basin it finds and the tiny LR locks it there; training loss looks smooth and converged while validation accuracy sits points below what a longer high-LR phase would reach — the smoothness is the trap.
- **A schedule pinned to the wrong horizon** — cosine needs $T$ fixed in advance. Train 2× longer after the fact and the LR is near zero for the whole second half (wasted compute); stop at half the budget and you've stopped mid-decay with the LR still too hot and the model unsettled.
- **Skipping warmup at large batch size or high peak LR** — the first steps on random weights produce enormous gradients; loss spikes to NaN in the first 100 steps, or survives but plateaus permanently because the early violent updates wrecked the initialization. Cruelly, the same config works at small scale, so it surfaces exactly when the run is most expensive.

## 5 Interview Questions

**1. Conceptual — "Why decay the learning rate at all? Why not just pick one good value?"**
Direct answer: because no single value suits both phases — early on you want large steps to cover distance; late, a large step overshoots, so SGD with constant LR converges not to the minimum but to a noise ball around it whose radius scales with $\alpha$. Reason: with mini-batch noise, SGD's stationary behavior is a random walk in a bowl — the LR sets its temperature, and shrinking it shrinks the noise ball until you're effectively *at* the minimum. Likely follow-up: *"So why not start tiny and be safe?"* — you'd take forever to cover the distance, and small-step trajectories funnel into the nearest sharp minimum instead of the wide flat ones that generalize better.

**2. Mathematical — "Halfway through training, which has decayed further: exponential with $k = 2/T$, or cosine?"**
Direct answer: at $t = T/2$, exponential sits at $e^{-1} \approx 0.37\alpha_0$, cosine at exactly $0.5\alpha_0$ — exponential is lower — but cosine's *rate* of decay peaks right there ($d\alpha/dt \propto \sin(\pi t/T)$) while exponential's rate only shrinks. Reason: cosine concentrates its decay in the middle and is deliberately flat at both ends — full-speed exploration, then a long slow glide — whereas exponential spends most of training at a small LR. Likely follow-up: *"Why does the flat start matter?"* — the high-LR phase does the basin *selection*; cut it short and you lock in the first basin you stumble into.

**3. Practical — "Your transformer's loss explodes to NaN within the first 200 steps. Walk me through your fix."**
Direct answer: (1) add or lengthen linear warmup — this is the classic no-warmup signature; (2) lower the peak LR, since warmup and peak trade off; (3) add gradient clipping (global norm ~1.0) as a backstop; (4) if NaNs persist, look past the schedule at fp16 overflow, bad data, or initialization. Reason: at step 1 the weights are random and Adam's $\hat{v}$ is estimated from a handful of samples, so the effective step $\alpha/\sqrt{\hat{v}}$ is enormous in exactly the directions with the least reliable statistics — warmup keeps steps small until the estimates mean something. Likely follow-up: *"How many warmup steps?"* — a few hundred to a few thousand, scaling with batch size and peak LR; a quick small-scale sweep settles it.

**4. Gotcha — "Adam already adapts the learning rate per parameter — isn't a schedule on top of it redundant?"**
Direct answer: no — Adam's $1/\sqrt{\hat{v}}$ sets the *relative* step size across parameters, but $\alpha$ remains a global multiplier on every update, and the right global scale near convergence is genuinely smaller than at the start. Reason: Adam answers "which parameters should move more than others?", the schedule answers "how big should steps be at this stage of training?" — and empirically warmup + decay on top of Adam is worth real loss on transformers, which is why every LLM recipe uses both. Likely follow-up: *"Then what does Adam's adaptivity buy you?"* — robustness across parameters with wildly different gradient scales (embeddings vs. layernorm gains), not freedom from tuning the global LR trajectory.

**5. System design — "You're launching a 3-week pretraining run on 256 GPUs. How do you design the LR schedule, and what operational risks does it create?"**
Direct answer: linear warmup over the first ~0.1–1% of steps, cosine decay to ~10% of peak pinned to the token budget, peak LR scaled to the large batch and validated at small scale — and the risks: (1) the schedule is a function of total steps, so a mid-run budget change breaks it; (2) every checkpoint must serialize the step counter, or a resume after a node failure silently replays warmup at step 200,000 and craters the loss; (3) loss-spike recovery means rewinding and skipping bad data, which only works if the schedule is reproducible from the step count alone. Reason: at this scale the schedule is part of the system contract — a pure function of global step, so restarts, rewinds, and elastic resizing all land on the identical LR. Likely follow-up: *"What if you might extend the run?"* — pick a schedule that doesn't need the horizon up front: constant LR with a cooldown appended (warmup-stable-decay) or inverse-square-root decay.
