# Optimizers — SGD, Momentum, Adam

## The Intuition

Training a model is hiking down a foggy mountain at night: you can only feel the slope under your feet (the gradient), and you want the valley floor (low loss). Plain SGD takes a step directly downhill from wherever it stands — honest, but jittery, and agonizing in narrow ravines where it ping-pongs between the walls instead of moving along the floor. Momentum fixes that by behaving like a heavy ball instead of a hiker: it accumulates velocity, so the side-to-side jitters cancel out while the persistent downhill direction compounds. Adam adds one more trick — it gives *each parameter its own step size*, scaled down where gradients have been consistently large and up where they've been small, so a parameter that rarely gets a gradient (a rare word's embedding) still learns at a decent clip. That per-parameter adaptivity is why Adam is the default for deep learning, and why "just use Adam" is both a meme and mostly correct advice.

## The Math

**SGD** (stochastic gradient descent):

$$\theta_{t+1} = \theta_t - \alpha \nabla L(\theta_t)$$

- $\theta_t$ — the parameters at step $t$
- $\alpha$ — learning rate
- $\nabla L(\theta_t)$ — gradient of the loss (computed on a mini-batch, hence "stochastic")

Step directly downhill, distance proportional to steepness.

**Momentum:**

$$v_{t+1} = \beta v_t + \nabla L(\theta_t) \qquad \theta_{t+1} = \theta_t - \alpha\, v_{t+1}$$

- $v_t$ — the velocity: an exponentially-decaying running sum of past gradients
- $\beta$ — momentum coefficient, typically 0.9: how much old velocity survives each step

Move with accumulated velocity; oscillating gradient components cancel inside $v$, consistent ones add up.

**Adam** (adaptive moment estimation):

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)\nabla L \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2)(\nabla L)^2$$

$$\hat{m}_t = \frac{m_t}{1-\beta_1^t} \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t} \qquad \theta_{t+1} = \theta_t - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

- $m_t$ — first moment: running mean of gradients (the momentum part)
- $v_t$ — second moment: running mean of *squared* gradients, element-wise (the adaptive part)
- $\beta_1, \beta_2$ — decay rates, typically 0.9 and 0.999
- $\hat{m}_t, \hat{v}_t$ — bias-corrected estimates (both start at zero, so early values are inflated to compensate)
- $\epsilon$ — tiny constant (~$10^{-8}$) preventing division by zero

Each parameter steps in its momentum direction, divided by the typical size of its own gradients — steep, twitchy directions get reined in; flat, quiet ones get amplified.

## Open the visualization

`open visualize.html in any browser` — then click anywhere on the loss surface to restart all three optimizers from that point.

## When to use this

- **SGD (+momentum) for computer vision models you'll deploy** — ResNets trained with tuned SGD still generalize slightly better than Adam-trained ones; worth it when you'll train once and serve millions.
- **Adam/AdamW for transformers, NLP, and any "just make it train" situation** — sparse gradients, wildly different parameter scales, and no time for LR archaeology all favor per-parameter adaptivity. AdamW (decoupled weight decay) is the modern default.
- **Momentum whenever the loss surface is a ravine** — deep nets with correlated parameters produce exactly the narrow curved valleys where plain SGD oscillates and momentum glides.

## What breaks it

- **Too-high learning rate** — every optimizer diverges; loss climbs to NaN within a few steps. Adam fails *more politely* (the $\sqrt{\hat{v}}$ denominator partially self-limits), which can hide the problem until late in training.
- **SGD on badly-scaled features** — one feature in meters, another in millimeters → elongated loss surface → SGD crawls along the shallow axis. Normalize inputs, or pay with 10× more iterations.
- **Adam on small datasets / simple convex problems** — its adaptive steps add variance precisely when the problem doesn't need them; it can converge to measurably worse minima than tuned SGD (the "marginal value of adaptive methods" result). When training is cheap, SGD + LR sweep is the stronger baseline.

## 5 Interview Questions

**1. Conceptual — "Why does momentum speed up training?"**
Direct answer: it dampens oscillation and accelerates persistent directions — gradient components that flip sign step-to-step cancel inside the velocity buffer, while components pointing the same way every step accumulate up to $1/(1-\beta)$ times the bare gradient. Reason: loss surfaces in deep learning are ravines (high curvature across, low curvature along); plain SGD's step is dominated by the across-direction it should ignore, and momentum is a cheap low-pass filter that fixes exactly that. Likely follow-up: *"What's Nesterov momentum?"* — evaluate the gradient at the look-ahead position $\theta + \beta v$ instead of $\theta$; it corrects the velocity *before* committing the step, giving slightly better theoretical and practical convergence.

**2. Mathematical — "Why does Adam need bias correction?"**
Direct answer: $m$ and $v$ are initialized to zero, so after $t$ steps each is a sum of $(1-\beta)$-weighted terms missing a $(1-\beta^t)$ fraction of its mass — early estimates are biased toward zero, and dividing by $(1-\beta^t)$ restores the right scale. Reason: without it, $\hat{v}$ is badly underestimated in early steps, the effective step $\alpha/\sqrt{\hat{v}}$ is huge, and training can blow up in the first hundred iterations — with $\beta_2 = 0.999$ the $v$ estimate stays meaningfully biased for ~1000 steps. Likely follow-up: *"Is that why warmup helps Adam?"* — related but distinct: bias correction fixes the *expected scale*, warmup additionally protects against high *variance* of the early $\hat{v}$ estimate (the motivation behind RAdam).

**3. Practical — "Training loss plateaus after 10 epochs. What do you try, in order?"**
Direct answer: (1) drop the learning rate (schedule or ReduceLROnPlateau) — most plateaus are step-size-too-large hovering around a minimum; (2) check it's a true plateau, not data: shuffle, verify label quality, look at per-class loss; (3) try a warm restart or switch optimizer family; (4) only then add capacity or change architecture. Reason: LR is the cheapest knob with the highest prior probability of being the culprit — a step too large to descend into the minimum looks exactly like a plateau. Likely follow-up: *"Loss plateaus from step 1, never decreases at all?"* — that's not an optimizer problem: suspect a broken pipeline (gradients not flowing, LR=0, frozen weights, shuffled labels).

**4. Gotcha — "Adam has per-parameter adaptive learning rates, so I don't need a learning rate schedule, right?"**
Direct answer: wrong — $\alpha$ in Adam is still a global multiplier on every step, and schedules (warmup + cosine/linear decay) measurably improve transformer training on top of Adam's adaptivity. Reason: Adam adapts the *relative* step per parameter via $\sqrt{\hat{v}}$, but the *absolute* scale still needs to shrink as you approach a minimum, and early training still benefits from warmup while moment estimates stabilize. Likely follow-up: *"What schedule would you actually use for a transformer?"* — linear warmup over the first few hundred to few thousand steps, then cosine decay to ~10% of peak; it's the boring default because it works.

**5. System design — "You're distributed-training a large model across 64 GPUs. How does the optimizer choice interact with the system design?"**
Direct answer: three pressures — (1) memory: Adam keeps two extra fp32 buffers per parameter, tripling optimizer state, which is why ZeRO/FSDP shard optimizer state across workers; (2) batch size: 64 GPUs means huge effective batches, requiring LR scaling rules and warmup, or batch-size-robust optimizers (LAMB was built for this); (3) communication: gradient all-reduce dominates step time, enabling tricks like gradient compression. Reason: at scale the optimizer is a systems object — its state competes with the model for GPU memory and its update rule determines how far you can push batch size before convergence degrades. Likely follow-up: *"Why does naive large-batch training hurt generalization?"* — fewer, less-noisy updates per epoch tend toward sharp minima; mitigations are LR warmup + linear scaling, longer training, or explicit sharpness-aware methods (SAM).
