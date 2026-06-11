# Batch Normalization

## The Intuition

A deep network is an assembly line where every layer learns to process whatever the layer before it hands over. The problem: while layer 5 is busy learning, layers 1–4 are *also* updating, so the distribution of inputs layer 5 sees keeps shifting under its feet — it's practicing free throws while someone moves the hoop. Worse, with squashing activations like sigmoid, the signal decays: each layer shrinks the spread of its outputs a little, so by layer 6 every neuron sees nearly identical inputs and the gradients are dust. Batch norm's fix is almost rude in its simplicity: between the matrix multiply and the activation, *re-standardize*. Take the current mini-batch, subtract the mean, divide by the standard deviation — now every layer, at every training step, sees inputs on the same calm scale: centered at zero, spread of one. Then, so no expressive power is lost, each feature gets two learnable knobs (a scale γ and a shift β) to dial back in whatever distribution it actually wants. The hoop stops moving, you can crank the learning rate way up, and training that took weeks of initialization tuning just... works.

## The Math

**Batch statistics** (per feature, across the mini-batch):

$$\mu_B = \frac{1}{n}\sum_{i=1}^{n} x_i \qquad \sigma_B^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i - \mu_B)^2$$

- $x_i$ — the value of one feature for example $i$ in the current mini-batch
- $n$ — the batch size
- $\mu_B$ — the mean of that feature over this batch
- $\sigma_B^2$ — the variance of that feature over this batch

Measure where this feature currently lives — using only the examples that happen to share the batch, which is both BN's power and its classic bug.

**Normalize:**

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

- $\hat{x}_i$ — the standardized value: mean 0, variance 1 across the batch
- $\epsilon$ — a tiny constant (~$10^{-5}$) so a near-zero variance can't cause division by zero

Whatever scale the upstream weights produced, the feature now arrives centered at zero with unit spread.

**Scale and shift** (the learnable part):

$$y_i = \gamma\, \hat{x}_i + \beta$$

- $\gamma$ — learnable per-feature scale, initialized to 1
- $\beta$ — learnable per-feature shift, initialized to 0

The network can recover *any* mean and spread it wants — including undoing the normalization entirely — but now the distribution is set by two clean, directly-trainable parameters instead of being an accident of every upstream weight.

**Train vs. inference** (running statistics):

$$\mu_{\text{run}} \leftarrow (1-m)\,\mu_{\text{run}} + m\,\mu_B \qquad \sigma^2_{\text{run}} \leftarrow (1-m)\,\sigma^2_{\text{run}} + m\,\sigma_B^2$$

$$\hat{x} = \frac{x - \mu_{\text{run}}}{\sqrt{\sigma^2_{\text{run}} + \epsilon}} \quad \text{(inference)}$$

- $m$ — momentum of the exponential moving average (0.1 in `implementation.py`)
- $\mu_{\text{run}}, \sigma^2_{\text{run}}$ — running averages of the batch statistics, accumulated during training

During training, every batch updates the running averages; at inference those *frozen* averages replace the batch statistics, so the model's output depends only on its one input — not on whoever else is in the batch. Demo 2 in `implementation.py` shows exactly what goes wrong when you forget this.

## Open the visualization

`open visualize.html in any browser` — a small MLP trains live while you toggle batch norm on and off, watching each hidden layer's pre-activation histogram either drift into sigmoid saturation or snap to mean ≈ 0, std ≈ 1.

## When to use this

- **CNNs for vision, the canonical home** — ResNets, EfficientNets, and essentially every modern conv architecture put BN after each conv; with typical batch sizes (≥32) it's the difference between training a 100-layer network and not being able to.
- **Any deep net with saturating activations or fragile initialization** — if your sigmoid/tanh network's deeper layers go silent (the geometric std collapse in Demo 1), BN re-standardizes at every layer and lets the signal survive arbitrary depth.
- **When you need to crank the learning rate** — BN decouples each layer's input scale from all upstream weights, so a 5–10× larger learning rate that would diverge without BN trains stably; in compute-constrained settings that speedup is the whole game.

## What breaks it

- **Tiny batch sizes** — statistics estimated from 2–4 examples are mostly noise, injected into every forward pass; accuracy degrades sharply below batch ~8, and at batch 1 in training mode the math degenerates completely (mean = the input, variance = 0, output = β for *every* input). Detection/segmentation models with per-GPU batches of 1–2 use GroupNorm for exactly this reason.
- **Train/inference mismatch** — the most common BN bug in production: serving with `training=True` (or forgetting `model.eval()`), so outputs depend on the serving batch's composition — the same user gets different predictions depending on who else is in the request batch. The flip side: running statistics that never converged (short training, heavy augmentation) make eval mode itself inaccurate.
- **RNNs and sequence models** — statistics would have to be computed per time step, sequences have different lengths, and at generation time there is no batch at all; BN is essentially unusable here, which is why transformers and RNNs standardize across *features* instead (LayerNorm) — same normalize-then-γβ recipe, no batch dependence.

## 5 Interview Questions

**1. Conceptual — "Why does batch norm let you train with much higher learning rates?"**
Direct answer: because it makes each layer's input scale independent of all upstream weights — a big update in layer 1 can no longer explode or collapse the activations arriving at layer 5, since layer 5's inputs are re-standardized to mean 0, std 1 regardless. Reason: without BN, the safe learning rate is set by the worst-case compounding of scale changes through depth; BN cuts that chain, and empirically also smooths the loss landscape (more predictable gradients → bigger safe steps). Likely follow-up: *"Is the original 'internal covariate shift' explanation correct?"* — be honest: it was the paper's motivation, but later work (Santurkar et al., 2018) showed BN helps even with covariate shift artificially re-injected; the smoother-landscape account now carries more weight, and knowing the debate exists is the senior-level answer.

**2. Mathematical — "If γ can rescale and β can re-shift, the network can just undo the normalization. So what did BN accomplish?"**
Direct answer: BN changes the *parametrization*, not the set of representable functions — each feature's mean and variance become two explicit, directly-trainable parameters instead of an emergent property of thousands of upstream weights. Reason: optimization in the (γ, β) parametrization is far better conditioned — shifting a feature's mean means nudging one β, not coordinating a whole weight matrix — and the normalization holds *during* training even while γ and β drift. Likely follow-up: *"What happens to a bias placed before BN?"* — cancelled exactly (BN subtracts the mean; a bias only moves the mean), so it's redundant; β replaces it, which is why `Conv2d(..., bias=False)` precedes BN in real codebases.

**3. Practical — "Your model gets 95% accuracy in training, near-random in production. What do you check first?"**
Direct answer: BN mode — confirm the deployed model runs with frozen running statistics (`model.eval()` / `training=False`), because serving with batch statistics makes each prediction depend on the other requests in the batch, and a serving batch of one degenerates to outputting β. Reason: it's the most common train/serve divergence involving BN, it's silent (no error, just wrong numbers), and it's a one-line check — Demo 2 in `implementation.py` reproduces it: two very different inputs map to *identical* outputs. Likely follow-up: *"Eval mode is on and it's still bad?"* — then suspect the running statistics themselves: trained too briefly to converge, computed under augmentation that doesn't match serving data, or genuine distribution shift; the cheap fix is re-estimating BN statistics with a few hundred forward passes over production-like data.

**4. Gotcha — "What does batch norm output in training mode with a batch size of 1?"**
Direct answer: β, for every possible input — the batch mean equals the input itself, the variance is exactly 0, so $\hat{x} = 0$ and $y = \gamma \cdot 0 + \beta$. Reason: the normalization is computed *across the batch*, and a batch of one has no spread to measure; this isn't a numerical edge case softened by ε, it's the math working as specified on a degenerate input. Likely follow-up: *"How do GroupNorm and LayerNorm avoid this?"* — they normalize across features within a *single* example, so their statistics are well-defined at batch size 1 and identical between training and inference — no running statistics at all.

**5. System design — "You're training a detection model across 32 GPUs with a per-GPU batch of 4. How does BN interact with that, and what do you do?"**
Direct answer: vanilla BN computes statistics per device, so each GPU normalizes with a noisy 4-sample estimate — quality drops, and the effective model differs across replicas; the fixes are SyncBN (all-reduce the mean and variance across GPUs each forward pass, paying communication per BN layer) or switching to GroupNorm (batch-independent, no sync). Reason: BN is the rare layer whose *forward pass* couples examples, so it's the rare layer that couples *devices* — its statistics become a distributed-systems object, not just math. Likely follow-up: *"And at serving time?"* — inference BN is an affine transform with frozen constants, so fold it into the preceding conv/linear weights ($W' = \gamma W / \sigma_{\text{run}}$, plus the matching bias adjustment): one fused layer, zero runtime cost, identical outputs.
