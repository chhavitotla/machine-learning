# RNNs & LSTMs

## The Intuition

A feedforward network sees each input with total amnesia — fine for a photo, useless for a sentence, where the meaning of "bank" depends on words from ten steps ago. A recurrent network fixes this with one move: it keeps a hidden state, a running summary vector that it updates at every timestep by blending the new input with the previous summary — *using the same weights every step*. Reading a sequence is folding it, item by item, into that one vector. The catch appears when you train it: to learn that step 50's output depended on step 1's input, blame must travel backwards through 50 copies of the same transformation, getting multiplied by roughly the same factor each step. Multiply anything slightly less than one by itself 50 times and it's gone; slightly more than one, and it explodes. So vanilla RNNs are structurally forgetful — not for lack of capacity, but because the *gradient* can't survive the trip. The LSTM's answer is to give memory a protected lane: a cell state that flows along the chain nearly untouched, with three learned gates deciding what to erase, what to write, and what to reveal. Along that lane the dangerous repeated multiplication becomes a chain of gentle gatekeeping near 1, so blame — and therefore memory — can finally travel long distances.

## The Math

The recurrent update (vanilla RNN):

$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b)$$

- $x_t$ — the input at timestep $t$ (a character, word embedding, sensor reading)
- $h_t$ — the hidden state: the network's running summary after seeing inputs $1 \ldots t$
- $W_{hh}, W_{xh}$ — recurrent and input weight matrices, *shared across all timesteps*
- $b$ — bias; $\tanh$ squashes the state into $(-1, 1)$

Each step blends the old summary with the new input through the same weights — one layer, applied over and over, is the entire network.

Vanishing gradients through repeated multiplication:

$$\frac{\partial h_T}{\partial h_t} = \prod_{k=t+1}^{T} \frac{\partial h_k}{\partial h_{k-1}} = \prod_{k=t+1}^{T} \text{diag}\!\left(\tanh'(z_k)\right) W_{hh}^T$$

- $\frac{\partial h_T}{\partial h_t}$ — how much the state at the end still feels a change at step $t$; the gradient to early steps is proportional to it
- $\tanh'(z_k) \le 1$ — the activation's local slope, always shrinking the product
- $T - t$ — the gap being bridged: the number of factors in the product

Blame reaching $T - t$ steps back is a product of $T - t$ Jacobians of the *same* matrix; if their typical scale is $0.9$, then $0.9^{50} \approx 0.005$ — the signal vanishes (and at $1.1$ it explodes $117\times$), so a vanilla RNN literally cannot learn dependencies much longer than ~10–20 steps.

The LSTM gates:

$$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f) \qquad i_t = \sigma(W_i [h_{t-1}, x_t] + b_i) \qquad o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$$

- $f_t$ — the forget gate: per-coordinate, how much of the old memory to keep (0 = erase, 1 = preserve)
- $i_t$ — the input gate: how much new information to write
- $o_t$ — the output gate: how much of the memory to reveal as $h_t$
- $\sigma$ — the sigmoid, making every gate a soft dial in $(0, 1)$; $[h_{t-1}, x_t]$ — concatenation

Three small networks look at the same context and set dials: keep, write, reveal — all learned, all differentiable.

The cell state — the protected lane:

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t, \qquad \tilde{c}_t = \tanh(W_c [h_{t-1}, x_t] + b_c), \qquad h_t = o_t \odot \tanh(c_t)$$

- $c_t$ — the cell state: long-term memory, updated *additively*, never squashed through repeated matrix multiplications
- $\tilde{c}_t$ — the candidate content to write; $\odot$ — element-wise multiplication

The crucial difference is in the backward pass: $\frac{\partial c_t}{\partial c_{t-1}} = \text{diag}(f_t)$ — no weight matrix, no activation derivative — so gradient flowing along the cell is multiplied only by forget-gate values; wherever the network learns to hold $f_t \approx 1$, blame travels back across hundreds of steps essentially undamped (the same identity-path idea as a residual connection, applied through time).

## Open the visualization

`open visualize.html in any browser` — it animates a character sequence flowing through an unrolled chain (one column per timestep, cyan/red bars showing each hidden state), lets you toggle between Vanilla RNN and LSTM, and a "Gradient flow" mode runs the backward pass step by step, displaying each step's local gradient factor and how much gradient actually survives back to step 0 in each architecture.

## When to use this

- **Streaming and low-latency sequence inference** — keyword spotting on earbuds, real-time anomaly detection on sensor feeds: an RNN/LSTM carries a fixed-size state and does $O(1)$ work per new element, with no need to re-attend over a growing window like a transformer.
- **Modest-data time series and sequence labeling** — equipment-failure prediction from telemetry, ECG/EEG classification, handwriting recognition: with thousands (not billions) of training sequences, an LSTM's built-in sequential inductive bias often beats a transformer that hasn't enough data to learn order from scratch.
- **Understanding the modern toolbox** — gating, saturating activations, and the additive-path trick are the conceptual ancestors of residual connections, GRUs, and today's state-space models (Mamba); the vanishing-gradient analysis here is the same one you'll reuse on any deep or recurrent computation graph.

## What breaks it

- **Very long-range dependencies, even with gates** — an LSTM stretches usable memory from ~20 steps to hundreds, but everything must still squeeze through one fixed-size vector; recalling an exact token from 5,000 steps back is a lossy-compression problem that attention solves by just looking back directly, which is a core reason transformers displaced LSTMs in NLP.
- **No parallelism across time** — $h_t$ needs $h_{t-1}$, so training is inherently sequential along the sequence; on long sequences GPUs sit half-idle while a transformer processes every position at once, making LSTM training many times slower at equal scale.
- **Exploding gradients and truncation artifacts** — gating fixes vanishing along the cell, not exploding elsewhere: without gradient clipping a single steep region can produce a giant update that destroys the weights (loss → NaN); and truncated BPTT, the standard trick for long sequences, silently caps the horizon — the model *cannot* learn any dependency longer than the truncation window, no matter the architecture.

## 5 Interview Questions

**1. Conceptual — "Why do vanilla RNNs forget, and how does the LSTM fix it — in one architecture change?"**
Direct answer: they don't forget at inference so much as they *can't learn* to remember: the gradient from step $T$ back to step $t$ is a product of $T-t$ near-identical Jacobians, which vanishes (or explodes) exponentially in the gap, so long-range credit assignment never happens. The LSTM's change is an additive memory path: the cell state updates as $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$, so the backward factor along it is just $f_t$ — a learned gate that can sit near 1 — instead of $W_{hh}^T$ times an activation derivative. Reason: turning repeated *multiplication by a matrix* into repeated *gating near identity* is exactly what keeps the product from collapsing. Likely follow-up: *"How is that related to residual connections?"* — same trick, different axis: ResNets give gradients an identity path through depth, LSTMs give one through time; both make deep products of Jacobians behave by anchoring them near the identity.

**2. Mathematical — "Derive the condition under which the vanilla RNN gradient vanishes."**
Direct answer: $\frac{\partial h_T}{\partial h_t} = \prod_{k=t+1}^{T} D_k W_{hh}^T$ where $D_k = \text{diag}(\tanh'(z_k))$; taking norms, $\lVert \frac{\partial h_T}{\partial h_t} \rVert \le (\gamma \cdot \sigma_{\max}(W_{hh}))^{T-t}$ with $\gamma = \max \tanh' \le 1$ — so if the largest singular value $\sigma_{\max}(W_{hh}) < 1/\gamma$, the bound decays exponentially in $T - t$ and the gradient provably vanishes; if the smallest relevant singular values exceed 1, it can explode. Reason: weight sharing means the *same* $W_{hh}$ appears in every factor, so its spectrum compounds geometrically — depth-in-time with no fresh matrices to rescue the product. Likely follow-up: *"Why not just initialize $W_{hh}$ orthogonal so all singular values are 1?"* — it helps at step 0 (and IRNN/orthogonal-init papers do it), but training drifts the spectrum, and the $\tanh'$ factors still drag the product below 1 whenever units saturate; you delay the disease rather than cure it — gating cures it.

**3. Practical — "Your LSTM's training loss goes NaN partway through the first epoch. Walk me through the fix."**
Direct answer: that's exploding gradients (or a numerical cousin): (1) add gradient-norm clipping (clip total norm to ~1–5) — this alone usually fixes it, (2) lower the learning rate, (3) check for unscaled inputs or a missing normalization, (4) shorten the BPTT truncation window, (5) inspect for division/log of zero in the loss. Reason: recurrence reuses the same weights hundreds of times, so a mildly large spectral norm compounds into astronomically large gradients on rare long sequences — clipping caps the *step size* without biasing its direction, which is why it's the standard, almost mandatory, RNN training ingredient. Likely follow-up: *"Clipping saved it, but now the model still can't learn long dependencies — why?"* — clipping fixes exploding, not vanishing; check the truncation length (it hard-caps the learnable horizon), consider initializing the forget-gate bias to 1–2 so $f_t$ starts near "remember everything," and verify the dependency you want actually fits in the cell's capacity.

**4. Gotcha — "The forget gate multiplies the cell state at every step — isn't that the same repeated multiplication that kills vanilla RNNs?"**
Direct answer: it's repeated multiplication, but by a fundamentally different object: $\frac{\partial c_t}{\partial c_{t-1}} = \text{diag}(f_t)$ — a *learned, input-dependent* gate with no weight matrix and no activation derivative in the path — whereas the RNN multiplies by $D_k W_{hh}^T$, a fixed-spectrum matrix the gradient cannot route around. Reason: the network can set $f_t \approx 1$ on exactly the coordinates and timesteps where memory matters, making the product approach $1^{T-t} = 1$ along those coordinates — vanishing becomes a *choice* the model controls per-dimension, not a structural inevitability; with $b_f$ initialized positive it starts in the remember regime. Likely follow-up: *"So can an LSTM still vanish?"* — yes: if the task pushes forget gates low, or blame must repeatedly route through the $h_t$ pathway (which still has matrices and tanh), decay returns — which is why even LSTMs top out at hundreds, not tens of thousands, of steps.

**5. System design — "Design the model for live anomaly detection on 10,000 industrial sensors streaming at 10 Hz on an edge box — and justify LSTM over transformer."**
Direct answer: one shared stacked-LSTM (2 layers, a few hundred units) trained for next-window prediction, with per-sensor states: at inference each new reading is folded into that sensor's persistent $(h, c)$ in $O(1)$, the anomaly score is the prediction error against the actual next reading, and training uses truncated BPTT (window ≈ a few minutes of data) with gradient clipping and forget-bias init of 1. Reason: the workload is *streaming* — a transformer must keep and re-attend over a sliding window per sensor (memory and compute grow with context × 10,000 sensors), while the LSTM's entire memory is one fixed vector pair per sensor, easily resident on an edge device, and at this data scale its sequential inductive bias trains well; latency per tick is a single matrix-vector pass. Likely follow-up: *"Anomalies relative to a 24-hour cycle — your truncation window is minutes, so how can it learn that?"* — it can't through recurrence alone, and pretending otherwise is the trap: feed periodicity in as features (time-of-day/day-of-week encodings, or deseasonalize the signal upstream) so the LSTM only has to carry short-horizon dynamics — architecture for the short range, feature engineering for the provably-out-of-reach long range.
