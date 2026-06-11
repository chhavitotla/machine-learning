# Calculus & Gradients

## The Intuition

You're standing on a hillside in thick fog. You can't see the valley — you can only feel the slope under your feet. The derivative is that feeling: at any point on a curve, it tells you which way is uphill and how steep. And here's the move that powers every model you've ever heard of: *feel the slope, step the other way, repeat.* That's gradient descent. Nothing more. The derivative itself is just a sharpened version of "rise over run" — take two points on the curve, draw the line through them (a secant), then slide the second point toward the first. As the gap $h$ shrinks, the secant settles onto the tangent, and its slope settles onto a single number: the derivative. Training a neural network is this exact fog-walk in a million dimensions, where the "hill" is the loss and your position is the weights. The fog matters too: the walker only ever feels *local* slope, which is why it can settle into a shallow valley while a deeper one sits unseen next door.

## The Math

The derivative as a limit:

$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

- $f$ — the function (think: the loss)
- $x$ — where you're standing
- $h$ — the gap between the two points of the secant line
- $f'(x)$ — the slope of the tangent: instantaneous rise over run

This says: the slope between two nearby points becomes *the* slope as the points merge.

The central difference (how computers estimate it):

$$f'(x) \approx \frac{f(x+h) - f(x-h)}{2h}$$

- $h$ — a small but finite step, e.g. $10^{-5}$

Straddling $x$ symmetrically makes the error shrink like $h^2$ instead of $h$ — the lopsided errors of each side cancel; this is what `gradcheck` uses to catch buggy backprop code.

The gradient descent update:

$$x \leftarrow x - \alpha f'(x)$$

- $\alpha$ — the learning rate: how far you step per unit of slope
- $-f'(x)$ — the downhill direction (the derivative points uphill, so negate it)

Steep slope ⇒ big step; flat slope ⇒ tiny step; slope of zero ⇒ you stop — which is exactly what "minimum" means.

The stopping condition has a name worth knowing:

$$f'(x^*) = 0$$

- $x^*$ — a critical point: a minimum, a maximum, or a saddle

Gradient descent can only ever halt where the slope vanishes — but the slope also vanishes at *local* minima and saddles, which is why where you start and how big you step decide where you end up.

## Open the visualization

`open visualize.html in any browser`

## When to use this

- **Training literally anything** — every weight update in every neural network is $w \leftarrow w - \alpha \,\partial L/\partial w$; Adam, momentum, RMSProp are all garnishes on this one line.
- **Gradient checking** — when you implement backprop by hand, compare your analytic gradients against central differences; a mismatch beyond ~$10^{-7}$ relative error means your math has a bug.
- **Hyperparameter intuition** — the learning-rate trade-off (too small: glacial; too large: oscillate or diverge) is the single most common training pathology you'll diagnose in practice, and you can see all of it on a 1D parabola.

## What breaks it

- **Learning rate too large** — on a steep loss, $\alpha f'(x)$ overshoots the minimum and lands somewhere *steeper*; the next step is bigger, and the loss rockets to infinity within a handful of iterations.
- **Local minima and saddle points** — descent stops wherever $f'(x)=0$, not where $f$ is smallest; a ball dropped in the wrong basin will sit there contentedly forever while a deeper valley waits next door.
- **Non-differentiable or flat regions** — kinks (ReLU at 0, absolute value) leave the derivative undefined, and plateaus give a gradient of ~0, so the walker stalls with no slope signal; this is why saturated sigmoids killed deep nets before ReLU.

## 5 Interview Questions

**1. Conceptual — "What does the derivative actually tell you, and why does gradient descent subtract it?"**
Direct answer: $f'(x)$ is the instantaneous slope — the direction and rate of steepest *increase* in 1D — so stepping along $-f'(x)$ is the locally fastest way to decrease $f$. Reason: near $x$, the function is approximately the tangent line $f(x) + f'(x)\,\Delta$, so making $\Delta$ oppose $f'(x)$ guarantees a first-order decrease. Likely follow-up: *"What's the multivariable version?"* — the gradient $\nabla f$, the vector of partial derivatives, which points in the direction of steepest ascent; descent steps along $-\nabla f$.

**2. Mathematical — "Why is the central difference $\frac{f(x+h)-f(x-h)}{2h}$ better than the forward difference?"**
Direct answer: Taylor-expand both sides — $f(x \pm h) = f(x) \pm h f'(x) + \frac{h^2}{2}f''(x) \pm \frac{h^3}{6}f'''(x) + \dots$ — and subtract: the even-order terms cancel, leaving an error of $O(h^2)$ versus the forward difference's $O(h)$. Reason: symmetry kills the $f''$ term that dominates the forward difference's error. Likely follow-up: *"Why not just make $h$ tiny, like $10^{-15}$?"* — floating-point cancellation: $f(x+h) - f(x-h)$ subtracts two nearly equal numbers, so below $h \approx 10^{-5}$ round-off error grows as truncation error shrinks; there's an optimal middle.

**3. Practical — "Your training loss oscillates wildly and sometimes spikes to NaN. Diagnose."**
Direct answer: the learning rate is too large — each step overshoots the minimum into steeper territory, so steps grow until the numbers overflow; drop $\alpha$ by 10× (or add gradient clipping / warmup) and confirm the loss decays smoothly. Reason: for a quadratic bowl with curvature $L$, descent converges only if $\alpha < 2/L$; above that threshold each step multiplies the error by something bigger than 1. Likely follow-up: *"Loss decreases but absurdly slowly — same knob?"* — yes, the other side: $\alpha$ too small, or the loss surface is ill-conditioned (steep in some directions, flat in others), which is what momentum and Adam are built to fix.

**4. Gotcha — "Gradient descent stopped: the gradient is zero. Have you found the minimum?"**
Direct answer: not necessarily — $f'(x)=0$ also holds at local minima, local maxima, and saddle points; in 1D you'd check $f''(x) > 0$, and even then it's only a *local* minimum. Reason: zero slope is a necessary condition for a minimum, not a sufficient one; descent has no view beyond local slope. Likely follow-up: *"Is this a disaster for deep learning?"* — less than you'd think: in very high dimensions most critical points are saddles rather than bad minima, and stochastic gradient noise tends to kick the optimizer off saddles; in practice local minima of big nets are mostly of similar quality.

**5. System design — "You must tune the learning rate for a model that takes 8 hours per run. Design the process."**
Direct answer: run a learning-rate range test first — a single short run where $\alpha$ ramps exponentially while you record loss; pick just below the value where loss stops improving or explodes, then use warmup plus a decay schedule (cosine or step) for the full run, monitoring gradient norms to catch divergence within minutes instead of hours. Reason: the usable $\alpha$ range spans orders of magnitude and the failure modes (divergence, oscillation, stall) are all visible early, so cheap short probes buy you the expensive long run. Likely follow-up: *"Why warmup?"* — early in training the weights are random and the local curvature estimates (especially in Adam's moment buffers) are unreliable, so big early steps can launch the model into a bad region it never recovers from; warmup keeps steps small until the geometry settles.
