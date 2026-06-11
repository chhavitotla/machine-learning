# Backpropagation

## The Intuition

A neural network makes a prediction, gets it wrong, and now faces a blame-assignment problem: there are thousands of weights, and each one contributed *something* to the mistake. Backpropagation answers "how much was each weight's fault?" — and it does it cheaply. The trick is the chain rule: the error's sensitivity to any weight is just the product of sensitivities along the path from that weight to the output. So you compute blame once at the output, then pass it backwards layer by layer, each layer multiplying in its own local derivative — like a receipt being itemized backwards through every department that touched the order. One forward pass to predict, one backward pass to assign blame, and every weight knows exactly which direction to move. That's the entire reason deep learning is trainable at all.

## The Math

Forward pass, for each layer $l$:

$$z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)} \qquad a^{(l)} = \sigma(z^{(l)})$$

- $W^{(l)}$ — weight matrix of layer $l$
- $a^{(l-1)}$ — activations coming in from the previous layer ($a^{(0)}$ is the input)
- $b^{(l)}$ — bias vector of layer $l$
- $z^{(l)}$ — pre-activation (the weighted sum, before the nonlinearity)
- $\sigma$ — activation function (sigmoid, ReLU, …)

Each layer does a weighted sum of what came in, then bends it with a nonlinearity.

Output error (for squared loss $L = \frac{1}{2}(a^{(L)} - y)^2$):

$$\delta^{(L)} = (a^{(L)} - y) \odot \sigma'(z^{(L)})$$

- $\delta^{(L)}$ — the "error signal" at the output layer: how much the loss changes per unit of pre-activation
- $y$ — the target
- $\odot$ — element-wise multiplication
- $\sigma'$ — derivative of the activation function

The blame at the output is the raw error, scaled by how responsive the output neuron was.

The backward recurrence (the heart of backprop):

$$\delta^{(l)} = \left( (W^{(l+1)})^T \delta^{(l+1)} \right) \odot \sigma'(z^{(l)})$$

- $(W^{(l+1)})^T$ — the *transpose* of the next layer's weights: blame flows backwards along the same connections activations flowed forward

Each neuron's blame is the blame of the neurons it fed into, weighted by how strongly it fed them, scaled by its own responsiveness.

The gradients you actually use:

$$\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} (a^{(l-1)})^T \qquad \frac{\partial L}{\partial b^{(l)}} = \delta^{(l)}$$

A weight's gradient is (blame of the neuron it feeds) × (activation of the neuron it comes from) — a weight is guilty in proportion to how much signal it carried into a blamed neuron.

## Open the visualization

`open visualize.html in any browser`

## When to use this

- **Training literally any neural network** — every `loss.backward()` in PyTorch and every `tape.gradient()` in TensorFlow is this algorithm; understanding it is how you debug training when it silently fails.
- **Diagnosing vanishing/exploding gradients** — when a deep network won't learn, you inspect per-layer gradient norms; knowing that $\delta$ is a *product* of terms tells you why depth multiplies the problem.
- **Implementing custom layers or losses** — any time you write an operation autograd doesn't cover (a custom CUDA kernel, a non-differentiable approximation), you must hand-derive its backward pass.

## What breaks it

- **Saturated activations** — a sigmoid neuron stuck at 0.999 has $\sigma' \approx 0$, which zeroes the $\delta$ flowing through it; whole layers go silent and the network stops learning with no error message. (This is why ReLU took over.)
- **Bad weight initialization** — initialize all weights to the same value and every neuron in a layer computes identical activations *and identical gradients* forever (the symmetry problem); the layer collapses to one effective neuron.
- **Depth without skip connections** — $\delta^{(1)}$ is a product of ~$L$ Jacobians; if their norms average 0.9 you get $0.9^{50} \approx 0.005$ (vanishing), at 1.1 you get $117\times$ (exploding). Either way, the early layers train at a uselessly different speed than the late ones.

## 5 Interview Questions

**1. Conceptual — "Explain backpropagation to someone who knows calculus but not ML."**
Direct answer: it's an efficient way to compute the derivative of the loss with respect to every parameter, by applying the chain rule from the output backwards and *reusing* shared intermediate terms. Reason: computing each weight's gradient independently would re-walk the same paths exponentially many times; backprop computes each layer's error signal once and shares it across all weights in the layer, making the backward pass cost roughly the same as the forward pass. Likely follow-up: *"So it's just the chain rule?"* — yes, plus dynamic programming: the insight isn't the calculus, it's the caching of $\delta^{(l)}$ so the total cost is $O(\text{edges})$, not $O(\text{paths})$.

**2. Mathematical — "Why does the transpose $W^T$ appear in the backward pass?"**
Direct answer: in the forward pass, neuron $j$ in layer $l$ sends its activation to *every* neuron $k$ in layer $l{+}1$ via weight $W_{kj}$; in the backward pass, neuron $j$'s blame is the sum over all those recipients, $\sum_k W_{kj}\,\delta_k$ — and summing over the first index of $W$ instead of the second is exactly multiplication by $W^T$. Reason: it's the same connections traversed in the opposite direction, so the same matrix appears with its indices swapped. Likely follow-up: *"What's the shape check?"* — $\delta^{(l+1)}$ is $(n_{l+1} \times 1)$, $W^{(l+1)}$ is $(n_{l+1} \times n_l)$, so only $W^T\delta$ yields the required $(n_l \times 1)$.

**3. Practical — "Your loss is stuck at exactly the same value from step one. Walk me through debugging."**
Direct answer: check, in order — (1) the optimizer actually received the model's parameters, (2) gradients are nonzero (`param.grad.norm()` per layer), (3) you didn't forget `optimizer.step()` or call `zero_grad()` in the wrong place, (4) no `detach()`/`no_grad` accidentally cut the graph, (5) the learning rate isn't 0 or the labels aren't constant. Reason: a *perfectly flat* loss almost always means gradients aren't flowing or aren't being applied — a broken pipeline, not a hard optimization problem; a hard problem produces a noisy-but-moving loss. Likely follow-up: *"Gradients are nonzero but tiny in early layers only?"* — that's vanishing gradients: switch to ReLU-family activations, add residual connections, check initialization (He/Xavier), or add normalization layers.

**4. Gotcha — "Can you train a deep network with all weights initialized to zero?"**
Direct answer: no for hidden layers — every neuron in a layer gets the same activations and, by symmetry, the same gradients, so they update identically forever and the layer never differentiates. Reason: backprop's gradient for a weight depends on input activation × downstream blame, and both are identical across neurons that start identical; nothing in the algorithm breaks the tie. Likely follow-up: *"Does the same apply to biases, or to logistic regression?"* — biases can be zero-initialized (the random weights already break symmetry), and logistic regression is fine with all-zeros because it has no hidden layer — there's no symmetry to break.

**5. System design — "You're training a 100-layer network and gradients explode. Design the fixes, in priority order."**
Direct answer: (1) residual/skip connections so the gradient has an identity path around every block, (2) proper init (He for ReLU) so per-layer Jacobian norms start near 1, (3) normalization layers (BatchNorm/LayerNorm) to keep activations in range, (4) gradient clipping as the safety net, (5) lower learning rate with warmup. Reason: skip connections and init fix the *cause* (multiplicative depth), while clipping only suppresses the *symptom* — interviewers want cause-first ordering. Likely follow-up: *"Why does clipping alone work for RNNs in practice?"* — RNNs reuse the same $W$ at every timestep so you can't re-architect the product away as easily; clipping plus gating (LSTM/GRU) is the standard answer there.
