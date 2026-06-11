# Transformers & Attention

## The Intuition

Reading the sentence "The animal didn't cross the street because **it** was too tired," you instantly know "it" means the animal, not the street. You did that by letting "it" *look at* every other word and weigh how relevant each one is. That's attention. Each token builds three things from its embedding: a **query** ("here's what I'm looking for"), a **key** ("here's what I contain"), and a **value** ("here's what I'll contribute if you pick me"). Every token's query is compared against every token's key; good matches get high weights; each token's new representation is the weighted average of everyone's values. Do this with several independent sets of Q/K/V (heads) — one head might track grammar, another coreference — stack the whole thing in layers, and you have a transformer. The killer feature over RNNs: every token reaches every other token in *one step*, no matter how far apart, and all of it computes in parallel.

## The Math

Scaled dot-product attention:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- $Q$ — query matrix ($n \times d_k$): one query vector per token, "what am I looking for?"
- $K$ — key matrix ($n \times d_k$): one key vector per token, "what do I advertise?"
- $V$ — value matrix ($n \times d_v$): one value vector per token, "what do I contribute?"
- $d_k$ — dimension of the query/key vectors
- $n$ — number of tokens

Each token scores every token, turns the scores into a probability distribution, and takes that weighted blend of values.

Where Q, K, V come from:

$$Q = XW^Q, \quad K = XW^K, \quad V = XW^V$$

- $X$ — the input embeddings ($n \times d_{model}$)
- $W^Q, W^K, W^V$ — learned projection matrices

The same input is projected three different ways; the projections are what training learns.

Why divide by $\sqrt{d_k}$: for random vectors, dot products grow with dimension — their variance is $d_k$. Unscaled, softmax inputs get huge, softmax saturates to a one-hot, and its gradient vanishes. Dividing by $\sqrt{d_k}$ keeps the variance at 1.

$$\text{softmax}(s_i) = \frac{e^{s_i}}{\sum_j e^{s_j}}$$

- $s_i$ — the raw attention score of token $i$

Turns arbitrary scores into positive weights that sum to 1 — a "where to look" budget.

Multi-head attention:

$$\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

- $h$ — number of heads, each with its own $W^Q_i, W^K_i, W^V_i$ over a slice $d_k = d_{model}/h$
- $W^O$ — output projection mixing the heads back together

Several attention patterns run in parallel and their findings are concatenated — committee, not single reader.

## Open the visualization

`open visualize.html in any browser`

## When to use this

- **Anything language** — translation, summarization, chat, code generation; every serious LLM since 2018 (BERT, GPT, Llama, Claude) is a transformer.
- **Long-range dependency problems where RNNs choke** — document-level classification, protein folding (AlphaFold's Evoformer), DNA sequence modeling: signal must hop 10,000 positions without degrading.
- **Set- and graph-shaped data** — vision (ViT treats image patches as tokens), recommendation (user history as a sequence), multimodal models; attention doesn't care that the "sequence" isn't text.

## What breaks it

- **Quadratic cost in sequence length** — attention is $O(n^2)$ in time and memory; doubling context length quadruples cost. A 1M-token document doesn't fit; you need FlashAttention, sliding windows, or sparse variants, each trading away something.
- **Small data** — transformers have weak inductive biases (no built-in locality or recurrence), which is their superpower at scale and their downfall on 5k examples; a fine-tuned small model or even gradient boosting will beat a from-scratch transformer there.
- **No positional signal** — pure attention is permutation-invariant: "dog bites man" and "man bites dog" are identical without positional encodings. Forget them (or extrapolate beyond trained positions) and word order vanishes from the model.

## 5 Interview Questions

**1. Conceptual — "Why did transformers replace RNNs?"**
Direct answer: parallelism and path length. RNNs process tokens one at a time (no parallel training) and signal between distant tokens must survive $O(n)$ sequential steps; attention connects any two tokens in $O(1)$ steps and the whole sequence computes as one matrix multiply. Reason: the $O(1)$ path length solves long-range dependency learning that gated RNNs only mitigated, and the parallelism is what made pretraining on internet-scale corpora economically feasible. Likely follow-up: *"What did we lose?"* — $O(n^2)$ memory/compute in sequence length and the natural streaming ability of RNNs, which is why state-space models (Mamba) are being explored as successors.

**2. Mathematical — "Why scale by √d_k in attention?"**
Direct answer: if query and key components are roughly unit-variance and independent, the dot product $q \cdot k$ has variance $d_k$; dividing by $\sqrt{d_k}$ restores variance 1 before the softmax. Reason: with variance $d_k$ (say 64–128), score gaps are huge, softmax becomes effectively argmax, and its gradient — which scales with $p_i(1-p_i)$ — collapses to near zero, freezing learning. Likely follow-up: *"Where else does this saturation logic appear?"* — sigmoid/tanh saturation in deep nets and temperature in softmax sampling; scaling is just temperature chosen to keep gradients alive.

**3. Practical — "Your fine-tuned transformer does great on short inputs and falls apart past 512 tokens. Diagnose."**
Direct answer: the model was pretrained or fine-tuned with a 512-token context — either positions beyond 512 were never learned (learned absolute embeddings have no entries there) or inputs are being silently truncated by the tokenizer. Reason: positional representations don't extrapolate by default; learned absolute embeddings fail hard, and even sinusoidal/RoPE degrade beyond the trained range. Likely follow-up: *"Fixes?"* — check tokenizer truncation first (cheapest bug), then use a long-context base model, RoPE scaling/interpolation, or chunk-and-aggregate the document.

**4. Gotcha — "Attention weights tell you which input tokens caused the prediction, right?"**
Direct answer: not reliably — attention weights are a mixing distribution, not an attribution method. Reason: a token can receive high attention but contribute a near-zero value vector (so it doesn't matter), information gets blended across positions in earlier layers (so by layer 12, "token 3" is no longer just token 3), and research ("Attention is not Explanation") shows you can often find different attention patterns yielding identical predictions. Likely follow-up: *"What would you use instead?"* — gradient-based attribution (integrated gradients), ablation/occlusion tests, or causal patching of the residual stream; treat attention maps as a debugging hint, not evidence.

**5. System design — "Design the serving stack for a chat LLM with 200ms p99 time-to-first-token."**
Direct answer: split prefill (parallel, compute-bound) from decode (sequential, memory-bound); use a KV cache so each new token attends to cached keys/values instead of recomputing the whole prefix; add continuous batching (vLLM-style) to share GPUs across requests, paged KV memory to avoid fragmentation, and quantization (INT8/FP8) for throughput. Reason: without a KV cache, generating token $t$ costs $O(t^2)$ — recomputing attention over the full prefix — making chat latency impossible; the KV cache makes each step $O(t)$, so cache memory becomes the real bottleneck and everything else in the design manages it. Likely follow-up: *"KV cache is blowing up GPU memory at long contexts — now what?"* — grouped-query attention (share K/V across head groups), sliding-window attention for old tokens, cache quantization, or offloading cold cache pages to CPU.
