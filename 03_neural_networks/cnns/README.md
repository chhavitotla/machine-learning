# Convolutional Neural Networks

## The Intuition

A dense layer looking at an image is absurd: it gives every pixel its own private weight to every neuron, so a cat in the top-left corner and the same cat in the bottom-right are, to the network, unrelated problems. A convolution fixes this with one move — take a tiny stencil of weights (say 3×3), slide it across the whole image, and at every stop compute a dot product between the stencil and the pixels under it. The output is a *feature map*: a heat map of "how much does this patch look like my pattern?" A stencil with negative weights on the left and positive on the right fires exactly where dark meets light — a vertical-edge detector, in nine numbers. Because the same nine weights are reused at every position, the detector works anywhere in the image, and the layer needs nine parameters instead of millions. Stack these layers — with ReLU to keep only detections and max-pooling to summarize and shrink — and edge detectors compose into texture detectors, into eyes and wheels, into faces and cars. The network *learns* the stencils by gradient descent; nobody hand-designs them.

## The Math

The convolution (what frameworks actually compute — cross-correlation):

$$\text{out}_{i,j} = \sum_{u=0}^{k-1}\sum_{v=0}^{k-1} K_{u,v} \cdot X_{i \cdot s + u - p,\; j \cdot s + v - p}$$

- $K$ — the $k \times k$ kernel (the learned weights, shared across all positions)
- $X$ — the input image (out-of-bounds reads are 0 — that's zero padding)
- $s$ — the stride: how far the window jumps between stops
- $p$ — the padding: extra zero border so the output doesn't shrink

Each output cell is one dot product between the kernel and one image patch.

The output size:

$$d_{\text{out}} = \left\lfloor \frac{d_{\text{in}} + 2p - k}{s} \right\rfloor + 1$$

- "Valid" conv ($p=0$): the map shrinks by $k-1$. "Same" conv ($p=\lfloor k/2 \rfloor$, $s=1$): size is preserved.

The nonlinearity and the downsampling:

$$\text{ReLU}(x) = \max(0, x) \qquad \text{maxpool}(X)_{i,j} = \max_{u,v \in \{0,1\}} X_{2i+u,\, 2j+v}$$

- ReLU keeps positive responses ("pattern found") and zeroes the rest
- 2×2 max-pool keeps the strongest response per patch, halving each dimension and buying tolerance to small shifts

Parameter count is the punchline: a conv layer with $C_{in}$ input channels, $C_{out}$ filters of size $k$ has $C_{out}(k^2 C_{in} + 1)$ parameters — independent of image size. A dense layer on a 224×224×3 image needs ~150k weights *per neuron*.

## Open the visualization

`open visualize.html in any browser`

## When to use this

- **Anything image-shaped** — classification, detection, segmentation, medical imaging; convolution's translation equivariance is the right prior whenever the same pattern can appear anywhere in the input.
- **Audio and 1D signals** — spectrograms are images, and 1D convs over raw waveforms or sensor streams exploit the same local-pattern-anywhere structure.
- **As the cheap local-feature stage in hybrid models** — many vision transformers and speech models still open with a small conv stem because it's an efficient, hard-to-beat way to extract local features before global attention takes over.

## What breaks it

- **Global relationships between distant pixels** — a 3×3 kernel sees 3 pixels; understanding that two objects on opposite sides of an image are interacting needs many stacked layers to grow the receptive field, and the signal degrades on the way. This is precisely the gap attention was built to fill.
- **Rotation and scale** — weight sharing buys *translation* equivariance only. Train on upright faces and a 90°-rotated face is an out-of-distribution input; you're forced into data augmentation because the architecture has no built-in answer.
- **Aggressive pooling when precise location matters** — every max-pool throws away "where exactly"; stack five of them and a segmentation or keypoint model can't localize boundaries anymore, which is why dense-prediction architectures need skip connections (U-Net) or dilated convs to recover what pooling destroyed.

## 5 Interview Questions

**1. Conceptual — "Why does a CNN need far fewer parameters than a dense network on images?"**
Direct answer: two reasons — local connectivity (each output looks at a $k \times k$ patch, not the whole image) and weight sharing (the same kernel is reused at every spatial position), so parameter count depends on kernel size and channel counts, never on image resolution. Reason: this hard-codes the prior that useful image features are local and can appear anywhere, which is true of edges, textures, and parts — so you spend your data learning *what* patterns matter, not relearning them at each location. Likely follow-up: *"What inductive bias is this, formally?"* — translation equivariance: shift the input and the feature map shifts identically; pooling then converts that into approximate translation *invariance*.

**2. Mathematical — "Input 224×224, kernel 7×7, stride 2, padding 3. Output size? And how many parameters with 3 input channels and 64 filters?"**
Direct answer: $\lfloor(224 + 6 - 7)/2\rfloor + 1 = 112$, so 112×112; parameters $= 64 \times (7 \times 7 \times 3 + 1) = 9{,}472$. Reason: the size formula counts how many stride-2 stops fit, and each filter owns one $7{\times}7{\times}3$ weight block plus a bias — image size never enters. Likely follow-up: *"And the receptive field after stacking two 3×3 convs?"* — 5×5, which is why two 3×3 layers (18 weights/channel, two nonlinearities) replaced single 5×5 layers (25 weights, one nonlinearity) from VGG onward.

**3. Practical — "Your CNN gets 99% train accuracy and 70% validation accuracy on a 5,000-image dataset. What do you do?"**
Direct answer: it's memorizing — attack variance: heavy augmentation (crops, flips, color jitter), transfer learning from a pretrained backbone instead of training from scratch, add weight decay and dropout, and shrink the model. Reason: 5,000 images is far too few to learn good filters from random init, but a backbone pretrained on millions already has generic edge/texture layers, so you only fine-tune the top. Likely follow-up: *"Which layers do you freeze?"* — freeze the early layers (generic edges/textures transfer everywhere), fine-tune the later, task-specific ones; unfreeze more as your dataset grows.

**4. Gotcha — "Max-pooling has no parameters, so it doesn't affect what the network can learn, right?"**
Direct answer: wrong — pooling is parameter-free but changes the function class profoundly: it discards precise spatial information, enlarges the receptive field of every later layer, and injects local translation invariance. Reason: gradients flow only through the max element per window, so it also acts as a routing/selection mechanism during training; what the next layer can possibly distinguish has changed. Likely follow-up: *"Why do many modern architectures use strided convs instead?"* — strided convolution downsamples too but with *learned* weights deciding what to keep, and avoids pooling's gradient sparsity; ResNets and most modern nets made this trade.

**5. System design — "Design the vision model for a factory line that flags defective parts at 60 fps on an edge device."**
Direct answer: a small CNN (MobileNet-class, depthwise-separable convs), pretrained backbone fine-tuned on a few thousand labeled defect images with heavy augmentation, quantized to INT8, served on the device with a confidence threshold tuned for high recall and a human-review queue for the gray zone. Reason: the latency and power budget rules out big transformers; defects are local textural patterns — exactly what conv filters detect — and translation equivariance handles parts shifting on the belt. Likely follow-up: *"Defects are 1-in-10,000 — how do you train and evaluate?"* — treat it as anomaly detection or heavily oversample/synthesize defects, evaluate with precision-recall (not accuracy), and monitor for distribution drift when lighting or part suppliers change.
