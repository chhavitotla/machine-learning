"""Convolution, ReLU, max-pooling from scratch — pure standard library.

Run it:  python3 implementation.py

The three ops that make a CNN:
    conv2d   — slide a small kernel over the image, dot product at each stop
    relu     — clamp negatives to zero (keep only "detections")
    maxpool  — downsample by keeping the strongest response in each 2x2 patch

The demo applies a vertical-edge kernel to a tiny image whose left half is
dark and right half is bright, and the feature map lights up exactly along
the boundary. Feature extraction, no magic.
"""


def conv2d(image, kernel, stride=1, padding=0):
    """2D convolution (technically cross-correlation, like every DL framework).

    For each position of the sliding window, the output value is the dot
    product of the kernel with the patch of image under it:
        out[i][j] = sum_over(u,v) kernel[u][v] * image[i*stride+u][j*stride+v]

    padding=0 is a "valid" convolution (output shrinks by kernel-1);
    padding=1 with a 3x3 kernel keeps the output the same size as the input.
    """
    if padding:
        w = len(image[0]) + 2 * padding
        z = [0.0] * padding
        image = ([[0.0] * w] * padding
                 + [z + row + z for row in image]
                 + [[0.0] * w] * padding)
    kh, kw = len(kernel), len(kernel[0])
    out_h = (len(image) - kh) // stride + 1
    out_w = (len(image[0]) - kw) // stride + 1
    out = []
    for i in range(out_h):
        row = []
        for j in range(out_w):
            acc = 0.0
            for u in range(kh):                 # the 9 multiplications
                for v in range(kw):             # (for a 3x3 kernel)
                    acc += kernel[u][v] * image[i * stride + u][j * stride + v]
            row.append(acc)
        out.append(row)
    return out


def relu(fmap):
    """max(0, x) elementwise: keep positive responses, kill the rest."""
    return [[max(0.0, x) for x in row] for row in fmap]


def maxpool2d(fmap, size=2):
    """Keep the strongest activation in each size x size patch.

    Halves each dimension (for size=2) and buys translation tolerance:
    the edge can shift a pixel and the pooled map barely changes.
    """
    out = []
    for i in range(0, len(fmap) - size + 1, size):
        row = []
        for j in range(0, len(fmap[0]) - size + 1, size):
            row.append(max(fmap[i + u][j + v]
                           for u in range(size) for v in range(size)))
        out.append(row)
    return out


def show(title, grid):
    """Print a grid as aligned numbers."""
    print(f"{title}  ({len(grid)}x{len(grid[0])})")
    for row in grid:
        print("   " + " ".join(f"{x:5.1f}" for x in row))
    print()


if __name__ == "__main__":
    # An 8x8 image: dark (0) on the left, bright (1) on the right.
    # The only structure is one vertical edge between columns 3 and 4.
    image = [[0.0] * 4 + [1.0] * 4 for _ in range(8)]

    # The classic vertical-edge (Sobel-style) kernel: negative weights on
    # the left column, positive on the right. It computes "right minus
    # left" — large output exactly where brightness jumps left-to-right.
    kernel = [[-1.0, 0.0, 1.0],
              [-2.0, 0.0, 2.0],
              [-1.0, 0.0, 1.0]]

    show("input image", image)
    fmap = conv2d(image, kernel)            # valid conv: 8x8 -> 6x6
    show("feature map = conv2d(image, vertical-edge kernel)", fmap)
    activated = relu(fmap)
    show("after ReLU (negatives clipped)", activated)
    pooled = maxpool2d(activated)           # 6x6 -> 3x3
    show("after 2x2 max-pool", pooled)

    print("The feature map fires (value 4.0) only in the columns straddling")
    print("the dark->bright boundary, and pooling keeps that detection while")
    print("shrinking the map 4x. That's a CNN layer: detect, then summarize.")
