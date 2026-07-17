# Appendix C - Exercises (Backpropagation & Optimization)
This continues the hand-training example from [appendix B](./b_exercises.md): work through it on
your own after the lecture. It's the same network: **4×4** input, **2×2** kernel (stride 1,
zero-padded, ReLU), **2×2** max pool (stride 2), flatten (2×2 → 1×4), learning rate `LR = 0.001`.

As a recap, here's what appendix B's feedforward pass produced:

**Padded input (6×6):**

```
0 0 0 0 0 0
0 1 1 1 1 0
0 1 0 0 1 0
0 1 0 0 1 0
0 1 1 1 1 0
0 0 0 0 0 0
```

**Kernel:** `[[0.2, 0.4], [0.6, 0.8]]`, **bias:** `0.5`.

**Conv layer output (4×4):**

```
1.3 1.9 1.9 1.9
1.7 1.7 1.1 1.9
1.7 1.3 0.5 1.7
1.7 2.1 1.9 2.3
```

**Max pooling output (2×2):**

```
1.9 1.9
2.1 2.3
```

**Flattened output (1×4):** `[1.9, 1.9, 2.1, 2.3]`

---

### 2. Backpropagation
Assume the dense layer sends back the following gradients:

```
[10, 20, 30, 40]
```

---

#### a) Flatten (backpropagation)
The gradients from the dense layer are reshaped back into a matrix:

```
10 20
30 40
```

---

#### b) Max pool (backpropagation)
The gradients are propagated back to the correct positions in the max pooling layer, i.e. to the spots where the max values were located in each pool. If a pool has two max values, the gradient is propagated back to the first position; the remaining positions get a gradient of 0.

The max pooling layer's input (the same conv output as above):

```
1.3 1.9 1.9 1.9
1.7 1.7 1.1 1.9
1.7 1.3 0.5 1.7
1.7 2.1 1.9 2.3
```

**Top-left corner:** gradient `10` is propagated to the max value `1.9` (top right):

```
0 10
0  0
```

**Top-right corner:** gradient `20` is propagated to the first occurrence of the max value `1.9` (top left):

```
20  0
0  0
```

**Bottom-left corner:** gradient `30` is propagated to the max value `2.1` (bottom right):

```
0  0
0 30
```

**Bottom-right corner:** gradient `40` is propagated to the max value `2.3` (bottom right):

```
0  0
0 40
```

Assembling all the pools' gradients into a single matrix gives us:

```
0 10 20  0
0  0  0  0
0  0  0  0
0 30  0 40
```

---

#### c) Conv layer (backpropagation)
Now we compute the gradient for the bias and kernel from the error propagated back.

**Bias gradient:** the sum of all values in the gradient matrix:

```math
biasGradient = 0 + 10 + 20 + 0 + 0 + 0 + 0 + 0 + 0 + 0 + 0 + 0 + 0 + 30 + 0 + 40 = 100
```

**Kernel gradients:** for each kernel element, we sum the product of the corresponding patch in the padded input image and the gradient matrix. Most gradients are zero, so we only need to compute the four positions where the gradient is non-zero.

Padded input image (6×6):

```
0 0 0 0 0 0
0 1 1 1 1 0
0 1 0 0 1 0
0 1 0 0 1 0
0 1 1 1 1 0
0 0 0 0 0 0
```

Gradient matrix from the pooling layer (4×4):

```
0 10 20  0
0  0  0  0
0  0  0  0
0 30  0 40
```

For each position (i, j) in the gradient matrix, the corresponding 2×2 patch is extracted from the input image. Each kernel element is multiplied by the corresponding value in the patch and summed over all positions:

```math
k00 = 0 \cdot 10 + 0 \cdot 20 + 1 \cdot 30 + 0 \cdot 40 = 30
```
```math
k01 = 0 \cdot 10 + 0 \cdot 20 + 0 \cdot 30 + 1 \cdot 40 = 40
```
```math
k10 = 1 \cdot 10 + 1 \cdot 20 + 1 \cdot 30 + 1 \cdot 40 = 100
```
```math
k11 = 1 \cdot 10 + 1 \cdot 20 + 1 \cdot 30 + 1 \cdot 40 = 100
```

The kernel gradients are therefore:

```
30  40
100 100
```

**Input gradients (padded input):** we compute how the error propagates back to the input by, for each position in the gradient matrix, "spreading" the kernel's weights multiplied by the gradient value at the
right spot into a new 6×6 matrix, and summing overlapping positions.

Only four gradients are non-zero, so we only need to compute their contributions:

* From `grad[0,1] = 10`: `dX(0,1) += 2`, `dX(0,2) += 4`, `dX(1,1) += 6`, `dX(1,2) += 8`
* From `grad[0,2] = 20`: `dX(0,2) += 4`, `dX(0,3) += 8`, `dX(1,2) += 12`, `dX(1,3) += 16`
* From `grad[3,1] = 30`: `dX(3,1) += 6`, `dX(3,2) += 12`, `dX(4,1) += 18`, `dX(4,2) += 24`
* From `grad[3,3] = 40`: `dX(3,3) += 8`, `dX(3,4) += 16`, `dX(4,3) += 24`, `dX(4,4) += 32`

After summing all contributions, we get the padded input gradient matrix:

```
0  2  8  8  0  0
0  6 20 16  0  0
0  0  0  0  0  0
0  6 12  8 16  0
0 18 24 24 32  0
0  0  0  0  0  0
```

Removing the outermost row and column (padding) leaves a 4×4 matrix matching the original image; this is the gradient with respect to the input, which is passed further back through the network:

```
6  20 16  0
0   0  0  0
6  12  8 16
18 24 24 32
```

---

#### d) Optimization
Finally, we update the kernel and bias using the learning rate `LR`:

```
kernel = kernel + LR * kernelGradient
bias   = bias   + LR * biasGradient
```

**Updated kernel:**

```
0.23 0.44
0.70 0.90
```

**Updated bias:**

```
0.6
```

---
