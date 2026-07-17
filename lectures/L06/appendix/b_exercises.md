# Appendix B - Exercises

## Hand-Training a CNN (Feedforward)
This example is worked through together during the lecture.

We have the following convolutional neural network:
* Input: **4×4**
* Conv layer:
  * Kernel: **2×2**
  * Stride = 1 *(the kernel moves one pixel at a time)*
  * Zero-padding *(we add zeros around the image so its size is preserved despite the extraction)*
  * Activation: **ReLU** *(all negative values are replaced with zero)*
  * Learning rate: **LR** = 0.001, i.e. 1 %.
* Max pool:
  * Pool size = 2
  * Stride = 2 *(the pools are non-overlapping)*
* Flatten:
  * 2×2 → 1×4

---

The figure below demonstrates the training process:

![](./images/conv_exercise.png)

We work through the feedforward pass below. Backpropagation and optimization for this same example
continue in [appendix C](./c_backprop_exercises.md) as self-study.

### 1. Passing Through the Conv Layer

#### a) Padding the input image
Assume we have the following input image, resembling the digit 0 made up of ones:

```
1 1 1 1
1 0 0 1
1 0 0 1
1 1 1 1
```

The first step is to use *zero-padding*: we add zeros around the image so its size is preserved even after the kernel filter has been applied. This way, the conv layer's output has the same dimensions as the input, **4×4** in this case.

The number of zeros `n` we add on each side is computed as:

```
n = kernel_size / 2
```

where `kernel_size` is the size of the kernel, `2` in this example, i.e. one zero on each side. The padded image then becomes **6×6**:

```
0 0 0 0 0 0
0 1 1 1 1 0
0 1 0 0 1 0
0 1 0 0 1 0
0 1 1 1 1 0
0 0 0 0 0 0
```

---

#### b) Kernel and bias
We use a kernel (a filter of weights) and a bias. Here we've chosen simple, fixed values to make the calculations easy to follow:

**Kernel weights:**

```
0.2 0.4
0.6 0.8
```

**Bias:**

```
0.5
```

---

#### c) Feedforward through the conv layer
For each position in the image, the sum of the element-wise multiplication between the kernel and the corresponding part of the image is computed, plus the bias. ReLU activation is then applied (all negative values are replaced with zero):

```
sum = a * k00 + b * k01 + c * k10 + d * k11 + bias
output = ReLU(sum)
```

where:
* **a, b, c, d** are pixel values from the current 2×2 section of the image.
* **k** stands for kernel (the filter), e.g. `k00` means the value at row 0, column 0 of the kernel matrix, i.e. the top-left corner.
* **bias** is a constant value added to the sum.

Once we've applied the kernel and bias across the entire image, we get the following conv output (4×4):

```
1.3 1.9 1.9 1.9
1.7 1.7 1.1 1.9
1.7 1.3 0.5 1.7
1.7 2.1 1.9 2.3
```

---

#### d) The max pooling layer (feedforward)
The max pooling layer looks for the largest value in each 2×2 pool (non-overlapping) from the conv layer's output. The purpose is to downsample the image: we remove detail while keeping the most important features, which reduces the amount of data and makes the model less sensitive to small variations.

We split the conv layer's output into four pools:

**Top-left corner:** max value `1.9`:

```
1.3 1.9
1.7 1.7
```

**Top-right corner:** the max value `1.9` occurs in three places; we pass forward the first
instance:

```
1.9 1.9
1.1 1.9
```

**Bottom-left corner:** max value `2.1`:

```
1.7 1.3
1.7 2.1
```

**Bottom-right corner:** max value `2.3`:

```
0.5 1.7
1.9 2.3
```

The max pooling layer's output is therefore:

```
1.9 1.9
2.1 2.3
```

---

#### e) Flatten (feedforward)
The max pooling layer's output is flattened into a vector so it can be fed to the next layer:

```
[1.9, 1.9, 2.1, 2.3]
```

---

Continue with backpropagation and optimization for this example in [appendix C](./c_backprop_exercises.md).

---
