# Appendix B - Exercises
These exercises are for self-assessment, reviewing the whole course:
* **Part 1** tests whether you can recognize a layer type from its algorithm alone.
* **Part 2** tests implementation-level understanding: shapes, parameter counts, and what has to be remembered for backpropagation.
* **Part 3** is a short set of exam-style conceptual questions.

Try to answer everything yourself first. Solutions are provided in [appendix D](./d_solutions.md).

---

## Part 1 - Recognize the Layer
Below are six feedforward algorithms, written as pseudocode. For each one, decide which layer type
it describes: a linear regression layer, a dense layer, a conv layer, a max pooling layer, or a
flatten layer.

### A

```python
output = k * input + m
```

### B

```python
for n in nodes:
    sum = bias[n]
    for i in inputs:
        sum += weight[n][i] * input[i]
    output[n] = activation(sum)
```

### C

```python
for row in range(outputHeight):
    for col in range(outputWidth):
        sum = bias
        for kr in range(kernelSize):
            for kc in range(kernelSize):
                sum += kernel[kr][kc] * input[row + kr][col + kc]
        output[row][col] = activation(sum)
```

### D

```python
for row in range(outputHeight):
    for col in range(outputWidth):
        window = input[row * poolSize : row * poolSize + poolSize,
                        col * poolSize : col * poolSize + poolSize]
        output[row][col] = max(window)
```

### E

```python
output = reshape(input, (1, height * width))
```

### F

```python
sum = bias[0]
for i in inputs:
    sum += weight[0][i] * input[i]
output[0] = sum   // no activation function applied
```

---

## Part 2 - Implementation Questions
Consider the following network, processing a single-channel **8×8** input image:

```
input (8×8) → conv layer (one 3×3 kernel, stride 1, zero-padded) → max pooling layer (2×2, stride 2) → flatten layer → dense layer (10 nodes)
```

### 1. Tracing shapes
What is the shape of the data after each layer (conv, max pooling, flatten, dense)?

### 2. Counting trainable parameters
How many trainable parameters does each layer have? What is the total for the whole network?

### 3. What must be remembered for backpropagation?
The max pooling layer has no trainable parameters, yet it still needs to store something during
`feedforward()` so that `backpropagate()` works correctly. What does it need to remember, and why?

### 4. Conv vs. dense, parameter count
A dense layer connecting the flattened 8×8 image (64 values) directly to a single output node
would need 64 weights + 1 bias = 65 trainable parameters just to combine the whole image once. The
conv layer in the network above combines local pixel neighborhoods using only how many trainable
parameters? Why is the difference so large?

### 5. Flatten in reverse
During `feedforward()`, a flatten layer reshapes a **3×4** matrix into a **1×12** vector. During
`backpropagate()`, it receives a **1×12** gradient vector from the next layer. What shape does it
reshape that gradient vector back into, and why?

---

## Part 3 - Exam-Style Questions

### True or false?
For each statement, decide if it's true or false, and briefly justify your answer.
1. A max pooling layer has trainable parameters that are updated during `optimize()`.
2. The same kernel (the same weights) is reused at every spatial position in a conv layer.
3. A flatten layer's `backpropagate()` performs the exact same reshape operation as its `feedforward()`.
4. A dense layer with exactly one node, no activation function, and no hidden layer around it
   behaves identically to a linear regression layer.
5. Adding more filters (kernels) to a conv layer increases its trainable parameter count, but
   doesn't by itself change the width/height of its output.

### Short answer
6. In one or two sentences, explain why swapping a dense layer for a conv layer generally means
   far fewer trainable parameters for the same size of input image.
7. A student claims: *"Max pooling layers slow down training because they have so many weights to
   update."* Is this correct? Why or why not?
8. Every layer type in this course implements the same three methods: `feedforward()`,
   `backpropagate()`, and `optimize()`. What does `optimize()` do for a layer type that has no
   trainable parameters, such as a max pooling or flatten layer?

---

Solutions for all three parts are in [appendix D](./d_solutions.md).

---
