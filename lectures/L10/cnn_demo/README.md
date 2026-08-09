# Complete Convolutional Neural Network
This is a complete, working C++17 implementation of a convolutional neural network (CNN) trained
to classify four hand-drawn 4×4 pixel patterns (the digits 0-3).

The four training patterns ([source/main.cpp](./source/main.cpp)), where `1` is a filled pixel and `0` is empty, are shown below:

```
Digit 0:      Digit 1:      Digit 2:      Digit 3:
1 1 1 1       0 1 0 0       1 0 0 1       0 0 1 0
1 0 0 1       0 1 0 0       0 1 1 0       0 0 1 0
1 0 0 1       0 1 0 0       0 1 1 0       1 1 1 1
1 1 1 1       0 1 0 0       1 0 0 1       0 0 1 0
```

---

## Structure

```
cnn_demo/
├── Makefile
├── source/
│   ├── main.cpp
│   └── ml/
│       ├── cnn/cnn.cpp
│       ├── conv_layer/{conv,max_pool}.cpp
│       ├── dense_layer/dense.cpp
│       ├── flatten_layer/flatten.cpp
│       ├── factory/factory.cpp
│       ├── random/generator.cpp
│       └── utils.cpp
└── include/ml/
    ├── types.hpp, utils.hpp
    ├── act_func/{interface,relu,tanh,none,type}.hpp
    ├── cnn/{interface,cnn}.hpp
    ├── conv_layer/{interface,conv,max_pool,stub}.hpp
    ├── dense_layer/{interface,dense,stub}.hpp
    ├── flatten_layer/{interface,flatten,stub}.hpp
    ├── factory/{interface,factory,stub}.hpp
    └── random/{interface,generator}.hpp
```

Each layer type (`conv_layer`, `dense_layer`, `flatten_layer`) follows the same pattern: an
`Interface` that defines the contract, a concrete class that does the real computation (`Conv`,
`MaxPool`, `Dense`, `Flatten`), and a `Stub` variant that only validates dimensions and returns
`true`/`false` without computing anything. `ml::factory::Factory` builds the real layers,
`ml::factory::Stub` builds the stub variants; which one is used is chosen at runtime via
`ml::factory::create(bool stub = false)`, not via compiler flags.

**Data types** (`ml/types.hpp`): `Matrix1d = std::vector<double>`, `Matrix2d = std::vector<Matrix1d>`,
`Matrix3d = std::vector<Matrix2d>`.

---

## The Layers

### `conv_layer::Conv`
A simple conv layer implementation with a single kernel and a bias, using "same" padding (the
output has the same spatial size as the input).
* **Constructor:** takes the input size, kernel size (1-11), and an activation function.
  Initializes the kernel and bias with random starting values.
* **`feedforward()`:** zero-pads the input, and computes for each output cell
  `sum = bias + Σ kernel[i][j] * padded_input[...]`, followed by the activation function.
* **`backpropagate()`:** receives error gradients from the next layer, accumulates the bias and
  kernel gradients, and computes gradients for the padded input, which are then "unpadded" into
  `inputGradients()` for the previous layer.
* **`optimize()`:** updates the kernel and bias with `gradient * learningRate`.

### `conv_layer::MaxPool`
Non-overlapping max pooling, defined in the same namespace (`ml::conv_layer`) since it shares an
interface with `Conv`.
* **Constructor:** takes the input size and pool size (the input size must be evenly divisible by
  the pool size).
* **`feedforward()`:** selects the max value in each non-overlapping block.
* **`backpropagate()`:** looks up which cell in each block produced the max value during
  feedforward, and routes the entire error gradient there; the remaining cells get no gradient.
* **`optimize()`:** does nothing (just returns `true`); the layer has no trainable parameters.

### `flatten_layer::Flatten`
Reshapes a 2D matrix into a 1D vector (and back during backpropagation). Has no trainable
parameters, and therefore has no `optimize()` method at all in its interface.

### `dense_layer::Dense`
A regular fully connected layer with weights and a bias.
* **`feedforward()`:** `sum = bias[i] + Σ weight[i][j] * input[j]` per node, followed by the
  activation function.
* **`backpropagate()`:** computes `error[i] = (outputGradients[i] - myOutput[i]) *
  activationFunction.delta(preActivationSum[i])`. See the note below on how `outputGradients` is
  interpreted differently depending on whether the layer is the output layer or a hidden layer.
* **`optimize()`:** updates the weights and bias with `error * learningRate * input`.

---

## `ml::cnn::Cnn` - How the Layers Connect
The constructor creates and connects the layers via the injected factory:
1. A `Conv` layer and a `MaxPool` layer are added to the same list, `myConvLayers` (in that order).
2. A `Flatten` layer is created based on the max-pool layer's output size.
3. A `Dense` layer is created based on the flatten layer's output size.

**`feedforward(input)`:** `input → Conv → MaxPool → Flatten → Dense → prediction`. Each layer is
fed the previous layer's `output()`.

**`backpropagate(target)`:** goes the same path in reverse. The target vector (`target`) is passed
directly into the last dense layer's `backpropagate()`: that's where
`(outputGradients[i] - myOutput[i])` actually becomes `(target[i] - prediction[i])`. Each preceding
layer is then fed the next layer's `inputGradients()`: the dense layers in reverse order, then the
flatten layer, then `MaxPool`, and finally `Conv`.

**`optimize(learningRate)`:** updates every layer with the gradients already computed by
`backpropagate()`. The dense layers also need the previous layer's `output()` (the value actually
used during feedforward) as input for the weight update.

**`train(trainIn, trainOut, epochCount, learningRate)`:** pure stochastic gradient descent (batch
size 1, no mini-batching). The training order is reshuffled every epoch. If `feedforward()`,
`backpropagate()`, or `optimize()` fails for any example, training is aborted immediately.

**`predict(input)`:** runs `feedforward()` and returns the last dense layer's `output()`.

---

## Activation Functions

| Function | `output(x)` | `delta(x)` |
|---|---|---|
| `Relu` | `x` if `x > 0`, otherwise `0` | `1` if `x > 0`, otherwise `0` |
| `Tanh` | `tanh(x)`, range `[-1, 1]` | `1 - tanh(x)²` |
| `None` | `x` (identity) | `1` |

`delta()` is always computed from the pre-activation sum (the value before the activation function
was applied), not from the already-activated output value.

---

## Random Number Generator
`random::Generator` is a singleton that initializes `std::srand` once, on first use. It's used both
to randomize initial weight and bias values (via `randomStartVal()` in `Conv` and `Dense`), and to
reshuffle the training order every epoch.

---

## Build and Run

```bash
make
```

This builds and runs the program directly (the `build` target followed by the `run` target).
`clean` removes the built binary.

---

## Implementation Details Worth Knowing
* **The `-DSTUB` flag in the Makefile currently has no effect.** The comment in the Makefile
  suggests it switches to the stub implementations, but there's no `#ifdef STUB` anywhere in the
  code that reacts to the flag. The stub classes exist and can be used (via
  `ml::factory::create(true)`), but `main.cpp` always calls `ml::factory::create()` with no
  arguments, which gives the real implementation. So the stub layers are never used in this demo.
* **`Dense::backpropagate()` reuses the same formula for the output layer and hidden layers.** For
  the last dense layer, `outputGradients` is literally the target vector, so
  `outputGradients[i] - myOutput[i]` becomes `target - prediction`, which is the deviation the
  course defines. For earlier dense layers, what's passed in instead is the next layer's
  already-computed `inputGradients()`, which is a gradient, not a target. Subtracting `myOutput[i]`
  from it adds a term that isn't part of the gradient of any loss, so for a hidden layer this is a
  deliberate simplification rather than the exact derivative. The networks in this course still
  train, because the term is small next to the one that carries the signal, and keeping one formula
  keeps the layer readable. Worth knowing before you compare this code against a textbook, or
  against a framework, and find they don't agree.

---
