# Convolutional Neural Network - Work in Progress
This is the working copy of a C++17 convolutional neural network (CNN), built up one layer at a
time across L08, L09, and L10. It's trained to classify four hand-drawn 4×4 pixel patterns (the
digits 0–3).

The four training patterns ([source/main.cpp](./source/main.cpp)), where `1` is a filled pixel and `0` is empty, are shown below:

```
Digit 0:      Digit 1:      Digit 2:      Digit 3:
1 1 1 1       0 1 0 0       1 0 0 1       0 0 1 0
1 0 0 1       0 1 0 0       0 1 1 0       0 0 1 0
1 0 0 1       0 1 0 0       0 1 1 0       1 1 1 1
1 1 1 1       0 1 0 0       1 0 0 1       0 0 1 0
```

---

## Current status
Not every layer is real yet: `source/ml/factory/factory.cpp` decides, per layer type, whether to
build the real class or a `Stub`, and each not-yet-built one is marked with a `//! @todo` comment
there:
* **`Conv`** (`include`/`source`/`ml/conv_layer/conv.{hpp,cpp}`): placeholder, see L08's
  [appendix A](../appendix/a_conv_layer.md).
* **`MaxPool`** (`include`/`source`/`ml/conv_layer/max_pool.{hpp,cpp}`): placeholder, see L09's
  [appendix A](../../L09/appendix/a_max_pool_layer.md).
* **`Flatten`** (`include`/`source`/`ml/flatten_layer/flatten.{hpp,cpp}`): placeholder, see L10's
  [appendix A](../../L10/appendix/a_flatten_layer.md).
* **`Dense`**: already real (built separately back in L05).

The project builds and runs as-is at every stage: with any layer still stubbed, `make run` trains
and prints predictions without error, they just won't be meaningful until every layer is real
(the end of L10). That's expected, not a bug.

---

## Structure

```
cnn_work/
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
`true`/`false` without computing anything. `ml::factory::Factory` picks, layer by layer, which one
to build; see `factory.cpp`.

**Data types** (`ml/types.hpp`): `Matrix1d = std::vector<double>`, `Matrix2d = std::vector<Matrix1d>`,
`Matrix3d = std::vector<Matrix2d>`.

---

## `dense_layer::Dense`
The one layer that's already real. A regular fully connected layer with weights and a bias.
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
size 1, no mini-batching). The training order is reshuffled every epoch. If `feedforward`,
`backpropagate`, or `optimize` fails for any example, training is aborted immediately.

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
* **`Dense::backpropagate()` reuses the same formula for the output layer and hidden layers.** For
  the last dense layer, `outputGradients` is literally the target vector, so
  `outputGradients[i] - myOutput[i]` becomes `target - prediction`. For earlier dense layers, what's
  passed in instead is the previous layer's already-computed `inputGradients()`; i.e. not a target
  vector. The same formula is still applied in both cases. That's how the code actually works, not
  a bug that needs fixing; but worth knowing when reading or reasoning about the backpropagation
  code.

---
