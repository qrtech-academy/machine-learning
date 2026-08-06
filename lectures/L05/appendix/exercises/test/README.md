# L05 Test Suite
Unit tests for the dense layer and the utility functions built in
[appendix B](../../b_exercises.md), written with the
[`QAcademy Test`](https://github.com/qrtech-academy/test-framework) framework.

This suite **supersedes the L04 one**: it carries `stub_test.cpp` and `shallow_test.cpp` over
unchanged and adds `ml/dense_layer/dense_test.cpp` and `ml/utils_test.cpp`. The stub doesn't retire
when `Dense` arrives, it's still what the network tests run against. Once you've carried your code
forward into this lecture's exercises directory, this is the only suite you need to run.

---

## Recovering the Bias
`Dense` randomizes its own bias and weights, so nothing it computes can be predicted from the
constructor arguments. `weights()` is exposed, but there's no `bias()`, so at first glance only
shapes and ranges look testable.

The way in is to feed the layer a vector of zeros. Every weight term drops out:

```
sum = myBias[i] + Σ myWeights[i][j] * 0.0  ==  myBias[i]
```

With `ActFunc::None` the activation is the identity, so `output()` after that call **is** the bias
vector. Combined with `weights()`, every expected value in these tests is then computed by hand and
compared exactly, rather than checked for being merely plausible. `recoverBias()` at the top of
[dense_test.cpp](./ml/dense_layer/dense_test.cpp) does this, and the `Tanh` tests invert
`std::tanh()` with `std::atanh()` to do the same through a bounded activation function.

---

## Setting Up
There's nothing to copy or clone. This `test` directory already sits inside the exercises
directory you write your code in, and the test framework is checked out as a submodule at the root
of this repository:

```
exercises/
├── Makefile
├── include/
│   └── ml/
├── source/
│   ├── main.cpp
│   └── ml/
└── test/                <- this directory
```

If `libs/test` at the repository root is empty, run this once to fetch the framework:

```bash
git submodule update --init
```

---

## Building and Running the Tests
Thanks to the included [Makefile](./Makefile), the tests can be built and run with the following
command (in this directory):

```bash
make
```

`make build`, `make run`, and `make clean` do what their names suggest. The test suite exits with
code 0 when everything passes, and -1 otherwise.

---

## What's Covered

### `ml/utils.hpp`

| Test                     | Covers                                      |
|--------------------------|---------------------------------------------|
| `RandomStartValRange`    | every draw is finite and inside `[0.0, 1.0]` |
| `RandomStartValVaries`   | the draws aren't one repeated number        |
| `ActFuncOutputRelu`      | passes positives through, clamps the rest   |
| `ActFuncOutputTanh`      | matches `std::tanh`, stays in `[-1.0, 1.0]` |
| `ActFuncOutputNone`      | returns its input unchanged                 |
| `ActFuncDeltaRelu`       | `1.0` above zero, `0.0` at and below        |
| `ActFuncDeltaTanh`       | `1 - tanh²`, and `0.0` once saturated       |
| `ActFuncDeltaNone`       | `1.0` everywhere                            |

The range is closed at both ends: `randomStartVal()` divides by `RAND_MAX`, so a draw of exactly
`RAND_MAX` gives exactly `1.0`.

### `ml::dense_layer::Dense`

| Test                                       | Covers                                        |
|--------------------------------------------|-----------------------------------------------|
| `ConstructedDimensions`                    | reports the dimensions it was built with      |
| `ConstructedMatrixShapes`                  | output, error, and weights are right-shaped   |
| `ConstructedParametersAreRandomized`       | bias and weights in `[0.0, 1.0]`, and varying |
| `LayersAreIndependentlyRandomized`         | two layers don't share parameters             |
| `FeedforwardComputesWeightedSum`           | output equals bias plus the weighted sum      |
| `FeedforwardAppliesActivationFunction`     | the activation function is applied to the sum |
| `FeedforwardIsDeterministic`               | same input, same output                       |
| `FeedforwardDoesNotTrain`                  | feedforward alone changes no parameter        |
| `BackpropagateOutputLayerComputesError`    | error equals reference minus output           |
| `BackpropagateUsesPreActivationDerivative` | the derivative is taken at the weighted sum   |
| `BackpropagateHiddenLayerComputesError`    | error from the next layer's error and weights |
| `OptimizeUpdatesBiasAndWeights`            | each delta is error x rate x input            |
| `OptimizeRejectedLeavesParametersUnchanged`| a rejected call touches nothing               |
| `FeedforwardChecksInputSize`               | input holds one value per **weight**          |
| `BackpropagateChecksReferenceSize`         | references hold one value per **node**        |
| `BackpropagateChecksNextLayerWeightCount`  | the next layer holds one weight per node      |
| `OptimizeChecksInputSize`                  | input holds one value per weight              |
| `OptimizeChecksLearningRate`               | the rate lies inside `(0.0, 1.0)`             |
| `UsableThroughInterface`                   | works through the interface                   |
| `NetworkLearnsXorPattern`                  | two layers in a network actually converge     |

`BackpropagateUsesPreActivationDerivative` is the one worth reading. It's the mistake appendix B
warns about, and it can only be caught with `ActFunc::Tanh`: for `Relu` the two values agree, since
`max(0.0, s)` is positive exactly when `s` is, and for `None` the derivative is `1.0` either way.
Both `backpropagate()` overloads are checked this way for the same reason.

`NetworkLearnsXorPattern` trains a `Shallow` with two `Dense` layers on 2-bit XOR, which isn't
linearly separable, so reaching the target is only possible if the hidden layer does real work. It
asserts against **both** a fixed error threshold and the network's own error before training, since
the threshold alone would pass for a network that started out lucky.

---

## Choosing the Convergence Threshold
The threshold is `0.05` mean absolute error after 5000 epochs with 8 hidden `Tanh` nodes. That
combination was measured over 500 runs before being written down:

* worst post-training error: **0.0092**
* best *untrained* error: **0.443**

A factor of five of margin below, and a factor of nearly fifty above.

`Tanh` rather than `Relu`, and 8 hidden nodes rather than 2, for a reason worth knowing: the same
network built from two hidden `Relu` nodes reaches that error in only about 72% of runs, because
`randomStartVal()` never returns a negative number and ReLU struggles with XOR from an
all-positive start. Appendix B section 7 invites you to try exactly that and watch it fail. Fine
for an experiment you re-run by hand, hopeless as a test.

---

## Not Covered
The constructor's `std::terminate()` paths aren't tested, for the same reason as in L03 and L04:
`std::terminate()` takes the whole process down, so testing it means running the binary in a
separate process and inspecting how it died. Verify by hand:

```cpp
// Feed the layer a node count of zero.
// Expect the program to terminate.
ml::dense_layer::Dense layer{0U, 3U};

// Feed the layer a weight count of zero.
// Expect the program to terminate.
ml::dense_layer::Dense layer2{3U, 0U};
```

---

## Adding New Test Files
Add a new test file to the build by adding its path to `TEST_FILES` in the [Makefile](./Makefile).
For example, if a file named `ml/dense_layer/adaptive_test.cpp` is added to this directory, add the
corresponding line as shown below:

```makefile
# Test files - update this list as new test files are added to this directory.
TEST_FILES := ml/dense_layer/adaptive_test.cpp \ # Added 'ml/dense_layer/adaptive_test.cpp'.
              ml/dense_layer/dense_test.cpp \
              ml/dense_layer/stub_test.cpp \
              ml/neural_network/shallow_test.cpp \
              ml/utils_test.cpp \
              testsuite.cpp \
```

---
