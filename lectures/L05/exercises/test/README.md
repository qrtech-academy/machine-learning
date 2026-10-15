# L05 Test Suite
Unit tests for the dense layer and the utility functions built in
[appendix B](../../appendix/b_exercises.md), written with the
[`QAcademy Test`](https://github.com/qrtech-academy/test-framework) framework.

This suite **supersedes the L04 one**: it carries `stub_test.cpp` over unchanged, updates
`shallow_test.cpp` for the new return value of `train()` (see
[appendix B](../../appendix/b_exercises.md), section 7), and adds `ml/dense_layer/dense_test.cpp`
and `ml/utils_test.cpp`. The stub doesn't retire
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

`Relu` needs one more step. The bias can start negative, and ReLU clamps a negative sum to zero, so
a zero input loses it. `recoverReluBias()` instead feeds each node an input that lifts its weighted
sum by a known amount, far above any start value. The sum is then positive, ReLU passes it through
unchanged, and subtracting the lift leaves the bias.

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

The suite holds 64 test cases: 14 for the stub, carried over from L04, 19 for the network, 22 for
`Dense` and 9 for the helpers in `ml/utils`.

---

## What's Covered

### `ml/utils.hpp`

| Test                            | Covers                                         |
|---------------------------------|------------------------------------------------|
| `RandomStartValRange`           | every draw is finite and inside `[-1.0, 1.0]`  |
| `RandomStartValCoversBothSigns` | the draws include negative and positive values |
| `RandomStartValVaries`          | the draws aren't one repeated number           |
| `ActFuncOutputRelu`             | passes positives through, clamps the rest      |
| `ActFuncOutputTanh`             | matches `std::tanh`, stays in `[-1.0, 1.0]`    |
| `ActFuncOutputNone`             | returns its input unchanged                    |
| `ActFuncDeltaRelu`              | `1.0` above zero, `0.0` at and below           |
| `ActFuncDeltaTanh`              | `1 - tanh²`, and `0.0` once saturated          |
| `ActFuncDeltaNone`              | `1.0` everywhere                               |

The range is closed at both ends: `randomStartVal()` divides by `RAND_MAX` and rescales with
`2.0 * x - 1.0`, so a draw of exactly `0` gives exactly `-1.0`, and a draw of exactly `RAND_MAX`
gives exactly `1.0`. `RandomStartValCoversBothSigns` is what catches a missing rescale: a function
returning `[0.0, 1.0]` passes the range check, since that range lies inside `[-1.0, 1.0]`.

### `ml::dense_layer::Dense`

| Test                                        | Covers                                         |
|---------------------------------------------|------------------------------------------------|
| `ConstructedDimensions`                     | reports the dimensions it was built with       |
| `ConstructedMatrixShapes`                   | output, error, and weights are right-shaped    |
| `ConstructedParametersAreRandomized`        | bias and weights in `[-1.0, 1.0]`, and varying |
| `ConstructedParametersCoverBothSigns`       | bias and weights include both signs            |
| `LayersAreIndependentlyRandomized`          | two layers don't share parameters              |
| `InitParamsDrawsNewValues`                  | the reset replaces bias and weights in place   |
| `FeedforwardComputesWeightedSum`            | output equals bias plus the weighted sum       |
| `FeedforwardAppliesActivationFunction`      | the activation function is applied to the sum  |
| `FeedforwardIsDeterministic`                | same input, same output                        |
| `FeedforwardDoesNotTrain`                   | feedforward alone changes no parameter         |
| `BackpropagateOutputLayerComputesError`     | error equals reference minus output            |
| `BackpropagateUsesPreActivationDerivative`  | the derivative is taken at the weighted sum    |
| `BackpropagateHiddenLayerComputesError`     | error from the next layer's error and weights  |
| `OptimizeUpdatesBiasAndWeights`             | each delta is error x rate x input             |
| `OptimizeRejectedLeavesParametersUnchanged` | a rejected call touches nothing                |
| `FeedforwardChecksInputSize`                | input holds one value per **weight**           |
| `BackpropagateChecksReferenceSize`          | references hold one value per **node**         |
| `BackpropagateChecksNextLayerWeightCount`   | the next layer holds one weight per node       |
| `OptimizeChecksInputSize`                   | input holds one value per weight               |
| `OptimizeChecksLearningRate`                | the rate lies inside `(0.0, 1.0)`              |
| `UsableThroughInterface`                    | works through the interface                    |
| `NetworkLearnsXorPattern`                   | two layers in a network actually converge      |

`BackpropagateUsesPreActivationDerivative` is the one worth reading. It's the mistake appendix B
warns about, and it can only be caught with `ActFunc::Tanh`: for `Relu` the two values agree, since
`max(0.0, s)` is positive exactly when `s` is, and for `None` the derivative is `1.0` either way.
Both `backpropagate()` overloads are checked this way for the same reason.

`InitParamsDrawsNewValues` is the first test in the course where the parameter reset has anything
to do. The stub tests in L03 and L04 only check that calling `initParams()` changes nothing, which
is all an empty override can be asked for; here the values it writes are read back out of
`weights()` and out of a zero-input feedforward, which recovers the bias.

`NetworkLearnsXorPattern` trains a `Shallow` with two `Dense` layers on 2-bit XOR, which isn't
linearly separable, so reaching the target is only possible if the hidden layer does real work. It
asserts against **both** a fixed error threshold and the network's own error before training, since
the threshold alone would pass for a network that started out lucky. It also checks the precision
`train()` returns against `1.0` minus the error it measures itself, so a `train()` that reports a
number it never measured fails here.

### `ml::neural_network::Shallow`

The 16 network tests from L04 are carried over, updated for the new return type of `train()`:
every `true` or `false` they expected is now an optional with or without a value. Every call that
runs out of epochs now also feeds each training set forward once more, for the final precision
measurement, so the feedforward tallies grow by one pass per set. Three tests are new:

| Test                            | Covers                                                  |
|---------------------------------|---------------------------------------------------------|
| `TrainReturnsOptionalPrecision` | `train()` returns `std::optional<double>`               |
| `TrainReturnsFinalPrecision`    | a run too short to evaluate still returns its precision |
| `TrainReturnsNegativePrecision` | a precision of `-1.0` is a value, not a failure         |

The stubs make each returned precision exact: an output layer that always predicts `0.75` scores
`0.5` against XOR, and one that always predicts `-1.5` misses by `2.0` on average and scores exactly
`-1.0`. That second case is why `train()` returns an optional rather than a negative number on
invalid arguments.

---

## Choosing the Convergence Threshold
The test trains towards a precision of `0.99`, i.e. a mean absolute error of `0.01`, over at most
10000 epochs with 8 hidden `Tanh` nodes, and asserts a mean absolute error below `0.05`. Measured
over 1000 runs before being written down:

* worst error of a run that reached the precision threshold: **0.0100**, since `train()` stops on
  the same measurement the test then makes
* best *untrained* error: **0.328**
* runs that failed to converge at all: **1 in 1000**, ending near `0.25` rather than near `0.01`
* epochs to the threshold: **1950 on average**, **6400 at worst**, comfortably inside the
  budget of 10000 epochs

That 1 run in 1000 is why the test trains up to three times rather than once. Each call to
`train()` resets both layers, so an attempt that ends stuck is followed by one starting from newly
drawn parameters; three independent attempts put the odds of a spurious failure at roughly one in
a billion. Retrying would be pointless without the reset: a second call that continued from
the stuck parameters would stay stuck.

`Tanh` rather than `Relu`, and 8 hidden nodes rather than 2, for a reason worth knowing: the same
network built from two hidden `Relu` nodes reaches that error in only about 15% of runs. A ReLU
node whose weighted sum is negative for every input never learns, and with start values in
`[-1.0, 1.0]`, about two networks in three contain such a node before training has even begun.
Appendix B section 8 invites you to try exactly that and watch it fail. Fine for an experiment you
re-run by hand, hopeless as a test.

Widening the hidden layer doesn't help either, which is less obvious: at 16 nodes the failure rate
climbs from 1 run in 1000 to about 1 in 100. The network sets its own learning rate now, and the
rule pushes it to the top of its range whenever progress is slow; a wider layer takes larger
combined steps at that rate and saturates its `Tanh` nodes.

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
