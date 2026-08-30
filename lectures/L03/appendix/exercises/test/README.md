# L03 Test Suite
Unit tests for the dense layer stub built in [appendix B](../../b_exercises.md), written with the
[`QAcademy Test`](https://github.com/qrtech-academy/test-framework) framework.

There's no linear regression here: **L03 starts a new `ml` codebase**, so this suite replaces
nothing and stands on its own. If you kept the regression code alongside it, keep its test suite
alongside this one too.

---

## Why Test a Stub?
The stub computes nothing, so at first glance there's nothing to test. Two reasons it's worth it
anyway:
1. **The range checks are the only real logic it has**, and the network you write in **L04** leans
   on them entirely. A stub whose `feedforward()` accepted an input of any size would let every
   L04 wiring test pass without proving anything. These tests are what make the L04 tests
   trustworthy.
2. **Most of them carry straight over to L05.** Your real `Dense` layer implements the same
   `dense_layer::Interface` with the same dimension contract, so the same checks apply to it
   unchanged.

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
│   └── main.cpp
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

Note that `SOURCE_FILES` in the [Makefile](./Makefile) is empty: the interface and the stub are
header-only, so there's nothing to compile besides the tests themselves. Add to that list once you
have `.cpp` files under `source/ml/`.

---

## What's Covered

| Test                                      | Covers                                       |
|-------------------------------------------|----------------------------------------------|
| `ConstructedDimensions`                   | reports the dimensions it was built with     |
| `DefaultOutputValue`                      | output defaults to `0.5`, one value per node |
| `ExplicitOutputValue`                     | an explicit output value is used instead     |
| `SetOutput`                               | `setOutput()` replaces every output value    |
| `ErrorIsZero`                             | the error is node-sized and all zero         |
| `WeightsAreZero`                          | the weights are the right shape and all zero |
| `FeedforwardChecksInputSize`              | input holds one value per **weight**         |
| `BackpropagateChecksReferenceSize`        | references hold one value per **node**       |
| `BackpropagateChecksNextLayerWeightCount` | the next layer holds one weight per node     |
| `OptimizeChecksInputSize`                 | input holds one value per weight             |
| `OptimizeChecksLearningRate`              | the learning rate lies inside `(0.0, 1.0)`   |
| `FeedforwardCount`                        | counts every call, accepted or not           |
| `ComputationLeavesStateUnchanged`         | a compute cycle and a reset change nothing   |
| `UsableThroughInterface`                  | works through the interface                  |

`initParams()` is checked inside those last two rather than in a test of its own, since there's
nothing for it to do here: the stub has no trainable parameters, so its override is empty, and both
tests assert that calling it leaves the layer exactly as it was. What matters at this point is that
it's *callable*, directly and through an `Interface&`. The network in **L04** calls it on both of
its layers before every training run, and `Dense` gives it a real body in **L05**.

The three size checks are deliberately fussy about *which* count they compare against, because
mixing up the node count and the weight count is the easiest mistake to make here, and the one
that would quietly break the network in L04. `FeedforwardChecksInputSize` therefore also rejects an
input sized to the node count, not just inputs that are too short or too long.

---

## Reusing These Tests in L05
When you replace the stub with a real `Dense` layer, change one line at the top of
[stub_test.cpp](./ml/dense_layer/stub_test.cpp):

```cpp
using DenseLayer = ml::dense_layer::Stub; // -> ml::dense_layer::Dense
```

These carry over as they are, because they only test the interface contract:
* `ConstructedDimensions`
* `FeedforwardChecksInputSize`
* `BackpropagateChecksReferenceSize`
* `BackpropagateChecksNextLayerWeightCount`
* `OptimizeChecksInputSize`
* `OptimizeChecksLearningRate`

These don't, because they describe a layer that deliberately computes nothing:
* `DefaultOutputValue` / `ExplicitOutputValue` / `SetOutput`: a real layer computes its output from
  the input and takes an activation function rather than an output value, so it has no
  `outputValue` to set. `setOutput()` isn't part of `dense_layer::Interface` and exists on the stub
  alone.
* `FeedforwardCount`: counting calls is a testing aid, not layer behaviour. Like `setOutput()`,
  `feedforwardCount()` and `clearFeedforwardCount()` live on the stub only.
* `ErrorIsZero` / `WeightsAreZero`: a real layer starts with randomized weights and computes a
  real error during backpropagation.
* `ComputationLeavesStateUnchanged`: the exact opposite is true of a real layer, for the reset as
  much as for the compute cycle. `Dense::initParams()` draws new values every time it's called.

---

## Not Covered
The specification requires the constructor to call `std::terminate()` when the node count or the
weight count is zero. That path isn't tested here: `std::terminate()` takes the whole process down
with it, so testing it means running the test binary in a separate process and inspecting how it
died, which the framework doesn't support. Verify it by hand instead:

```cpp
// Try to create a dense layer with no nodes.
// Expect the program to terminate.
ml::dense_layer::Stub layer{0U, 3U};
```

---

## Adding New Test Files
Add a new test file to the build by adding its path to `TEST_FILES` in the [Makefile](./Makefile).
For example, if a file named `ml/dense_layer/dense_test.cpp` is added to this directory, add the
corresponding line as shown below:

```makefile
# Test files - update this list as new test files are added to this directory.
TEST_FILES := ml/dense_layer/dense_test.cpp \ # Added 'ml/dense_layer/dense_test.cpp' to the build.
              ml/dense_layer/stub_test.cpp \
              testsuite.cpp \
```

---
