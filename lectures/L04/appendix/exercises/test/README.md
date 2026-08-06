# L04 Test Suite
Unit tests for the dense layer stub and component tests for the neural network built in
[appendix B](../../b_exercises.md), written with the
[`QAcademy Test`](https://github.com/qrtech-academy/test-framework) framework.

This suite **supersedes the L03 one**: it carries the stub tests over unchanged and adds
`ml/neural_network/shallow_test.cpp` on top. Once you've carried your code forward into this
lecture's exercises directory, this is the only suite you need to run.

---

## Unit Tests vs Component Tests
The stub tests are **unit** tests: one class, no collaborators. The `Shallow` tests are
**component** tests, because a network is meaningless on its own. It's built from two
`dense_layer::Interface` references, and what's worth testing is how it drives them.

That's why the stub earns its keep here. Against a real `Dense` layer you couldn't tell a network
that trains correctly from one that trains twice as long, because both converge. Against a stub
that computes nothing but counts what it's asked to do, the wiring is all that's left, and the
wiring is exactly what L04 is about.

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
│       ├── neural_network/
│       └── utils.cpp
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

Unlike L03, `SOURCE_FILES` is no longer empty: `Shallow` and `ml/utils` both have `.cpp` files, so
`../source/ml/neural_network/shallow.cpp` and `../source/ml/utils.cpp` are compiled in. `main.cpp`
stays out of it, because the suite brings its own `main()` in [testsuite.cpp](./testsuite.cpp).

---

## What's Covered

### The network's prediction path

| Test                                   | Covers                                    |
|----------------------------------------|-------------------------------------------|
| `PredictReturnsOutputLayerOutput`      | returns the output layer's output         |
| `PredictReadsOutputLayerLive`          | `setOutput()` changes the next prediction |
| `PredictReturnsReferenceToOutputLayer` | returns a reference, not a copy           |
| `PredictFeedsBothLayersOnce`           | one prediction, one pass per layer        |
| `PredictBeforeTraining`                | works on an untrained network             |
| `PredictAfterTraining`                 | works once training has run               |
| `UsableThroughInterface`               | works through the interface               |

The two layers are deliberately built with **different** `outputValue`s. If both reported `0.5`,
a network that returned the hidden layer's output would pass every one of these.

### The training loop

| Test                                   | Covers                               |
|----------------------------------------|--------------------------------------|
| `TrainFeedsEveryTrainingSetEveryEpoch` | `epochCount * setCount` feedforwards |
| `TrainRejectsZeroEpochCount`           | returns `false`, no layer touched    |
| `TrainRejectsInvalidLearningRate`      | rejects `0.0`, `1.0`, and beyond     |
| `TrainAcceptsValidLearningRate`        | accepts values inside `(0.0, 1.0)`   |
| `TrainDefaultLearningRate`             | `train(epochCount)` alone works      |

`TrainFeedsEveryTrainingSetEveryEpoch` is the one that matters. A training loop that runs a single
pass instead of every epoch, or that visits one training set instead of all four, still lines up
dimensionally and still returns `true`. The feedforward count is the only thing that separates it
from a correct loop.

Dimensions are chosen so mistakes can't hide: 2 inputs, 3 hidden nodes, 1 output node. Because the
input count and the hidden node count differ, a vector handed to the wrong layer is a dimension
error rather than a coincidence that happens to fit.

---

## Not Covered
The constructor's two `std::terminate()` paths aren't tested, for the same reason as in
L03: `std::terminate()` takes the whole process down, so testing it means running the binary in a
separate process and inspecting how it died. Verify by hand:

```cpp
// Feed the network with non-matching layers (the output layer's weight count doesn't match the
// hidden layer's node count). 
// Expect the program to terminate.
ml::dense_layer::Stub hidden{3U, 2U};
ml::dense_layer::Stub output{1U, 99U};
ml::neural_network::Shallow network{hidden, output, trainInput, trainOutput};

// Feed the network with two empty training sets. 
// Expect the program to terminate.
const ml::Matrix2d empty{};
ml::neural_network::Shallow network2{hidden, output, empty, empty};
```

---

## Adding New Test Files
Add a new test file to the build by adding its path to `TEST_FILES` in the [Makefile](./Makefile).
For example, if a file named `ml/neural_network/deep_test.cpp` is added to this directory, add the
corresponding line as shown below:

```makefile
# Test files - update this list as new test files are added to this directory.
TEST_FILES := ml/dense_layer/stub_test.cpp \
              ml/neural_network/deep_test.cpp \ # Added 'ml/neural_network/deep_test.cpp'.
              ml/neural_network/shallow_test.cpp \
              testsuite.cpp \
```

---
