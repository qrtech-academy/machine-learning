# L10 Test Suite
Unit tests for all three convolutional layers, plus component tests for the assembled network,
written with the [`QAcademy Test`](https://github.com/qrtech-academy/test-framework) framework.

This suite **supersedes the L09 one**: it carries `ml/conv_layer/conv_test.cpp` and
`ml/conv_layer/max_pool_test.cpp` over unchanged and adds `ml/flatten_layer/flatten_test.cpp` and
`ml/cnn/cnn_test.cpp`. It's the final suite of the course, and the first point at which every
layer in `cnn_work` is real - so it's also the first point at which the network as a whole can be
tested at all.

---

## Unit Tests and Component Tests
The layer tests are **unit tests**: each builds one layer and checks it in isolation.

`ml/cnn/cnn_test.cpp` holds **component tests**, which run the whole pipeline - conv, max pool,
flatten, dense - through the real factory. They can only pass once every layer is real, which is
why they arrive now rather than in L08. Where a unit test tells you a layer is wrong, a component
test tells you the layers don't fit together.

Two of them train an actual network, which makes them the only tests here that aren't
deterministic. See *Choosing the Thresholds* below for how their numbers were picked.

---

## Setting Up
There's nothing to copy or clone. You're still working in the `cnn_work` codebase you started in
L08, and the [Makefile](./Makefile) already points back at it:

```
lectures/
├── L08/
│   └── cnn_work/            <- the code under test, still where you left it
└── L10/
    └── appendix/
        └── exercises/
            └── test/        <- this directory
```

Override it with `make ML_DIR=<path>` to test a tree somewhere else.

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
code 0 when everything passes, and -1 otherwise. The whole run takes about a second: the two
training tests account for nearly all of it.

---

## What's Covered
All 73 test cases should pass: 44 carried over from L09, plus 16 for the flatten layer and 13 for
the network.

### `ml::conv_layer::Conv` and `ml::conv_layer::MaxPool`
Unchanged from L09 - see [that suite's README](../../../L09/appendix/exercises/test/README.md).

### `ml::flatten_layer::Flatten`

| Test                                       | Covers                             |
|--------------------------------------------|------------------------------------|
| `ConstructedDimensions`                    | n x n flattens to n * n            |
| `ConstructedMatrixShapes`                  | vector out, square gradients       |
| `LargerInputSquaresOutputSize`             | holds at other sizes               |
| `FeedforwardMatchesHandTrainedExample`     | appendix B's flattening, exactly   |
| `FeedforwardIsRowMajor`                    | (i, j) lands at i * size + j       |
| `FeedforwardHandlesSingleValue`            | 1x1 flattens to one value          |
| `FeedforwardPreservesValues`               | values pass through untouched      |
| `FeedforwardIsRepeatable`                  | nothing accumulates between calls  |
| `FeedforwardRejectsWrongSizedInput`        | input size must match              |
| `FeedforwardRejectsRaggedInput`            | rows must be full length           |
| `BackpropagateReshapesGradientsToMatrix`   | flat gradients fold to a matrix    |
| `BackpropagateIsRowMajor`                  | unfolding matches folding          |
| `BackpropagateUndoesFeedforward`           | flatten then unflatten is identity |
| `BackpropagateRejectsWrongSizedGradients`  | gradients hold the size squared    |
| `BackpropagateOverwritesPreviousGradients` | gradients replaced, not summed     |
| `UsableThroughInterface`                   | works through the interface        |

`BackpropagateUndoesFeedforward` is the one that matters. The flatten layer is the only place in
the network where the rank changes, 2D to 1D and back, and the only thing that really has to hold
is that the two directions agree. Row-major or column-major doesn't matter on its own; a
*mismatch* does, because then gradients land on the wrong pixels and the conv layer learns from
noise. This test flattens a matrix and feeds the result straight back in, so it fails if the two
directions ever disagree.

### `ml::cnn::Cnn`

| Test                                                  | Covers                             |
|-------------------------------------------------------|------------------------------------|
| `ConstructedDimensions`                               | reports image size and node count  |
| `LayersAreSizedToFitEachOther`                        | the pipeline builds at other sizes |
| `AddDenseLayerChangesOutputSize`                      | a new layer becomes the output     |
| `PredictReturnsOneValuePerOutputNode`                 | a prediction is output-sized       |
| `PredictIsRepeatable`                                 | predicting doesn't train           |
| `TrainRejectsInvalidLearningRate`                     | rate must be in `(0.0, 1.0)`       |
| `TrainRejectsZeroEpochs`                              | needs at least one epoch           |
| `TrainRejectsEmptyTrainingData`                       | needs at least one set             |
| `TrainChangesThePrediction`                           | training moves the network         |
| `LearnsToRecognizeAllFourDigits`                      | the whole pipeline converges       |
| `SingleDenseLayerNetworkTrainsToLowError`             | the demo's topology trains too     |
| `BackpropagationReachesTheConvLayerThroughEveryLayer` | gradients survive every layer      |
| `UsableThroughInterface`                              | works through `cnn::Interface`     |

`LearnsToRecognizeAllFourDigits` is the payoff test for the whole course: it trains on the digits
0 - 3 and checks that the node matching each image ends up the strongest. It uses a hidden dense
layer, for two reasons. It converges far more reliably than a single one (see below), and it
exercises the multi-layer loops in the network's feedforward, backpropagation, and optimization,
which a single dense layer never reaches at all.

`BackpropagationReachesTheConvLayerThroughEveryLayer` wires the four layers up by hand instead of
using `Cnn`, and that's deliberate. The network keeps its layers private, so there's no way to ask
a `Cnn` whether its conv layer actually learned anything - but built by hand, the conv layer's
`kernel()` can be read before and after. If the flatten layer reshapes gradients wrongly, or the
pooling layer routes them nowhere, the kernel never moves and this test says so. It's also
completely deterministic, unlike the two training tests.

`TrainRejectsInvalidLearningRate` includes `1.0`, which is rejected in a roundabout way worth
knowing about: it passes the network's own check (`1.0 < learningRate` is false) and is then
turned down by the conv layer, whose `ml::checkLearningRate()` compares strictly.

---

## Choosing the Thresholds
Both training tests assert a mean absolute error, and both numbers were measured rather than
guessed. The network with a hidden layer, 8 nodes wide, trained for 5000 epochs at a learning rate
of 0.01, measured over 2000 runs:

* all four digits classified correctly: **2000 / 2000**
* worst error over those runs: **0.134**
* threshold: **0.25**, a margin of nearly two

The same network *without* the hidden layer is noticeably less reliable: over 2000 runs it
classified all four digits in only 1994 of them. That's why
`SingleDenseLayerNetworkTrainsToLowError` checks the error rather than the answers - a test that
fails once every few hundred runs is worse than no test at all. Its threshold is **0.5** against a
worst measured error of **0.271**.

Both tests also assert the trained error is below the same network's error *before* training,
which a threshold alone wouldn't catch: an untrained network scores about **1.4**, but the check
guards against one that happened to start out lucky.

---

## Not Covered
The `std::terminate()` paths aren't tested, for the same reason as in every earlier suite: it
takes the whole process down. For the flatten layer, verify by hand:

```cpp
// Feed the layer an input size of zero.
// Expect the program to terminate.
ml::flatten_layer::Flatten flattenLayer{0U};
```

**A conv layer that never learns, when the fault is inside `Cnn` itself.** If the network fed
zeros back to the conv layer - by skipping the flatten layer on the way back, say - the dense
layers would still learn and these tests would mostly still pass. Measured over 800 runs, a
network whose conv layer is frozen still classifies all four digits about 82% of the time, and its
error distribution overlaps a healthy one's (worst healthy: 0.132; best frozen: 0.114), so no
single-run threshold separates them.

This matters less than it sounds. `source/ml/cnn/cnn.cpp` is given to you, not written by you -
it's byte-identical to the reference implementation's. The three layers you *do* write are covered
from both directions: their own unit tests check the arithmetic, and
`BackpropagationReachesTheConvLayerThroughEveryLayer` checks that gradients survive the trip back
through all of them. If you modify `cnn.cpp` yourself, run `cnn_work` and check the predictions by
eye as well.

**The network's own learning rate check.** `Cnn::train()` rejects a rate outside `(0.0, 1.0]`, but
removing that check changes nothing observable: the conv layer turns the same rates down a moment
later, so `train()` still returns `false`. `TrainRejectsInvalidLearningRate` pins the behaviour
down, not which of the two layers enforces it.

**Multi-channel inputs.** Every layer here handles one channel. The course's CNN never stacks
feature maps, so neither do the tests.

---

## A Note on `noexcept` in Test Helpers
The `expectMatrixNear()` and `expectVectorNear()` helpers are deliberately **not** marked
`noexcept`, unlike most helpers in these suites. The `EXPECT_*` macros report a failure by
throwing, and a throw out of a `noexcept` function calls `std::terminate()` - so marking them
would turn every ordinary test failure into a core dump that takes the remaining cases with it.
Any helper that asserts has to be able to throw.

---

## Adding New Test Files
Add a new test file to the build by adding its path to `TEST_FILES` in the [Makefile](./Makefile).
For example, if a file named `ml/flatten_layer/stub_test.cpp` is added to this directory, add the
corresponding line as shown below:

```makefile
# Test files - update this list as new test files are added to this directory.
TEST_FILES := ml/cnn/cnn_test.cpp \
              ml/conv_layer/conv_test.cpp \
              ml/conv_layer/max_pool_test.cpp \
              ml/flatten_layer/flatten_test.cpp \
              ml/flatten_layer/stub_test.cpp \ # Added 'ml/flatten_layer/stub_test.cpp'.
              testsuite.cpp \
```

---
