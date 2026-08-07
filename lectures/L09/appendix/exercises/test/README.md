# L09 Test Suite
Unit tests for the convolutional and max pooling layers built in
[L08's appendix A](../../../L08/appendix/a_conv_layer.md) and
[appendix A](../../a_max_pool_layer.md), written with the
[`QAcademy Test`](https://github.com/qrtech-academy/test-framework) framework.

This suite **supersedes the L08 one**: it carries `ml/conv_layer/conv_test.cpp` over unchanged and
adds `ml/conv_layer/max_pool_test.cpp`. The conv layer doesn't retire when the pooling layer
arrives - it's still in the pipeline, so it's still tested. This is the only suite you need to run.

---

## Two Layers, Two Kinds of Test
`Conv` randomizes its bias and kernel, so its tests have to read those back through `kernel()` and
`bias()` and recompute the expected values at runtime. `MaxPool` has **no parameters at all**: it
randomizes nothing, learns nothing, and its output is a pure function of its input. So its expected
values are simply written out in full.

The main worked case is the hand-training example from
[L06's appendix B](../../../L06/appendix/b_exercises.md), reused here exactly as you did it on
paper. The conv layer's output goes in:

```
1.3 1.9 1.9 1.9
1.7 1.7 1.1 1.9
1.7 1.3 0.5 1.7
1.7 2.1 1.9 2.3
```

and the pooled output comes out as `1.9 1.9 / 2.1 2.3`, with the gradients `10 20 / 30 40` routing
back to exactly the four positions appendix B sends them to. If those numbers look familiar, they
should;  it's the same arithmetic, now run by your code instead of by hand.

---

## Setting Up
There's nothing to copy or clone. Unlike L01 - L05, the code under test isn't in this `exercises`
directory: you're still working in the `cnn_work` codebase you started in L08, and the
[Makefile](./Makefile) already points back at it:

```
lectures/
├── L08/
│   └── cnn_work/            <- the code under test, still where you left it
│       ├── include/ml/
│       └── source/ml/
└── L09/
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
code 0 when everything passes, and -1 otherwise.

Until you've written the class, the build fails with `'MaxPool' in namespace 'ml::conv_layer' does
not name a type`. That's expected: the suite is the specification, so it doesn't compile until the
thing it specifies exists.

---

## What's Covered
All 44 test cases should pass: the 22 carried over from L08 plus 22 new ones.

### `ml::conv_layer::Conv`
Unchanged from L08 - see [that suite's README](../../../L08/appendix/exercises/test/README.md) for
the breakdown.

### `ml::conv_layer::MaxPool`

| Test                                           | Covers                               |
|------------------------------------------------|--------------------------------------|
| `ConstructedDimensions`                        | downsamples by the pool size         |
| `ConstructedMatrixShapes`                      | pooled out, input-sized gradients    |
| `PoolSizeOneKeepsInputSize`                    | a 1x1 pool downsamples nothing       |
| `PoolSizeEqualToInputYieldsSingleValue`        | one block over the whole image       |
| `FeedforwardMatchesHandTrainedExample`         | appendix B's pooling, exactly        |
| `FeedforwardSelectsLargestInEachPool`          | each output from its own block       |
| `FeedforwardWithPoolSizeOneIsIdentity`         | a 1x1 pool passes values through     |
| `FeedforwardWithFullPoolYieldsGlobalMax`       | one block yields the global max      |
| `FeedforwardHandlesNegativeValues`             | negatives pool to the least negative |
| `FeedforwardIsRepeatable`                      | nothing accumulates between calls    |
| `FeedforwardRejectsWrongSizedInput`            | input size must match                |
| `FeedforwardRejectsRaggedInput`                | rows must be full length             |
| `BackpropagateMatchesHandTrainedExample`       | appendix B's routing, exactly        |
| `BackpropagateRoutesToFirstMaxOnTies`          | ties go to the first occurrence      |
| `BackpropagateWithPoolSizeOneRoutesEverything` | a 1x1 pool routes everything         |
| `BackpropagateResetsGradientsBetweenCalls`     | gradients cleared, not accumulated   |
| `BackpropagateUsesLatestInput`                 | routing follows the latest input     |
| `BackpropagateRejectsWrongSizedGradients`      | gradients are pooled-sized           |
| `BackpropagateRejectsRaggedGradients`          | gradients must be square             |
| `OptimizeAcceptsAnyLearningRate`               | no rate is invalid here              |
| `OptimizeChangesNothing`                       | optimization is a no-op              |
| `UsableThroughInterface`                       | works through the interface          |

Three are worth reading.

`FeedforwardHandlesNegativeValues` catches the most common way to get a max wrong: starting the
search at `0.0` instead of at the block's first value. Every test with non-negative input passes
either way, so without this one the bug ships. It matters in the real pipeline too - once the conv
layer uses `Tanh`, its output is routinely negative.

`BackpropagateRoutesToFirstMaxOnTies` pins down the rule appendix A section 5 spells out. Every
block in its input contains a tie, so a layer that routed to the *last* occurrence fails on all
four rather than getting away with three.

`OptimizeAcceptsAnyLearningRate` documents a real difference from `Conv`: the conv layer rejects a
rate outside `(0.0, 1.0)`, but the pooling layer accepts anything, including `0.0` and negatives,
because it has nothing to update and so nothing to corrupt.

---

## Not Covered
The constructor's `std::terminate()` paths aren't tested, for the same reason as in the earlier
suites: `std::terminate()` takes the whole process down, so testing it means running the binary in
a separate process and inspecting how it died. Verify by hand:

```cpp
// Feed the layer a pool size of zero.
// Expect the program to terminate.
ml::conv_layer::MaxPool maxPool{4U, 0U};

// Feed the layer an input size that isn't divisible by the pool size.
// Expect the program to terminate.
ml::conv_layer::MaxPool maxPool2{5U, 2U};
```

Backpropagating without a preceding `feedforward()` isn't tested either. The layer starts with an
all-zero `myInput`, so every block is one big tie and every gradient routes to its block's top-left
corner. That's defined behaviour rather than a bug, but it isn't behaviour worth depending on.

As in L08, nothing here checks that `factory.cpp` was switched over from `MaxPoolStub` to
`MaxPool` - these tests construct the class directly. Run `cnn_work` itself to confirm that.

---

## A Note on `noexcept` in Test Helpers
The helper `expectMatrixNear()` in
[max_pool_test.cpp](./ml/conv_layer/max_pool_test.cpp) is deliberately **not** marked `noexcept`,
unlike most helpers in these suites. The `EXPECT_*` macros report a failure by throwing, and a
throw out of a `noexcept` function calls `std::terminate()` - so marking it would turn every
ordinary test failure into a core dump that takes the remaining cases with it. Any helper that
asserts has to be able to throw.

---

## Adding New Test Files
Add a new test file to the build by adding its path to `TEST_FILES` in the [Makefile](./Makefile).
For example, if a file named `ml/conv_layer/max_pool_stub_test.cpp` is added to this directory, add
the corresponding line as shown below:

```makefile
# Test files - update this list as new test files are added to this directory.
TEST_FILES := ml/conv_layer/conv_test.cpp \
              ml/conv_layer/max_pool_stub_test.cpp \ # Added 'ml/conv_layer/max_pool_stub_test.cpp'.
              ml/conv_layer/max_pool_test.cpp \
              testsuite.cpp \
```

---
