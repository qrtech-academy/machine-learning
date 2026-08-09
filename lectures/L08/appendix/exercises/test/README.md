# L08 Test Suite
Unit tests for the convolutional layer built in [appendix A](../../a_conv_layer.md), written with
the [`QAcademy Test`](https://github.com/qrtech-academy/test-framework) framework.

This is the **first suite of part II**, and it starts fresh rather than carrying anything over from
L01-L05: those suites test a different codebase (the exercises directory you built up over part
I), while from L08 onwards you work in `cnn_work`. It covers `ml::conv_layer::Conv` only. The max
pooling and flatten layers are still stubs at this point; their tests arrive in L09 and L10.

---

## Testing a Layer That Randomizes Itself
`Conv` randomizes its own bias and every kernel weight in the constructor, so nothing it computes
can be predicted from the constructor arguments alone. Fed the same input twice, two different
layers produce two different outputs, and neither is wrong.

That's why appendix A asks you to expose two accessors that aren't on `conv_layer::Interface`:

```cpp
const Matrix2d& kernel() const noexcept;
double bias() const noexcept;
```

The tests read those, then redo the convolution the long way round, building the padded matrix, walking
the window, multiplying and summing, then compare against what the layer produced. So the expected values
are exact rather than merely plausible, and they hold no matter what the constructor happened to
randomize. `padded()`, `windowSum()` and `expectedOutput()` at the top of
[conv_test.cpp](./ml/conv_layer/conv_test.cpp) do this.

They stay off the interface deliberately: `MaxPool` implements the same interface in L09 and has no
kernel or bias to report.

Most tests use `act_func::Type::None` so the activation is the identity and the arithmetic stays
exact. `AppliesReluActivation` is the exception, and it's the one that checks the activation
function is applied at all.

---

## Setting Up
There's nothing to copy or clone. Unlike L01-L05, the code under test isn't in this `exercises`
directory: from L08 onwards you work in `cnn_work`, and the [Makefile](./Makefile) already points
at it:

```
lectures/L08/
├── appendix/
│   └── exercises/
│       └── test/        <- this directory
└── cnn_work/            <- the code under test
    ├── include/ml/
    └── source/ml/
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

Until you've written the class, the build fails with `'Conv' in namespace 'ml::conv_layer' does not
name a type`. That's expected: the suite is the specification, so it doesn't compile until the
thing it specifies exists.

---

## What's Covered
All 22 test cases should pass.

### `ml::conv_layer::Conv`

| Test                                                 | Covers                               |
|------------------------------------------------------|--------------------------------------|
| `ConstructedDimensions`                              | same size in and out                 |
| `ConstructedMatrixShapes`                            | output and gradients are input-sized |
| `ConstructedKernelShape`                             | kernel is square, as asked           |
| `ConstructedParametersAreRandomized`                 | bias and weights in `[0.0, 1.0]`     |
| `LargerKernelKeepsOutputSize`                        | a 3x3 kernel preserves size          |
| `FeedforwardMatchesHandComputedConvolution`          | bias plus each window's dot product  |
| `ZeroInputYieldsBiasEverywhere`                      | zero input gives the bias            |
| `UnitKernelIsAffinePerPixel`                         | a 1x1 kernel adds no padding         |
| `PaddingOnlyAffectsTheBorder`                        | padding sits where it should         |
| `AppliesReluActivation`                              | ReLU applied to the sum              |
| `FeedforwardRejectsWrongSizedInput`                  | input size must match                |
| `FeedforwardRejectsRaggedInput`                      | rows must be full length             |
| `BackpropagateRejectsWrongSizedGradients`            | gradient size must match             |
| `BackpropagateRejectsRaggedGradients`                | gradients must be square             |
| `ZeroGradientsYieldZeroInputGradients`               | no error in, none out                |
| `InputGradientsMatchHandComputedScatter`             | kernel scattered back, unpadded      |
| `OptimizeRejectsInvalidLearningRate`                 | rate must be in `(0.0, 1.0)`         |
| `OptimizeAcceptsValidLearningRate`                   | a valid rate is accepted             |
| `OptimizeMovesBiasByGradientTimesLearningRate`       | bias delta = gradient x rate         |
| `OptimizeMovesKernelByGradientTimesLearningRate`     | weight delta = gradient x rate       |
| `OptimizeWithZeroGradientsLeavesParametersUnchanged` | zero gradients change nothing        |
| `UsableThroughInterface`                             | works through the interface          |

Two are worth reading. `PaddingOnlyAffectsTheBorder` feeds a constant image: every window lying
wholly inside the original image then sees the same values, so the interior of the output must be
uniform, while the border positions see padded zeros and differ. Get the offset wrong and that
uniform block shifts off-centre. It's the cheapest check that `pad = kernelSize / 2` landed right.

`OptimizeMovesKernelByGradientTimesLearningRate` is what pins down `backpropagate()`. There's no
`kernelGradients()` accessor, so the gradients are never read directly; instead the test records
the kernel, runs a full feedforward/backpropagate/optimize cycle, and checks each weight moved by
exactly its hand-computed gradient times the learning rate. A wrong gradient shows up as a wrong
weight.

---

## Not Covered
The constructor's `std::terminate()` paths aren't tested, for the same reason as in L01 and L03 -
L05: `std::terminate()` takes the whole process down, so testing it means running the binary in a
separate process and inspecting how it died. Verify by hand:

```cpp
// Feed the layer a kernel size outside [1, 11].
// Expect the program to terminate.
ml::conv_layer::Conv convLayer{4U, 0U};

// Feed the layer a kernel larger than the input.
// Expect the program to terminate.
ml::conv_layer::Conv convLayer2{4U, 5U};
```

`Tanh` isn't exercised here either. Unlike the dense layer, where the derivative has to be taken at
the pre-activation sum and only `Tanh` can catch the mistake, the conv layer's tests read that
derivative back through `optimize()`, where `None` already pins it down exactly.

Nothing here checks that `factory.cpp` was actually switched over from `ConvStub` to `Conv`. These
tests construct `Conv` directly, so they pass whether or not you finished the wiring step in
appendix A. Run `cnn_work` itself to confirm that part.

---

## Adding New Test Files
Add a new test file to the build by adding its path to `TEST_FILES` in the [Makefile](./Makefile).
For example, if a file named `ml/conv_layer/conv_stub_test.cpp` is added to this directory, add the
corresponding line as shown below:

```makefile
# Test files - update this list as new test files are added to this directory.
TEST_FILES := ml/conv_layer/conv_stub_test.cpp \ # Added 'ml/conv_layer/conv_stub_test.cpp'.
              ml/conv_layer/conv_test.cpp \
              testsuite.cpp \
```

---
