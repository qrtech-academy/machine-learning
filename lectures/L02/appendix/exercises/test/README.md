# L02 Test Suite

Unit and component tests for the linear regression model built in
[appendix B](../../b_exercises.md), written with the
[`QAcademy Test`](https://github.com/qrtech-academy/test-framework) framework.

This suite is cumulative: it covers everything from **L01** as well as the randomized training
order and precision calculation added in **L02**. Once you've carried your code forward into this
lecture's exercises directory, this is the only suite you need to run.

The tests use nothing but the public interface of the model: the constructor, `predict()` and
`train()`. They never reach into the private members, so any implementation that follows the
exercise specification passes, no matter how it's written on the inside.

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
│   ├── ml/
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

To only build the tests, without running them:

```bash
make build
```

To only run the tests, without rebuilding first:

```bash
make run
```

Remove the build artifacts with:

```bash
make clean
```

Every test that fails prints the assertion that failed, the two values involved, and the file and
line it came from. The test suite exits with code 0 when everything passes, and -1 otherwise.

Note that `train()` prints its progress, so the test output is interleaved with precision reports.
Look for the `succeeded`/`failed` lines and the summary on the last line.

## What's Covered

| Suite | Covers |
|---|---|
| `LinRegFixed` | Construction, argument validation of `train()` (including the new precision threshold), and the parameter updates made by `optimize()`, with every expected value computed by hand. |
| `LinRegComponent` | Full training runs used through `ml::lin_reg::Interface`: convergence, generalization, the randomized training order, and early stopping on the precision threshold. |

Three details worth knowing about while reading the tests:

* **The training order is random from this lecture on.** Tests that check exact parameter values
  therefore train on a *single* training set, since an order of one can't be shuffled and the
  result stays reproducible. Tests that train on several sets only check properties that hold whatever
  the order turns out to be.
* **`shuffle()` and `precision()` are private**, so the tests can't call them. Both are checked
  through `train()` instead: shuffling shows up as ten identically trained models *not* all
  agreeing, and the precision calculation shows up as training stopping once the mean absolute
  error is within `1.0 - precisionThreshold`.
* **`ml::lin_reg::Fixed` stores references to the training data.** The training vectors must
  outlive the model, so every test declares them as local variables before constructing the model
  and never passes a temporary. Passing a temporary compiles cleanly and then reads freed memory.

## Not Covered

The specification requires the constructor to call `std::terminate()` when it's given no complete
training sets. That path isn't tested here: `std::terminate()` takes the whole process down with
it, so testing it means running the test binary in a separate process and inspecting how it died,
which the framework doesn't support. Verify it by hand instead:

```cpp
const ml::Matrix1d empty{};
ml::lin_reg::Fixed model{empty, empty}; // Must print an error and terminate.
```

The *exact* epoch count at which training stops is also left alone. It depends on the shuffled
order, so asserting it would make the suite fail at random.

## Adding New Test Files

Add a new test file to the build by adding its path to `TEST_FILES` in the [Makefile](./Makefile).
For example, if a file named `ml/lin_reg/adaptive_test.cpp` is added to this directory, add the
corresponding line as shown below:

```makefile
# Test files - update this list as new test files are added to this directory.
TEST_FILES := ml/lin_reg/adaptive_test.cpp \ # Added 'ml/lin_reg/adaptive_test.cpp' to the build.
              ml/lin_reg/fixed_test.cpp \
              testsuite.cpp \
```

Do the same with `SOURCE_FILES` for every new source file added to the implementation's
`source/` directory. Note that `main.cpp` is deliberately left out: the test suite
brings its own `main()` in [testsuite.cpp](./testsuite.cpp).

---
