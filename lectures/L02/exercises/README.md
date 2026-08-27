# L02 Exercise Solutions
Solutions and notes for the exercises in [appendix B](../appendix/b_exercises.md). The code in this
directory is the **L01** model carried forward and extended three times: 
* A randomized training order.
* A precision measure that stops training early.
* A second model that picks its own learning rate.

---

## What L02 Adds
**A randomized training order.** The model no longer walks its training sets front to back. It
keeps a vector of indices, shuffles it before every epoch, and visits the sets in that order. A
model trained in a fixed order learns the order along with the data, which shows up the moment the
data arrives in a different one.

**A precision measure, and a reason to stop.** `precision()` returns `1.0 - MAE`, the mean absolute
error over every training set subtracted from one. `train()` computes it every tenth epoch and
stops as soon as it reaches the threshold it was given, so the epoch count becomes a budget rather
than a schedule.

**A model that tunes its own learning rate.** `Adaptive` is `Fixed` with the learning rate taken
out of the caller's hands. It starts at `0.1` and revises the rate at every precision evaluation,
raising it while progress is slow and lowering it when the model stops improving.

The learning rate was the hardest number to pick in **L01**: the tuning table in that lecture's
[solutions](../../L01/exercises/README.md#tuning-the-training) shows the epoch count for this
training data swinging between 3 and "never" across rates from 0.01 to 0.29. `Adaptive` is the
first model in the course that doesn't ask anyone to guess it.

---

## Files

| File | Contents |
|---|---|
| [include/ml/types.hpp](./include/ml/types.hpp) | The matrix aliases from **L01**, plus `MatrixU32` for the training order. The order is a vector of indices into the training data, so it holds `std::uint32_t` rather than `double`. |
| [include/ml/utils.hpp](./include/ml/utils.hpp) | Declares `ml::initRandGen()`, the one-time seeding of `std::rand()`. It sits in a shared header because both models call it. |
| [source/ml/utils.cpp](./source/ml/utils.cpp) | Defines `initRandGen()`: a static local `initialized` flag, and `std::srand(std::time(nullptr))` on the first call only. |
| [include/ml/lin_reg/interface.hpp](./include/ml/lin_reg/interface.hpp) | Unchanged from **L01**: a virtual destructor and the pure virtual `predict()`. Both models implement it, which is what lets `main.cpp` swap one for the other. |
| [include/ml/lin_reg/fixed.hpp](./include/ml/lin_reg/fixed.hpp) | `ml::lin_reg::Fixed`, now holding `myTrainOrder` instead of `mySetCount`, with the private `shuffle()` and `precision()` added and `train()` extended to `train(epochCount, learningRate = 0.01, precisionThreshold = 0.999999)`. |
| [source/ml/lin_reg/fixed.cpp](./source/ml/lin_reg/fixed.cpp) | The implementation: the constructor fills the training order and seeds the generator, `train()` shuffles per epoch and evaluates the precision every tenth, and `optimize()` is unchanged from **L01**. |
| [include/ml/lin_reg/adaptive.hpp](./include/ml/lin_reg/adaptive.hpp) | `ml::lin_reg::Adaptive`: the same members and private methods as `Fixed`, but `train(epochCount, precisionThreshold = 0.999999)` with no learning rate argument. |
| [source/ml/lin_reg/adaptive.cpp](./source/ml/lin_reg/adaptive.cpp) | The implementation, plus the file-local `updateLearningRate()` that holds the whole adaptation rule. |
| [source/main.cpp](./source/main.cpp) | Demonstration: trains an `Adaptive` model on the five training sets from `y = 3x + 2` for at most 100 epochs, then prints one prediction per training input. |
| [Makefile](./Makefile) | Builds the four source files into `linreg_demo` with `g++ -Wall -Werror -std=c++17`. Targets: `build`, `run`, `clean`, with `build run` as the default. |

The [test](./test/) directory holds this lecture's test suite and is documented in its
[own README](./test/README.md).

---

## Building and Running

```bash
make
```

This builds and runs the application, which prints an early-stop line followed by the predictions:

```
Target precision 1.00 reached after 11 epochs!
Input: 0, prediction: 2
Input: 1, prediction: 5
Input: 2, prediction: 8
Input: 3, prediction: 11
Input: 4, prediction: 14
```

No learning rate appears anywhere in `main.cpp`. The 100 epochs it asks for are a ceiling the model
never reaches: it stops at the first evaluation where the precision is at least `0.999999`, which
for this training data is the one after the eleventh epoch. The epoch count in the message can vary
with the shuffled order.

## Running the Tests

```bash
make -C test
```

All 38 test cases pass: 20 for `Fixed` and 18 for `Adaptive`.

---

## Notes on the Implementation

### The training order is a vector of indices
`myTrainOrder` holds `0, 1, 2 ... N-1` and gets shuffled, while the training data itself is never
touched. That matters for two reasons: the model holds the training data by `const` reference and
therefore *can't* reorder it, and shuffling indices is cheaper than moving values around, which
stays true as the data grows.

It also replaces `mySetCount` from **L01**. One index per complete training set means
`myTrainOrder.size()` already is the set count, so keeping a separate member for it would be two
sources of truth for one number.

### `initRandGen()` lives in a shared file
Seeding is a program-wide, once-only affair, which is what the static `initialized` flag inside the
function enforces. That guarantee is per function, though, not per program: leaving `initRandGen()`
in an anonymous namespace in `fixed.cpp` and copying it into `adaptive.cpp` would give each file its
own flag, and constructing one of each model would call `std::srand()` twice. The second call
re-seeds the generator, and within the same second it re-seeds it with the same value, restarting
the sequence that was already running.

One definition in [ml/utils.cpp](./source/ml/utils.cpp), declared in
[ml/utils.hpp](./include/ml/utils.hpp), gives both models the same flag. Later lectures add more
shared helpers to the same pair of files.

### Precision is `1.0 - MAE`, and can be negative
Precision here isn't a percentage in disguise. An untrained model predicting `0` for `y = 3x + 2`
has a mean absolute error of `8`, i.e. a precision of `-7`. That's why a threshold of `0.0` or less
is rejected: it accepts a model whose average prediction is off by a full unit or more. A threshold
of `1.0` or more is rejected for the opposite reason, since the mean absolute error can't go below
zero, so the threshold could never be reached and the check would never fire.

### The precision is evaluated every tenth epoch
Computing it scans every training set, i.e. it costs about as much as an epoch of training. Doing
it every epoch would nearly halve the training throughput to answer a question that changes slowly.
Checking every tenth epoch caps that overhead at ten percent, at the price of overshooting the
threshold by up to nine epochs' worth of training, which no caller can tell apart from a model that
stopped exactly on time.

### The adaptive rule reads the precision it already has
`Adaptive::train()` computes the precision once per evaluation and passes the value to
`updateLearningRate()`. The rule works on the difference between that value and the one from the
previous evaluation:

| Precision change since the last evaluation | Learning rate |
|---|---|
| Improved by at least `0.1` | left alone; the current rate is working |
| Improved by less than `0.1` | raised by `0.05`, capped at `0.25` |
| No improvement, or worse | lowered by `0.05`, floored at `1e-6` |

The two clamps are what keep the rule from destroying the model. The cap of `0.25` stays below the
rate at which this kind of training data starts to oscillate rather than converge, which **L01**'s
tuning table puts at around `0.29`. The floor keeps the rate positive, so a model that keeps
overshooting slows down instead of stopping dead.

The learning rate and the previous precision are locals in `train()`, not members. Every call
therefore starts from `0.1` again rather than inheriting whatever rate the previous call happened
to end on, which is what keeps repeated `train()` calls predictable.

### `Fixed` and `Adaptive` don't share their common parts
Everything except `train()` is duplicated between the two models: the constructor, `predict()`,
`optimize()`, `shuffle()`, `precision()`, and all five member variables. Factoring the shared parts
into a common base class is the obvious next move, and deliberately not made here: two small
independent models make the single difference between them easy to point at, which is the whole
subject of this part of the lecture.

---

## What the Adaptation Buys You
On the training data used throughout the course, `x = 0, 1, 2, 3, 4`, the adaptation makes no
visible difference: an initial rate of `0.1` reaches the default threshold at the first evaluation,
before the rule has ever run. The difference shows on data the initial rate is too cautious for.

The table below trains both models on `y = 2x + 2` sampled at `x = 0, 0.1, 0.2, 0.3, 0.4`, where
inputs a tenth of the usual size make every weight update a tenth as large. `Fixed` is given the
rate `Adaptive` starts at, so the gap between the columns is what the rule contributed. Each figure
is the median mean absolute error over 101 runs:

| Epochs | `Adaptive` | `Fixed` at 0.1 |
|:------:|:----------:|:--------------:|
| 11     | 0.235      | 0.249          |
| 51     | 0.046      | 0.085          |
| 101    | 0.0032     | 0.024          |
| 201    | 0.00002    | 0.0018         |
| 501    | converged  | converged      |

The models are level at the first evaluation, since neither has adapted anything yet. From there
`Adaptive` pulls away as the rule ratchets the rate up towards the cap, and by 201 epochs it's
roughly two orders of magnitude closer to the line. Give both enough epochs and they meet again at
the threshold: adaptation buys epochs, not accuracy the fixed rate could never reach.

This is also the property the test suite checks in
`LinRegAdaptiveComponent.LearningRateRisesWhileProgressIsSlow`, on the same training data. It's the
only way to observe a learning rate the model never reports.

---
