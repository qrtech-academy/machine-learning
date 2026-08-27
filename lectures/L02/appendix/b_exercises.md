# Appendix B - Exercises
You'll extend the `ml::lin_reg::Fixed` class from **L01**, first with a randomized training order, then with precision calculation. Finally you'll add a second model, `ml::lin_reg::Adaptive`, which picks its own learning rate while it trains.

---

## Randomizing the Training Order

### 1. MatrixU32 alias
Add the alias `MatrixU32` to the header file `ml/types.hpp` as a substitute for `std::vector<std::uint32_t>`, in the namespace `ml`.

Include `<cstdint>` in `ml/types.hpp`.

---

### 2. Random number generator
Add a function named `initRandGen()` in an anonymous namespace in `source/ml/lin_reg/fixed.cpp`.
The function should initialize the random number generator exactly once when called:
* Use a static local variable named `initialized` (of type `bool`, initialized to `false`) to track whether the generator has already been initialized.
* Initialize the generator with `std::srand(std::time(nullptr))`.
* Set `initialized` to `true` after initialization.
* Include `<cstdlib>` and `<ctime>` in `fixed.cpp`.
* Should be marked `noexcept`.

---

### 3. Training order
Add a private member variable named `myTrainOrder` to `Fixed`:
* Should hold the indices of the training sets as unsigned integers (`ml::MatrixU32`).
* Mustn't be marked `const`, since its contents are rearranged before every epoch.

The member variable `mySetCount` from **L01** can now be removed: `myTrainOrder` holds one index per training set, so `myTrainOrder.size()` is the number of training sets.

Update the constructor in `source/ml/lin_reg/fixed.cpp`:
* Compute the number of complete training sets as in **L01**, i.e. `std::min(trainIn.size(), trainOut.size())`, but keep it in a local variable rather than a member.
* Print an error message and call `std::terminate()` if the set count is 0, as in **L01**.
* Call `initRandGen()`, so the random number generator is ready before the first shuffle. Calling it here rather than in `train()` means it's initialized exactly once per model, no matter how often training is started.
* Resize `myTrainOrder` to the set count, then fill it with the indices `0, 1, 2 ... N-1`.

---

### 4. Shuffling
Add a private method named `shuffle()` to `Fixed`:
* Should shuffle the contents of `myTrainOrder` in random order.
* For each index `i`, pick a random index `r` and swap `myTrainOrder[i]` and `myTrainOrder[r]`.
* Should be marked `noexcept`.

---

### 5. Updating the training method
Update the method `train()` in `source/ml/lin_reg/fixed.cpp`:
* Call `shuffle()` at the start of every epoch.
* Iterate through the training sets in the order given by `myTrainOrder` instead of sequentially by index, e.g. with a range-based for loop over `myTrainOrder`. Note that the loop variable is now the index into the training data, not the counter itself.

---

### 6. Compiling and testing
Compile and test-run the program. The training order should now be random every epoch, but the predicted result should still match the training data once training is complete.

---

## Precision Calculation

### 1. Precision calculation
Add a private method named `precision()` to the `Fixed` class. The method should compute the model's precision given the training data:
* Compute the mean absolute error (MAE) across all training sets:
    * For each training set, compute the absolute error: `abs(output - predict(input))`.
    * Sum all absolute errors and divide by the number of training sets.
* Return `1.0 - MAE` as a floating-point number.
* Should be marked `[[nodiscard]]`, `const`, and `noexcept`.

Include `<cmath>` in `fixed.cpp` for `std::abs`.

---

### 2. Updating the training method
Update the method `train()` in `source/ml/lin_reg/fixed.cpp` so training stops early once the precision exceeds a given threshold:
* Add an argument named `precisionThreshold` (a floating-point number specifying the precision threshold) as the **last** parameter, so the full signature reads `train(epochCount, learningRate = 0.01, precisionThreshold = 0.999999)`. Default value: `0.999999` (99.9999 %). Note that the argument shouldn't be named `precision`: inside `train()`, a parameter of that name hides the method `precision()`, so calling it no longer compiles (unless `this->precision()` is used).
* Return `false` if `precisionThreshold` falls outside the range `(0.0, 1.0)`:
    * A threshold of `1.0` or more can never be reached, since the precision is `1.0 - MAE` and the mean absolute error can't be negative. Training would always run the full epoch count.
    * A threshold of `0.0` or less accepts a model whose mean absolute error is `1.0` or worse, which defeats the purpose of the check.
* As in **L01**, `train()` reports invalid arguments through its return value. Only the constructor calls `std::terminate()`, because it has no way to return a failure code to the caller.
* Call `precision()` every tenth epoch, starting with the first; computing it scans every training set, so checking less often than every epoch keeps that overhead down over thousands of epochs. Use a named constant for the interval rather than a bare `10`.
* Stop training and return `true` as soon as the precision reaches or exceeds `precisionThreshold`, printing the achieved precision and the number of epochs it took.
* Return `true` once the epochs run out, if the threshold was never reached.

---

### 3. Compiling and testing
Compile and test-run the program. The model should now stop training as soon as the precision reaches the threshold, long before the 1000 epochs `main.cpp` asks for:

```
Target precision 1.00 reached after 11 epochs!
Input: 0, prediction: 2
Input: 1, prediction: 5
Input: 2, prediction: 8
Input: 3, prediction: 11
Input: 4, prediction: 14
```

The training order is random, so the epoch count can vary between runs. For this training data it settles at 11: the precision is only checked every tenth epoch, so the check that first passes is the one after the eleventh epoch has run. Report the number of epochs completed, not the loop index, so the first epoch reads as 1 rather than 0.

---


## An Adaptive Learning Rate
The learning rate is the one training parameter with no sensible default. Too low and the model
crawls, too high and it never settles, and where the line between the two falls depends on the
training data rather than on the algorithm. Instead of asking the caller to guess, you'll now write
a second model that starts at a cautious rate and revises it while training, using the precision it
already computes every tenth epoch.

The new class is called `Adaptive`, and it's the one the `Fixed` name has been pointing at since
**L01**.

---

### 1. Shared utilities
`initRandGen()` currently sits in an anonymous namespace in `fixed.cpp`, which makes it private to
that file. The new model needs the same one-time seeding, and copying the function across would
give each file its own `initialized` flag: the second copy would call `std::srand()` again when the
second model is constructed, and a re-seed within the same second restarts the very sequence that
was already running.

Move the function into a pair of shared files instead:
* Create `include/ml/utils.hpp`. Declare `initRandGen()` there, in the namespace `ml`, taking no arguments, returning nothing, and marked `noexcept`.
* Create `source/ml/utils.cpp`. Move the definition there unchanged, along with the `<cstdlib>` and `<ctime>` includes it needs.
* Remove the function and the `<ctime>` include from `fixed.cpp`, and include `"ml/utils.hpp"` instead. Keep `<cstdlib>`: `shuffle()` still calls `std::rand()`. The call in the constructor stays exactly as it was: `Fixed` lives in `ml::lin_reg`, so the unqualified name still finds `ml::initRandGen()` in the enclosing namespace.
* Add `source/ml/utils.cpp` to `SRC_FILES` in the makefile.

Rebuild and re-run before going further. Nothing about the program's behaviour should have changed;
this step only moves code.

---

### 2. The Adaptive class
Add the header `include/ml/lin_reg/adaptive.hpp` and the source file `source/ml/lin_reg/adaptive.cpp`, and
add the source file to `SRC_FILES` in the makefile.

In the namespace `ml::lin_reg`, implement a class named `Adaptive` that inherits `Interface` via
public inheritance and is marked `final`. Everything except `train()` is the same as in `Fixed`:
* The constructor takes `trainIn` and `trainOut` and is marked `explicit` and `noexcept`. It checks the set count, terminates when it's 0, fills `myTrainOrder` with `0, 1, 2 ... N-1`, and calls `initRandGen()`.
* `~Adaptive()` is `default`, `noexcept`, and `override`.
* `predict()` returns `y = kx + m`, and is marked `[[nodiscard]]`, `const`, `noexcept`, and `override`.
* The private methods `optimize()`, `shuffle()`, and `precision()` are the same as in `Fixed`.
* The private member variables are the same: `myTrainOrder`, `myTrainIn`, `myTrainOut`, `myBias`, and `myWeight`.
* The default constructor, copy and move constructors, and the corresponding operators are deleted.

`train()` is where the two models part ways:

```cpp
bool train(std::size_t epochCount, double precisionThreshold = 0.999999) noexcept;
```

There's no `learningRate` argument, because the model no longer takes one. Copying the shared parts
across rather than sharing them with `Fixed` is deliberate here: two small independent models make
the one difference between them easy to see.

---

### 3. The learning rate rule
Add a function named `updateLearningRate()` in an anonymous namespace in `adaptive.cpp`:

```cpp
void updateLearningRate(double& learningRate, double& prevPrecision,
                        double currentPrecision) noexcept;
```

It takes the current learning rate and the previous precision by reference, since it updates both,
and the precision just measured by value. Give each of the four values it needs a named constant:
a maximum learning rate of `0.25`, a minimum of `1e-6`, a smallest expected improvement of `0.1`,
and a step of `0.05`.

The rule itself works on the difference between the two precisions, i.e. how much the model
improved since the last evaluation:
* **Improved by at least the expected minimum:** leave the learning rate alone. It's working.
* **Improved, but by less than that:** raise the learning rate by the step, and clamp it to the maximum. Progress has stalled, so take bigger steps.
* **Didn't improve at all:** lower the learning rate by the step, and clamp it to the minimum. The model is overshooting, so take smaller steps.

Store the precision just measured in `prevPrecision` before returning, so the next call has
something to compare against.

The two clamps are what keep the rule safe. The maximum stays below the rate at which this kind of
training data starts to oscillate instead of converging, and the minimum keeps the rate positive,
so a model that keeps missing can still creep towards the line rather than stopping dead.

---

### 4. Training
Implement `train()` in `adaptive.cpp`:
* Give the evaluation interval (`10`) and the initial learning rate (`0.1`) named constants.
* Declare the learning rate and the previous precision as local variables, initialized to the initial learning rate and `0.0`. Being local means every call starts from the initial rate again, rather than inheriting whatever the previous call ended on.
* Return `false` if `epochCount` is 0, or if `precisionThreshold` falls outside the range `(0.0, 1.0)`, exactly as `Fixed` does. There's no learning rate left to validate.
* For each epoch: call `shuffle()`, then run `optimize()` over all training sets in the order given by `myTrainOrder`, passing the current learning rate.
* Every tenth epoch, except the first, compute the precision once and keep it in a local variable:
    * Stop training and return `true` if it reaches or exceeds `precisionThreshold`, printing the achieved precision and the number of epochs it took, as in the precision section.
    * Otherwise, pass it to `updateLearningRate()`.
* Return `true` once the epochs run out.

Note that the precision is computed once per evaluation and then used twice. Calling `precision()`
a second time inside `updateLearningRate()` would scan every training set again for a value that's
already in hand.

---

### 5. Compiling and running
Update `main.cpp` to train an `ml::lin_reg::Adaptive` model rather than a `Fixed` one, and drop the
learning rate from the `train()` call. An epoch budget of 100 is plenty now, since the model stops
as soon as it reaches the default precision threshold.

Compile and run. The predictions should match the training data, without a learning rate having
been chosen anywhere in the program:

```
Input: 0, prediction: 2
Input: 1, prediction: 5
Input: 2, prediction: 8
Input: 3, prediction: 11
Input: 4, prediction: 14
```

Above them sits the early-stop line your own `train()` prints. The epoch count in it varies with
the shuffled order.

---

## Running the Tests
An updated test suite is available in [exercises/test](../exercises/test/). It's cumulative: it
covers everything from **L01** as well as the randomized training order, the precision threshold,
and the adaptive learning rate added here, so it supersedes the **L01** suite entirely.

Once you've carried your code forward into this lecture's exercises directory, build and run it as
before:

```bash
make -C test
```

All 38 test cases should pass: 20 for `Fixed` and 18 for `Adaptive`. Note that `train()` prints its
progress, so the output is interleaved with early-stop reports. Look for the summary on the last
line.

See the [test suite's README](../exercises/test/README.md) for more information, including why the
tests that check exact values train on a single training set now that the order is randomized, and
how a learning rate the model keeps to itself can be tested at all.

---
