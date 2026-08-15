# Appendix B - Exercises
You'll extend the `ml::lin_reg::Fixed` class from **L01**, first with a randomized training order, then with precision calculation.

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
* Print the achieved precision after the last epoch if the threshold was never reached, then return `true`.

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

### 4. Running the tests
An updated test suite is available in [exercises/test](./exercises/test/). It's cumulative: it
covers everything from **L01** as well as the randomized training order and the precision
threshold added here, so it supersedes the **L01** suite entirely.

Once you've carried your code forward into this lecture's exercises directory, build and run it as
before:

```bash
make -C test
```

All 20 test cases should pass. Note that `train()` prints its progress, so the output is
interleaved with precision reports. Look for the summary on the last line.

See the [test suite's README](./exercises/test/README.md) for more information, including why the
tests that check exact values train on a single training set now that the order is randomized.

---
