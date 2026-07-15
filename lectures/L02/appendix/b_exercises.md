# Appendix B - Exercises
You'll extend the `ml::lin_reg::Fixed` class from **L01**, first with a randomized training order, then with precision calculation.

---

## Randomizing the Training Order

### 1. MatrixU32 alias
Add the alias `MatrixU32` to the header file `ml/types.hpp` as a substitute for `std::vector<std::uint32_t>`, in the namespace `ml`.

---

### 2. Random number generator
Add a function named `initRandom()` in an anonymous namespace in `source/ml/lin_reg/fixed.cpp`. The function should initialize the random number generator exactly once when called:
* Use a static local variable named `initialized` (of type `bool`, initialized to `false`) to track whether the generator has already been initialized.
* Initialize the generator with `std::srand(std::time(nullptr))`.
* Set `initialized` to `true` after initialization.
* Include `<cstdlib>` and `<ctime>` in `fixed.cpp`.
* Should be marked `noexcept`.

---

### 3. Training order
Add a private member variable named `myTrainOrder` to `Fixed`:
* Should hold the indices of the training sets as unsigned integers (`ml::MatrixU32`).
* Initialized in the constructor with the indices `0, 1, 2 ... N-1`, where `N` is the number of training sets.

---

### 4. Shuffling
Add a private method named `shuffle()` to `Fixed`:
* Should shuffle the contents of `myTrainOrder` in random order.
* For each index `i`, pick a random index `r` and swap `myTrainOrder[i]` and `myTrainOrder[r]`.
* Should be marked `noexcept`.

---

### 5. Updating the training method
Update the method `train()` in `source/ml/lin_reg/fixed.cpp`:
* Call `initRandom()` at the start of the method.
* Call `shuffle(myTrainOrder)` at the start of every epoch.
* Iterate through the training sets in the order given by `myTrainOrder` instead of sequentially by index.

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
* Should be marked `const` and `noexcept`.

---

### 2. Updating the training method
Update the method `train()` in `source/ml/lin_reg/fixed.cpp` so training stops early once the precision exceeds a given threshold:
* Add an argument named `precision` (a floating-point number specifying the precision threshold). Default value: `0.999999` (99.9999 %).
* Print an error message and call `std::terminate()` (from `<exception>`) if `precision` falls outside the range (0.0, 1.0).
* Call `precision()` every tenth epoch — computing it scans every training set, so checking less often than every epoch keeps that overhead down over thousands of epochs.
* Stop training and return `true` if the precision exceeds `precision`.

---

### 3. Compiling and testing
Compile and test-run the program. The model should now stop training as soon as the precision exceeds the threshold. Print the number of epochs and the achieved precision in `train()` to verify.

---
