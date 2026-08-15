# Appendix B - Exercises

## Linear Regression by Hand
**1.** A regression model is to be trained using the five training sets defined by the formula `y = 3x + 2`:

| x | y  |
|:-:|:--:|
| 0 |  2 |
| 1 |  5 |
| 2 |  8 |
| 3 | 11 |
| 4 | 14 |

Assume the model's bias (m-value) and weight (k-value) are zero at the start:

$$\begin{cases} k = 0 \\ m = 0 \end{cases}$$

Run training for one epoch with a learning rate LR of 10%:

$$LR = 0.1$$

Use the same update rule as the worked example in [appendix A](./a_theory.md), i.e. `m += δ * LR`
and `k += δ * LR * x` for every training set, including the one at `x = 0`. The `x = 0` shortcut
introduced in section 12 below is a different rule and gives different answers here.

Then perform prediction for input consisting of all integers in the range [-5, 5].

---

## Implementing Linear Regression in C++
You'll implement an interface and an accompanying class for a linear regression model in C++.

### 1. Directory structure
You'll write your code in this lecture's [exercises](../exercises/) directory. Build up the
following structure inside it:

```
exercises/
├── include/
│   └── ml/
│       ├── lin_reg/
│       │   ├── fixed.hpp
│       │   └── interface.hpp
│       └── types.hpp
├── source/
│   ├── ml/
│   │   └── lin_reg/
│   │       └── fixed.cpp
│   └── main.cpp
├── test/                <- already here, see section 14
└── Makefile
```

The `test` directory is already in place; [section 14](#14-running-the-tests) covers running it.

---

### 2. Makefile
Write a makefile that compiles and builds a program with the file structure above via the `make` command.

Test-run the program by adding an empty main function to `main.cpp`.

---

### 3. main.cpp
Paste the contents from [main.cpp](#maincpp) at the bottom of this document into the corresponding file.

---

### 4. Matrix alias
In the header file `ml/types.hpp`, add a namespace named `ml`.

In this namespace, implement the following aliases:
* `Matrix1d`: alias for one-dimensional floating-point vectors (`std::vector<double>`).
* `Matrix2d`: alias for two-dimensional floating-point vectors (`std::vector<Matrix1d>`).
* `Matrix3d`: alias for three-dimensional floating-point vectors (`std::vector<Matrix2d>`).

---

### 5. Interface
In the header file `ml/lin_reg/interface.hpp`, add a namespace named `ml::lin_reg`.

In this namespace, implement an interface named `Interface` with the following public methods:
* **`~Interface()`:**
    * The interface's destructor.
    * Should be marked `default`, `virtual`, and `noexcept`.
* **`predict()`:**
    * Should perform a prediction.
    * Should take a single floating-point argument named `input`.
    * Should return the predicted value as a floating-point number.
    * Should be pure virtual (`= 0`).
    * Should be marked `const` and `noexcept`.

---

### 6. Subclass: Fixed
In the header file `ml/lin_reg/fixed.hpp`, add the namespace `ml::lin_reg`. Implement a subclass named `Fixed` that inherits `Interface` via public inheritance. The class shouldn't be inheritable further, i.e. it should be marked `final`.

The class is named `Fixed` because it's trained with a fixed learning rate. An adaptive-learning-rate variant, `Adaptive`, follows the same overall structure but adjusts its learning rate dynamically during training.

The class should have the following public methods:
* **`Fixed()`:**
    * The class's only implemented constructor.
    * Should take the following arguments:
        * `trainIn`: reference to a read-only vector of floating-point numbers (training data, input).
        * `trainOut`: reference to a read-only vector of floating-point numbers (training data, output).
    * Should be marked `explicit` and `noexcept`.
* **`~Fixed()`:**
    * Destructor overriding the interface's destructor.
    * Should be marked `default`, `noexcept`, and `override`.
* **`predict()`:**
    * Overrides the corresponding method in the interface.
    * Should be marked `const`, `noexcept`, and `override`.
* **`train()`:**
    * Trains the model with the arguments:
        * `epochCount`: number of epochs to train (unsigned integer).
        * `learningRate`: learning rate as a floating-point number. Default value: `0.01` (1 %).
    * Returns `false` if `epochCount` is 0, or if `learningRate` is outside the range `(0.0, 1.0)`. Otherwise returns `true` once training is complete.
    * Should be marked `noexcept`.

The class should have the following private method:
* **`optimize()`:**
    * Arguments:
        * `input`: input value (`x`) as a floating-point number.
        * `output`: reference value (`y_ref`) as a floating-point number.
        * `learningRate`: the learning rate (`LR`) as a floating-point number.
    * Returns nothing.
    * Should be marked `noexcept`.

---

### 7. Removed constructors and operators
Delete the class's default constructor, copy and move constructors, and the corresponding operators.

---

### 8. Private member variables
Add the following private member variables to `Fixed`:
* **`myTrainIn`:**
    * Reference to the training data's input points.
    * Initialized via the constructor.
    * Should be marked `const`.
* **`myTrainOut`:**
    * Reference to the training data's output points.
    * Initialized via the constructor.
    * Should be marked `const`.
* **`mySetCount`:**
    * The number of complete training sets, i.e. `std::min(trainIn.size(), trainOut.size())` (`std::min` is available in `<algorithm>`).
    * Training data holding more inputs than outputs, or the other way around, has no complete set beyond the shorter of the two vectors, so the surplus values are unusable.
    * Initialized via the constructor.
    * Should be an unsigned integer (`std::size_t`, from `<cstddef>`) and marked `const`.
* **`myBias`:**
    * The model's bias value (the m-value in `y = kx + m`).
    * Should be a floating-point type.
* **`myWeight`:**
    * The model's weight value (the k-value in `y = kx + m`).
    * Should be a floating-point type.

---

### 9. Constructor
Implement the constructor in `source/ml/lin_reg/fixed.cpp`:
* Initialize all member variables.
* Set `mySetCount` to the number of complete training sets, i.e. the smaller of the two vectors' sizes.
* `myBias` and `myWeight` must both be initialized to `0.0`, so that the model predicts `0` for every input until it has been trained.
* Print an error message to `stderr` and call `std::terminate()` (from `<exception>`) if `mySetCount` is 0. A model without training data can never be trained, and since the constructor can't return a failure code, there's no way to report the problem to the caller.

---

### 10. Prediction
Implement the method `predict()` in `source/ml/lin_reg/fixed.cpp`.

Return a prediction according to the formula:

$$y = kx + m$$

where:
* $y$ is the prediction returned.
* $k$ is the model's weight (`myWeight`).
* $x$ is the input (`input`).
* $m$ is the model's bias (`myBias`).

---

### 11. Training
Implement the method `train()` in `source/ml/lin_reg/fixed.cpp`:
* Return `false` immediately if `epochCount` is 0, or if `learningRate` is outside the range `(0.0, 1.0)`:
    * A learning rate of `0.0` or less can never improve the model.
    * A learning rate of `1.0` or more corrects by at least the full error each step, which makes training oscillate or diverge instead of converging.
* Missing training data is already handled by the constructor, so `train()` doesn't need to check for it.
* Train the model for the given number of epochs.
* During each epoch, iterate through all `mySetCount` training sets. Perform optimization by calling the method `optimize()` with the current training set (obtained from the vectors, e.g. via index).
* Return `true` after training is complete.

---

### 12. Optimization
Implement the method `optimize()` in `source/ml/lin_reg/fixed.cpp`:
* If `input == 0`: set `myBias = output` directly and return, leaving `myWeight` untouched. Since `y = k * 0 + m = m`, the reference value *is* the bias, and no value of `k` changes a prediction made with an input of zero.
* Otherwise:
    * Compute a prediction with the given input: `prediction = myWeight * input + myBias`.
    * Compute the error as the difference between the reference value and the prediction: `error = output - prediction`.
    * Adjust the model's parameters based on the error:
        * `myBias   = myBias   + error * learningRate`
        * `myWeight = myWeight + error * learningRate * input`

---

### 13. Compiling and running
Compile the program and make sure you get the following output:

```
Input: 0, prediction: 2
Input: 1, prediction: 5
Input: 2, prediction: 8
Input: 3, prediction: 11
Input: 4, prediction: 14
```

The model has been trained on the same five training sets you trained by hand at the top of this
document, so this output is what `y = 3x + 2` looks like once the model has found the line on its
own.

---

### 14. Running the tests
A ready-made test suite for the model is available in [exercises/test](../exercises/test/). It
checks the behaviour specified above: the initial parameter values, the argument validation in
`train()`, the parameter updates made by `optimize()`, and a handful of complete training runs.

There's nothing to copy: `test` already sits alongside the code you just wrote. Build and run it
from the exercises directory:

```bash
make -C test
```

The test framework is checked out as a submodule at the root of this repository. If `libs/test` is
empty, fetch it once with `git submodule update --init`.

All 15 test cases should pass. When one fails, the output names the assertion that failed, the two
values involved, and the file and line it came from. Use it to find the part of the specification
your implementation doesn't match yet.

See the [test suite's README](../exercises/test/README.md) for more information, including the one
piece of behaviour it deliberately doesn't cover.

---

### 15. Tuning the training
The `main.cpp` you pasted trains for 1000 epochs at a learning rate of 10%, which is far more work
than these five training sets need. Tune the two constants at the top of `main()`, i.e. lower
`epochCount` and raise `learningRate`, and find the fewest epochs that still reproduce the output
in [section 13](#13-compiling-and-running) exactly.

Change one constant at a time and rebuild with `make` after each change. A few things worth
knowing while you experiment:
* `learningRate` must stay inside the range `(0.0, 1.0)`, otherwise `train()` returns `false` and
  the program prints `Training failed!` instead of any predictions.
* Don't expect the epoch count to drop smoothly as the learning rate rises. There's a stretch where
  a higher learning rate needs *more* epochs, not fewer, because each pass overshoots the line and
  the next has to correct back.
* Raise the learning rate far enough and training stops converging altogether: the predictions grow
  by orders of magnitude per epoch and print as values like `-2.60711e+06`, or as `inf` and `nan`
  once the rate is higher still. Find roughly where that starts. It's well below the 1.0 the
  interface allows, which is why the limit in `train()` is an outer bound rather than a
  recommendation.
* The predictions are printed with `%g`, which shows six significant digits. Matching the output in
  section 13 therefore means "correct to six digits", not "exact". Print with `%.10f` instead if
  you want to see the error that's left.
* The test suite in section 14 doesn't use `main.cpp`, so it keeps passing no matter which
  constants you settle on.

It can be done in well under ten epochs. Note down the pair you end up with, along with roughly
where training started to diverge.

Keep the pair you settled on in `main.cpp` here, but set `epochCount` back to `1000` and
`learningRate` back to `0.1` when you carry the code forward into **L02**. That lecture teaches the
model to stop training on its own once it's accurate enough, which only shows as a difference when
the model is given more epochs than it needs.

---

## main.cpp

```cpp
/**
 * @file Application entry point.
 */
#include <cstdio>

#include "ml/lin_reg/fixed.hpp"
#include "ml/types.hpp"

namespace
{
/**
 * @brief Evaluate the model with the given inputs.
 *
 * @param[in] linReg Model to evaluate.
 * @param[in] inputs Input values for the evaluation.
 */
void evaluateModel(const ml::lin_reg::Interface& linReg, const ml::Matrix1d& inputs) noexcept
{
    for (const auto& input : inputs)
    {
        std::printf("Input: %g, ", input);
        std::printf("prediction: %g\n", linReg.predict(input));
    }
}
} // namespace

/**
 * @brief Train and predict with a linear regression model.
 *
 * @return 0 if the model was trained, -1 otherwise.
 */
int main()
{
    constexpr std::size_t epochCount{1000U};
    constexpr double learningRate{0.1};

    // Create linear regression model holding five training sets (y = 3x + 2).
    const ml::Matrix1d trainIn{0.0, 1.0, 2.0, 3.0, 4.0};
    const ml::Matrix1d trainOut{2.0, 5.0, 8.0, 11.0, 14.0};
    ml::lin_reg::Fixed linReg{trainIn, trainOut};

    // Train the model, print the results on success.
    const auto trained = linReg.train(epochCount, learningRate);

    if (trained) { evaluateModel(linReg, trainIn); }
    else { std::fprintf(stderr, "Training failed!\n"); }
    return trained ? 0 : -1;
}
```

---
