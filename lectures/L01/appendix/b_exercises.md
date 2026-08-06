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

Then perform prediction for input consisting of all integers in the range [-5, 5].

---

## Implementing Linear Regression in C++
You'll implement an interface and an accompanying class for a linear regression model in C++.

### 1. Directory structure
You'll write your code in this lecture's [exercises](./exercises/) directory. Build up the
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
        * `trainInput`: reference to a read-only vector of floating-point numbers (training data, input).
        * `trainOutput`: reference to a read-only vector of floating-point numbers (training data, output).
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
* **`myTrainInput`:** 
    * Reference to the training data's input points.
    * Initialized via the constructor.
    * Should be marked `const`.
* **`myTrainOutput`:** 
    * Reference to the training data's output points.
    * Initialized via the constructor.
    * Should be marked `const`.
* **`mySetCount`:**
    * The number of complete training sets, i.e. `std::min(trainInput.size(), trainOutput.size())` (`std::min` is available in `<algorithm>`).
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
--------------------------------------------------------------------------------
Input: 0, predicted output: 2
Input: 1, predicted output: 4
Input: 2, predicted output: 6
Input: 3, predicted output: 8
Input: 4, predicted output: 10
--------------------------------------------------------------------------------
```

---

### 14. Running the tests
A ready-made test suite for the model is available in [exercises/test](./exercises/test/). It
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

See the [test suite's README](./exercises/test/README.md) for more information, including the one
piece of behaviour it deliberately doesn't cover.

---

## main.cpp

```cpp
/**
 * @brief Linear regression demonstration.
 */
#include <cstdint>
#include <cstdio>

#include "ml/lin_reg/fixed.hpp"
#include "ml/types.hpp"

namespace
{
/**
 * @brief Predict with the given linear regression model.
 *
 * @param[in] linReg Linear regression model to predict with.
 * @param[in] inputData Input data to predict with.
 */
void predict(const ml::lin_reg::Interface& linReg, const ml::Matrix1d& inputData) noexcept
{
    // Check the number of input sets, terminate if missing.
    if (inputData.empty())
    {
        std::printf("No input data!\n");
        return;
    }

    // Perform prediction with each input set, print the result in the terminal.
    std::printf(
        "--------------------------------------------------------------------------------\n");
    for (const auto& input : inputData)
    {
        const auto prediction = linReg.predict(input);
        std::printf("Input: %g, predicted output: %g\n", input, prediction);
    }
    std::printf(
        "--------------------------------------------------------------------------------\n\n");
}
} // namespace

/**
 * @brief Train and predict with a linear regression model.
 *
 * @return 0 on success, or -1 on failure.
 */
int main()
{
    constexpr std::uint32_t epochCount{1000U};
    constexpr double learningRate{0.1};

    // Create linear regression model to predict y = 2x + 2.
    const ml::Matrix1d trainInput{0.0, 1.0, 2.0, 3.0, 4.0};
    const ml::Matrix1d trainOutput{2.0, 4.0, 6.0, 8.0, 10.0};
    ml::lin_reg::Fixed linReg{trainInput, trainOutput};

    // Train the model, terminate on failure.
    if (!linReg.train(epochCount, learningRate))
    {
        std::printf("Training failed!\n");
        return -1;
    }

    // Perform prediction with all input sets.
    predict(linReg, trainInput);
    return 0;
}
```

---
