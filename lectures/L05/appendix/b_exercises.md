# Appendix B - Exercises
You'll extend your `ml` codebase from **L04** by declaring the `Dense` class and implementing its
methods, so your network finally runs on a layer that computes something rather than the stub
it's been wired to since **L03**.

---

## Declaring the Dense Layer
You'll extend your `ml` codebase from **L04** by implementing a concrete subclass `Dense`, which takes over from `ml::dense_layer::Stub` as the layer your network is built from. The stub itself stays in the codebase; see section 7. See [appendix A](./a_theory.md) for a walkthrough of the dense layer's architecture before you start.

---

### 1. The `ActFunc` enum class
In `ml/types.hpp`, define an enum class named `ActFunc`. This enum class should be usable to select the activation function for a given dense layer. Add the following enumerators:
* `Relu`: for ReLU (`Rectified Linear Unit`), which for a given input x returns x if x > 0, otherwise 0.
* `Tanh`: for hyperbolic tangent, which produces output in the range [-1, 1].
* `None`: for no activation at all, i.e. the identity function, which returns x unchanged. Useful on an output layer that has to produce values outside the range a bounded activation function can reach.

---

### 2. Create `dense.hpp`
Create a new file named `dense.hpp` in `include/ml/dense_layer`. The file path should therefore be `include/ml/dense_layer/dense.hpp`, and the file should be includable as shown below:

```cpp
#include "ml/dense_layer/dense.hpp"
```

---

### 3. Create `dense.cpp`
Create a new file named `dense.cpp` in `source/ml/dense_layer`. The file path should therefore be `source/ml/dense_layer/dense.cpp`.

**Note! Don't forget to add this file to your makefile, see below!**

```makefile
# Application target.
TARGET := app

# Source files.
# File added to the build here.
SRC_FILES := source/main.cpp\
                source/ml/dense_layer/dense.cpp\
                source/ml/neural_network/shallow.cpp\
                source/ml/utils.cpp\

# Include directory.
INC_DIR := include

# Compiler flags.
COMPILER_FLAGS := -Wall -Werror -std=c++17

# Build and run the application as default.
default: build run

# Build the application.
build:
	@g++ $(SRC_FILES) -o $(TARGET) -I$(INC_DIR) $(COMPILER_FLAGS)

# Run the application.
run:
	@./$(TARGET)

# Clean the application.
clean:
	@rm -f $(TARGET)
```

---

### 4. The `Dense` class - declaration
In the header file `include/ml/dense_layer/dense.hpp`, in the namespace `ml::dense_layer`, add a class named `Dense` that inherits the corresponding interface, see the file `include/ml/dense_layer/interface.hpp`:
* Use public inheritance and mark the class `final` so it can't be inherited further.
* Override every method from the interface, including the destructor.

**Tip**: copy the entire contents of the interface and paste it into the new file. Then adapt the code for the new subclass `Dense` (no `virtual` or `= 0`, use `override`, etc.).

---

### 5. Removed constructors and operators
Delete the following:
* The default constructor.
* The copy constructor.
* The move constructor.
* The copy operator.
* The move operator.

---

### 6. Constructor
Create a constructor that can be used to create a dense layer with arbitrary dimensions and a selectable activation function.
* **Takes:**
    * `nodeCount`: the number of nodes in the layer (unsigned integer).
    * `weightCount`: the number of weights per node in the layer (unsigned integer).
    * `actFunc`: the activation function to use (of type `ActFunc`). `ReLU` should be used as the default.
* Should be marked `explicit` and `noexcept`.
* If `nodeCount` or `weightCount` equals 0, an error message should be printed and the program terminated by calling `std::terminate()`.

---

### 7. Private member variables
Add the following private member variables to the class:
* `myOutput`: vector holding the nodes' output (floating-point). Should hold `nodeCount` values.
* `myPreActivationOutput`: vector holding the nodes' weighted sum before the activation function has
  been applied (floating-point). Should hold `nodeCount` values. Used by `backpropagate()`
  (implemented later this lecture) to correctly compute the activation function's derivative.
* `myError`: vector holding the nodes' error (floating-point). Should hold `nodeCount` values.
* `myBias`: vector holding the nodes' bias values (floating-point). Should hold `nodeCount` values.
* `myWeights`: two-dimensional vector holding the nodes' weights. Should hold `nodeCount` x `weightCount` values.
* `myActFunc`: the layer's activation function (of type `ActFunc`).

---

### 8. Defining the methods
Define all methods, constructors, etc. that aren't marked `delete` or `default` in the file `source/ml/dense_layer/dense.cpp`:

**Constructor:**
* In the constructor, all member variables should be initialized:
    * `myOutput`, `myPreActivationOutput`, and `myError` should hold `nodeCount` floating-point values equal to 0.0 at the start.
    * `myBias` should hold `nodeCount` floating-point values randomized in `[0.0, 1.0]`.
    * `myWeights` should hold `nodeCount` x `weightCount` floating-point values randomized in `[0.0, 1.0]`.
    * `myActFunc` should be assigned the given activation function.
* If `nodeCount` or `weightCount` equals 0, the error message `Invalid dense layer parameters: nodeCount and weightCount must be greater than 0!` should be printed to standard error, after which the program should terminate by calling `std::terminate()`, as shown below:

```cpp
#include <cstdio>
#include <exception>

if ((0U == nodeCount) || (0U == weightCount))
{
    std::fprintf(stderr, "Invalid dense layer parameters: "
        "nodeCount and weightCount must be greater than 0!\n");
    std::terminate();
}
```

**Other methods:**
* Implement every method from the interface except `feedforward()`, `backpropagate()`, and `optimize()`.
* Follow the descriptions in `include/ml/dense_layer/interface.hpp`:
    * Methods such as `nodeCount()` and `weightCount()` should return the number of nodes and weights per node in the layer, respectively.
    * Getter methods such as `output()` and `error()` should return references to the corresponding member variables.
    * The three unimplemented methods return `bool` (see **L03**), so an empty body won't compile. Give each one a placeholder body of `return false;` until you implement it below. `false` rather than `true`, so that a method you forget to finish reports failure instead of quietly claiming success.

---

## Implementing the Dense Layer
You'll complete the class `Dense` you just declared by implementing the methods `feedforward()`, `backpropagate()`, and `optimize()`. See [appendix A](./a_theory.md) for an overview of how
the math from **L03** maps onto the code below.

---

### 1. Helper functions
`Dense` needs three helpers that don't belong to any one class. Add them to `ml/utils.hpp` and
`source/ml/utils.cpp`, the pair you created in **L04**, in the namespace `ml`. Declare each one in
the header and define it in the `.cpp`, alongside the `initRandGen()` already there.

* `randomStartVal()`: function to generate and return a random number in the range `[0.0, 1.0]`.
    * **Implementation:**
        * Generate a random number in the range `[0.0, 1.0]` by calling `std::rand()`, which generates a random number in `[0, RAND_MAX]`, and dividing by RAND_MAX. Both ranges are closed at each end: a draw of exactly `RAND_MAX` gives exactly `1.0`.
        * One of the operands must be converted to a floating-point number to avoid integer division, e.g. `static_cast<double>(RAND_MAX)`.
    * Should be marked `[[nodiscard]]` and `noexcept`.
    * It belongs next to `initRandGen()`: both wrap `std::rand()`, and calling this one without having called that one first gives the same sequence on every run.

* `actFuncOutput()`: function to compute and return the output (floating-point) of a given activation function.
    * **Takes:**
        * `actFunc`: the activation function to use (of type `ActFunc`).
        * `value`: input value to the activation function (floating-point).
    * **Implementation:**
        * Use a switch statement to compute the output depending on the given activation function:
            * `ActFunc::Relu`: return `value` if `value > 0.0`, otherwise `0.0`.
            * `ActFunc::Tanh`: return `std::tanh(value)` (requires `#include <cmath>`).
            * Default case (`ActFunc::None`): return `value` unchanged, which is the identity function.
    * Should be marked `[[nodiscard]]` and `noexcept`.

* `actFuncDelta()`: function to compute and return the derivative (floating-point) of a given activation function.
    * **Takes:**
        * `actFunc`: the activation function to use (of type `ActFunc`).
        * `value`: input value to the activation function (floating-point).
    * **Implementation:**
        * Use a switch statement to compute the derivative depending on the given activation function:
            * `ActFunc::Relu`: return `1.0` if `value > 0.0`, otherwise `0.0`.
            * `ActFunc::Tanh`: compute `const auto tanhOutput = std::tanh(value)` and return `1.0 - tanhOutput * tanhOutput`.
            * Default case (`ActFunc::None`): return `1.0`, the derivative of the identity function.
    * Should be marked `[[nodiscard]]` and `noexcept`.

Neither activation function needs an error case. `ActFunc` has exactly three enumerators, and all three are handled, so the default case is `None` rather than something that shouldn't happen.

---

### 2. Randomizing bias and weights
Randomize all bias values and weights:
* In the constructor, first call `initRandGen()` (from `ml/utils.hpp`) to seed the random number generator.
* Iterate through every node in the layer with a for loop: `for (std::size_t i{}; i < nodeCount; ++i)`.
* For each node `i`:
    * Assign a random number to its bias value by calling `randomStartVal()`: `myBias[i] = randomStartVal()`.
    * Iterate through the node's weights with a nested for loop: `for (std::size_t j{}; j < weightCount; ++j)`.
    * For each weight `j`:
        * Assign a random number to the weight by calling `randomStartVal()`: `myWeights[i][j] = randomStartVal()`.

---

### 3. The method `feedforward()`
**Input validation:**
* Check that the dimensions of the given input match the number of weights per node in the layer (`input.size() == weightCount()`).
* If the dimensions don't match: print the error message `"Input dimension mismatch: expected X, actual: Y!"` and return `false` without computing anything.

**Computation for each node:**
* Iterate through every node in the layer with a for loop: `for (std::size_t i{}; i < nodeCount(); ++i)`.
* For each node `i`, compute the weighted sum:
    1. Start with the node's bias value: `auto sum{myBias[i]}`.
    2. Add each weight multiplied by the corresponding input: `for (std::size_t j{}; j < weightCount(); ++j)` where `sum += myWeights[i][j] * input[j]`.
* Store the weighted sum before the activation function is applied: `myPreActivationOutput[i] = sum`.
  This value is needed by `backpropagate()` below to correctly compute the activation function's derivative.
* Apply the activation function to the sum: `myOutput[i] = actFuncOutput(myActFunc, sum)`.

**Return value:** `true` once every node has been computed.

---

### 4. The method `backpropagate()` (output layer)
Implement `backpropagate()` for output layers (with reference values):

**Input validation:**
* Check that the size of the reference vector matches the number of nodes (`reference.size() == nodeCount()`).
* If the dimensions don't match: print the error message `"Output dimension mismatch: expected X, actual: Y!"` and return `false` without computing anything.

**Error computation for each node:**
* Iterate through every node in the layer: `for (std::size_t i{}; i < nodeCount(); ++i)`.
* For each node `i`:
    1. Compute the raw error: `const auto err{reference[i] - myOutput[i]}`.
    2. Compute the gradient error: `myError[i] = err * actFuncDelta(myActFunc, myPreActivationOutput[i])`.
       **Note!** Use `myPreActivationOutput[i]` (the weighted sum before the activation function was
       applied in `feedforward()`), not `myOutput[i]`. `actFuncDelta()` expects the activation
       function's *input*, not its output; otherwise the derivative comes out wrong for
       `ActFunc::Tanh` (it happens to work for `ActFunc::Relu` by coincidence).

**Return value:** `true` once every node's error has been computed.

---

### 5. The method `backpropagate()` (hidden layer)
Implement `backpropagate()` for hidden layers (with error and weights from the next layer):

**Input validation:**
* Check that the next layer's weight count matches this layer's node count (`nextLayer.weightCount() == nodeCount()`).
* If the dimensions don't match: print the error message `"Layer dimension mismatch: expected X, actual: Y!"` and return `false` without computing anything.

**Error computation for each node:**
* Iterate through every node in this layer: `for (std::size_t i{}; i < nodeCount(); ++i)`.
* For each node `i`:
    1. Initialize a variable to store the computed raw error: `double err{}`.
    2. Sum all errors from the next layer: `for (std::size_t j{}; j < nextLayer.nodeCount(); ++j)`
        * `err += nextLayer.error()[j] * nextLayer.weights()[j][i]`.
    3. Compute the gradient error: `myError[i] = err * actFuncDelta(myActFunc, myPreActivationOutput[i])`
       (see the note in the previous section for why `myPreActivationOutput[i]` is used instead
       of `myOutput[i]`).

**Return value:** `true` once every node's error has been computed.

---

### 6. The method `optimize()`
**Input validation:**
* Check that the learning rate lies inside the range `(0.0, 1.0)`, the same range the stub checked in **L03**.
* If the learning rate is invalid: print the error message `"Invalid learning rate X!"` and return `false` without updating anything.
* Check that the input size matches the number of weights per node (`input.size() == weightCount()`).
* If the dimensions don't match: print the error message `"Input dimension mismatch: expected X, actual: Y!"` and return `false` without updating anything.

**Parameter update for each node:**
* Iterate through every node in the layer: `for (std::size_t i{}; i < nodeCount(); ++i)`.
* For each node `i`:
    1. Update the bias: `myBias[i] += myError[i] * learningRate`.
    2. Update all weights: `for (std::size_t j{}; j < weightCount(); ++j)`
        * `myWeights[i][j] += myError[i] * learningRate * input[j]`

**Return value:** `true` once every node's bias and weights have been updated.

---

### 7. Compiling and testing
Update `main.cpp` to use your new layer:
* Replace the two `ml::dense_layer::Stub` instances with `ml::dense_layer::Dense`, keeping the same
  wiring: the output layer's weight count must still match the hidden layer's node count.
* Give both layers `ActFunc::Tanh`, and make the hidden layer wider than the task strictly needs,
  e.g. 8 nodes. Keep the 2-bit XOR training data from **L04**.
* Train for a few thousand epochs (e.g. 5000) with a learning rate of `0.1`, then predict once per
  training set and print the results as before.

**Don't delete `stub.hpp`.** `Dense` replaces the stub in your *network*, not in your codebase. The
test suite still compiles both, and still uses the stub to test `Shallow`, since a layer that
computes nothing is what makes the network's wiring visible. See section 8.

Compile and test-run the program. Unlike **L04**, where every prediction was the stub's fixed
output, each prediction should now land within a few hundredths of its training target. Your exact
numbers will differ from run to run, because the bias and weights start randomized.

**Worth trying:** switch both layers to `ActFunc::Relu` and shrink the hidden layer to 2 nodes.
Roughly one run in four then fails to learn XOR at all, however long you train. `randomStartVal()`
never returns a negative number, so every node starts on ReLU's positive side, and a node whose
weighted sum is driven negative for every input has a derivative of exactly zero from then on: it
stops learning and can't recover. That's the dying ReLU problem from **L03**, and it's why the
test suite trains with `Tanh`.

---

### 8. Running the tests
The final test suite is available in [exercises/test](./exercises/test/). It's cumulative: it
carries the **L04** stub and network tests over unchanged and adds unit tests for `Dense` and for
the helpers in `ml/utils`.

Once you've carried your code forward into this lecture's exercises directory, build and run it:

```bash
make -C test
```

All 54 test cases should pass. Because `Dense` randomizes its own bias and weights, none of its
output can be predicted from the constructor arguments, so the tests recover the bias by feeding
the layer a vector of zeros: every weight term drops out and the weighted sum is the bias alone.
That makes every expected value exact rather than approximate.

The one to read is `BackpropagateUsesPreActivationDerivative`. It checks the **Note!** in sections
4 and 5 above, and only `ActFunc::Tanh` can catch that mistake: for `Relu` the two values agree,
since `max(0.0, s)` is positive exactly when `s` is.

See the [test suite's README](./exercises/test/README.md) for more information, including how the
convergence threshold was chosen.

---
