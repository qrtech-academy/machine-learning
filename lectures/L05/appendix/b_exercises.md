# Appendix B - Exercises
You'll extend your `ml` codebase from **L04**, first declaring the `Dense` class and implementing
its methods, then extending it further: observing overfitting, adding momentum to `optimize()`,
and debugging a numerically unstable network.

---

## Declaring the Dense Layer
You'll extend your `ml` codebase from **L04** by implementing a concrete subclass `Dense`, replacing the temporary stub class `ml::dense_layer::Stub`. See [appendix A](./a_theory.md) for a walkthrough of the dense layer's architecture before you start.

---

### 1. The `ActFunc` enum class
In `ml/types.hpp`, define an enum class named `ActFunc`. This enum class should be usable to select the activation function for a given dense layer. Add the following enumerators:
* `Relu`: for ReLU (`Rectified Linear Unit`), which for a given input x returns x if x > 0, otherwise 0.
* `Tanh`: for hyperbolic tangent, which produces output in the range [-1, 1].

---

### 2. Create `dense.hpp`
Create a new file named `dense.hpp` in `include/ml/dense_layer`. The file path should therefore be `include/ml/dense_layer/dense.hpp`, and the file should be includable as shown below:

```cpp
#include "ml/dense_layer/dense.hpp"
```

---

### 3. Create `dense.cpp`
Create a new file named `dense.cpp` in `source/dense_layer`. The file path should therefore be `source/dense_layer/dense.cpp`.

**Note! Don't forget to add this file to your makefile, see below!**

```makefile
# Application target.
TARGET := app

# Source files.
# File added to the build here.
SRC_FILES := source/dense_layer/dense.cpp\
                source/dense_layer/stub.cpp\
                source/main.cpp\
				source/neural_network/shallow.cpp\

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
In the header file `ml/dense_layer/dense.hpp`, add a class named `Dense` that inherits the corresponding interface, see the file `include/ml/dense_layer/interface.hpp`:
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
Define all methods, constructors, etc. that aren't marked `delete` or `default` in the file `source/dense_layer/dense.cpp`:

**Constructor:**
* In the constructor, all member variables should be initialized:
    * `myOutput`, `myPreActivationOutput`, and `myError` should hold `nodeCount` floating-point values equal to 0.0 at the start.
    * `myBias` should hold `nodeCount` floating-point values randomized between 0.0 - 1.0.
    * `myWeights` should hold `nodeCount` x `weightCount` floating-point values randomized between 0.0 - 1.0.
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
    * The three unimplemented methods should have an empty function body (they return nothing, `void`).

---

## Implementing the Dense Layer
You'll complete the class `Dense` you just declared by implementing the methods `feedforward()`, `backpropagate()`, and `optimize()`. See [appendix A](./a_theory.md) for an overview of how 
the math from **L03** maps onto the code below.

---

### 1. Helper functions
In the file `source/dense_layer/dense.cpp`, create an anonymous namespace. In this namespace, define 
the following helper functions:
* `initRandom()`: function to initialize the random number generator once.
    * **Implementation:**
        * Add a static local variable named `initialized` with an initial value of `false`. 
        Since the variable is static, it retains its value between function calls, which means 
        we only initialize the random number generator once.
        * If `initialized == true`, exit the function early (with `return`).
        * Initialize the random number generator with the current time by calling `std::srand(std::time(nullptr))`. To do this, include `<cstdlib>` and `<ctime>`.
        * After initialization, set `initialized = true` so initialization doesn't happen the next time the function is called.
    * Should be marked `noexcept`.

* `randomStartVal()`: function to generate and return a random number between 0.0 and 1.0.
    * **Implementation:**
        * Generate a random number in the range (0.0, 1.0) by calling `std::rand()`, which generates a random number between (0, RAND_MAX), and dividing by RAND_MAX.
        * One of the operands must be converted to a floating-point number to avoid integer division, e.g. `static_cast<double>(RAND_MAX)`.
    * Should be marked `noexcept`.

* `actFuncOutput()`: function to compute and return the output (floating-point) of a given activation function.
    * **Takes:**
        * `actFunc`: the activation function to use (of type `ActFunc`).
        * `input`: input value to the activation function (floating-point).
    * **Implementation:**
        * Use a switch statement to compute the output depending on the given activation function:
            * `ActFunc::Relu`: return `input` if `input > 0.0`, otherwise `0.0`.
            * `ActFunc::Tanh`: return `std::tanh(input)` (requires `#include <cmath>`).
            * Default case: print the error message `"Invalid activation function!"` and return `0.0`.
    * Should be marked `noexcept`.

* `actFuncDelta()`: function to compute and return the derivative (floating-point) of a given activation function.
    * **Takes:**
        * `actFunc`: the activation function to use (of type `ActFunc`).
        * `input`: input value to the activation function (floating-point).
    * **Implementation:**
        * Use a switch statement to compute the derivative depending on the given activation function:
            * `ActFunc::Relu`: return `1.0` if `input > 0.0`, otherwise `0.0`.
            * `ActFunc::Tanh`: compute `const auto tanhOutput = std::tanh(input)` and return `1.0 - tanhOutput * tanhOutput`.
            * Default case: print the error message `"Invalid activation function!"` and return `0.0`.
    * Should be marked `noexcept`.

---

### 2. Randomizing bias and weights
Randomize all bias values and weights:
* In the constructor, first call `initRandom()` to initialize the random number generator.
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
* If the dimensions don't match: print the error message `"Input dimension mismatch: expected X, actual: Y!"` and call `std::terminate()` (from `<exception>`).

**Computation for each node:**
* Iterate through every node in the layer with a for loop: `for (std::size_t i{}; i < nodeCount(); ++i)`.
* For each node `i`, compute the weighted sum:
    1. Start with the node's bias value: `auto sum{myBias[i]}`.
    2. Add each weight multiplied by the corresponding input: `for (std::size_t j{}; j < weightCount(); ++j)` where `sum += myWeights[i][j] * input[j]`.
* Store the weighted sum before the activation function is applied: `myPreActivationOutput[i] = sum`.
  This value is needed by `backpropagate()` below to correctly compute the activation function's derivative.
* Apply the activation function to the sum: `myOutput[i] = actFuncOutput(myActFunc, sum)`.

**Return value:** nothing (`void`).

---

### 4. The method `backpropagate()` (output layer)
Implement `backpropagate()` for output layers (with reference values):

**Input validation:**
* Check that the size of the reference vector matches the number of nodes (`reference.size() == nodeCount()`).
* If the dimensions don't match: print the error message `"Output dimension mismatch: expected X, actual: Y!"` and call `std::terminate()` (from `<exception>`).

**Error computation for each node:**
* Iterate through every node in the layer: `for (std::size_t i{}; i < nodeCount(); ++i)`.
* For each node `i`:
    1. Compute the raw error: `const auto err{reference[i] - myOutput[i]}`.
    2. Compute the gradient error: `myError[i] = err * actFuncDelta(myActFunc, myPreActivationOutput[i])`.
       **Note!** Use `myPreActivationOutput[i]` (the weighted sum before the activation function was
       applied in `feedforward()`), not `myOutput[i]`. `actFuncDelta()` expects the activation
       function's *input*, not its output - otherwise the derivative comes out wrong for
       `ActFunc::Tanh` (it happens to work for `ActFunc::Relu` by coincidence).

**Return value:** nothing (`void`).

---

### 5. The method `backpropagate()` (hidden layer)
Implement `backpropagate()` for hidden layers (with error and weights from the next layer):

**Input validation:**
* Check that the next layer's weight count matches this layer's node count (`nextLayer.weightCount() == nodeCount()`).
* If the dimensions don't match: print the error message `"Layer dimension mismatch: expected X, actual: Y!"` and call `std::terminate()` (from `<exception>`).

**Error computation for each node:**
* Iterate through every node in this layer: `for (std::size_t i{}; i < nodeCount(); ++i)`.
* For each node `i`:
    1. Initialize a variable to store the computed raw error: `double err{}`.
    2. Sum all errors from the next layer: `for (std::size_t j{}; j < nextLayer.nodeCount(); ++j)`
        * `err += nextLayer.error()[j] * nextLayer.weights()[j][i]`.
    3. Compute the gradient error: `myError[i] = err * actFuncDelta(myActFunc, myPreActivationOutput[i])`
       (see the note in the previous section for why `myPreActivationOutput[i]` is used instead
       of `myOutput[i]`).

**Return value:** nothing (`void`).

---

### 6. The method `optimize()`
**Input validation:**
* Check that the learning rate is valid (`learningRate > 0.0`).
* If the learning rate is invalid: print the error message `"Invalid learning rate X!"` and call `std::terminate()` (from `<exception>`).
* Check that the input size matches the number of weights per node (`input.size() == weightCount()`).
* If the dimensions don't match: print the error message `"Input dimension mismatch: expected X, actual: Y!"` and call `std::terminate()` (from `<exception>`).

**Parameter update for each node:**
* Iterate through every node in the layer: `for (std::size_t i{}; i < nodeCount(); ++i)`.
* For each node `i`:
    1. Update the bias: `myBias[i] += myError[i] * learningRate`.
    2. Update all weights: `for (std::size_t j{}; j < weightCount(); ++j)`
        * `myWeights[i][j] += myError[i] * learningRate * input[j]`

**Return value:** nothing (`void`).

---

## Overfitting and Generalization

### Setup
Using your `Shallow`/`Dense` classes from L04-L05:
1. Build a small 1D dataset: pick around 20-25 points with a single input value each, labeled by a
   simple rule (e.g. input greater than zero → target `1.0`, otherwise → target `-1.0`). Normalize
   inputs to roughly the `[-1, 1]` range.
2. Deliberately inject noise: for 3-4 points close to the decision boundary, flip their label. This
   creates a few examples that genuinely contradict the general pattern — exactly the kind of
   noise a model can only "solve" by memorizing.
3. Split the data roughly two-to-one: about two-thirds for training, the rest held out as a test
   set never used in `train()`.
4. Construct a hidden layer with noticeably more nodes than the task needs (e.g. 16 hidden nodes
   for a problem this simple) — deliberately over-provisioned capacity is what makes overfitting
   easy to observe. Use `ActFunc::Tanh` rather than `ActFunc::Relu` for this exercise: with this
   few inputs and this many hidden nodes, ReLU nodes can easily land in the "dying ReLU" state
   described in **L03** and stop learning entirely, which would prevent the network from fitting
   *anything* rather than demonstrating overfitting.

### Running it
Train for several thousand epochs (e.g. 20,000), and every so often — not every epoch — compute
and print the mean absolute error (the same calculation as L02's `precision()`, just not subtracted
from 1) on both the training set and the test set.

For reference, a setup built exactly this way (24 points, 3 deliberately mislabeled, 16 training /
8 test, 16 hidden `Tanh` nodes, learning rate 0.05) produced this pattern:

| Epoch  | Train MAE | Test MAE |
|-------:|----------:|---------:|
| 1      | 0.273     | 0.817    |
| 100    | 0.089     | 0.280    |
| 1,000  | 0.016     | 0.250    |
| 4,000  | 0.006     | 0.250    |
| 10,000 | 0.004     | 0.250    |
| 20,000 | 0.002     | 0.250    |

Your own run won't match these numbers exactly — weights and biases are randomly initialized, and
your specific data points will differ — but the *shape* should: training error keeps shrinking
toward zero long after test error has locked in place.

### Reflection
* At roughly what epoch does test error stop improving in your run?
* Try reducing the hidden layer from 16 nodes down to, say, 2 or 4, and re-running. You may find
  the gap barely changes — with only one input value, even a couple of `Tanh` nodes turn out to
  have enough flexibility to memorize a training set this small. Capacity isn't the only thing
  that drives overfitting; how long you train for matters just as much, and a smaller network
  given enough epochs can overfit just as thoroughly as a larger one.
* Now try increasing the amount of training data instead (e.g. from ~16 up to ~50 points, keeping
  a similar noise rate), with the network size unchanged. This should make a much bigger
  difference than shrinking the network did — expect test error to drop substantially, likely at
  the cost of a slightly *higher* training error than before, since the network can no longer
  perfectly memorize a training set that large.

---

## Momentum

### Implementation
Extend the `Dense` class from this lecture:
1. Add two new private member variables to store velocity: `myVelocity` (same shape as `myBias`)
   and `myWeightVelocity` (same shape as `myWeights`), both initialized to `0` in the constructor.
2. Add a momentum coefficient parameter to `optimize()`, e.g. `momentum` with a default value of
   `0.9`.
3. Replace the direct bias/weight updates with the two-step momentum pattern:
   * `myVelocity[i] = momentum * myVelocity[i] + (myError[i] * learningRate)`, then
     `myBias[i] += myVelocity[i]`.
   * `myWeightVelocity[i][j] = momentum * myWeightVelocity[i][j] + (myError[i] * learningRate * input[j])`,
     then `myWeights[i][j] += myWeightVelocity[i][j]`.

### Comparing Against Plain SGD
Using the same network and dataset as your L04-L05 exercise (e.g. the 2-bit XOR pattern):
1. Train two separate instances of your network from the same random seed/starting weights — one
   using the original plain-SGD `optimize()`, one using the momentum version.
2. Every so often (e.g. every 50 epochs), compute and print the precision (from L02) of both.
3. Compare how many epochs each needs to first reach a fixed target precision, e.g. `0.9`.

For reference, one such comparison (4 hidden `Tanh` nodes, learning rate `0.05`, `β = 0.9`, target
precision `0.9`) reached that target in **50-250 epochs with momentum**, versus **350-2,300 epochs
with plain SGD**, across five different random initializations — momentum consistently reached the
same precision in a fraction of the epochs, though the exact numbers will differ for your own run
since initialization is randomized. A detailed epoch-by-epoch comparison for one such run:

| Epoch | Plain SGD | Momentum |
|------:|----------:|---------:|
| 1     | 0.373     | 0.386    |
| 100   | 0.447     | 0.475    |
| 250   | 0.461     | 0.922    |
| 500   | 0.469     | 0.976    |
| 1,000 | 0.473     | 0.985    |
| 5,000 | 0.976     | 0.994    |

Notice plain SGD does eventually reach a comparable precision — momentum isn't solving a different
problem, it's solving the same one faster.

---

## Debugging Numerical Instability

Take your L05 network and set the learning rate to something deliberately unreasonable — `50` or
higher. Print your hidden layer's weights (or the network's precision) after the very first epoch,
then again after several more. In a network this size, a learning rate that aggressive is enough to
saturate or kill every hidden node within the very first epoch — sometimes the very first training
sample — after which every subsequent epoch prints exactly the same numbers. That flat, unchanging
output *is* what a broken network looks like with this architecture: not a crash, just silence.

---

---
