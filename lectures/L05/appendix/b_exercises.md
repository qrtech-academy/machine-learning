# Appendix B - Exercises
You'll extend your `ml` codebase from **L04** by declaring the `Dense` class and implementing its
methods, so your network finally runs on a layer that computes something rather than the stub
it's been wired to since **L03**.

---

## Declaring the Dense Layer
You'll extend your `ml` codebase from **L04** by implementing a concrete subclass `Dense`, which takes over from `ml::dense_layer::Stub` as the layer your network is built from. The stub itself stays in the codebase; see section 8 of the next part. See [appendix A](./a_theory.md) for a walkthrough of the dense layer's architecture before you start.

---

### 1. The `ActFunc` enum class
In `ml/types.hpp`, define an enum class named `ActFunc`. This enum class should be usable to select the activation function for a given dense layer. Add the following enumerators:
* `Relu`: for ReLU (`Rectified Linear Unit`), which for a given input x returns x if x > 0, otherwise 0.
* `Tanh`: for hyperbolic tangent, which produces output in the range (-1, 1).
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

**Note! Don't forget to add this file to your makefile!** The only change is the new
`source/ml/dense_layer/dense.cpp` line in `SRC_FILES`:

```makefile
# Target application.
TARGET := neural_network

# C++ compiler.
CXX_COMPILER := g++

# C++ compiler flags.
CXX_FLAGS := -Wall -Werror -std=c++17 -Iinclude

# Source files.
SRC_FILES := source/main.cpp \
             source/ml/dense_layer/dense.cpp \
             source/ml/neural_network/shallow.cpp \
             source/ml/utils.cpp \

# Build and run the target by default.
default: build run

# Build the target.
build:
	@$(CXX_COMPILER) $(SRC_FILES) -o $(TARGET) $(CXX_FLAGS)

# Run the target.
run:
	@./$(TARGET)

# Clean the target.
clean:
	@rm -f $(TARGET)

.PHONY: default build run clean
```

---

### 4. The `Dense` class - declaration
In the header file `include/ml/dense_layer/dense.hpp`, in the namespace `ml::dense_layer`, add a class named `Dense` that inherits the corresponding interface, see the file `include/ml/dense_layer/interface.hpp`:
* Use public inheritance and mark the class `final` so it can't be inherited further.
* Override every method from the interface, including the destructor.
* Keep `[[nodiscard]]` on every override whose interface method has it. Attributes aren't inherited, so the override needs its own (see **L03**).
* That includes `initParams()`, which `Stub` implements as an empty method for want of anything to reset (see **L03**). `Dense` has bias values and weights to draw again, so its override does real work; the implementation follows in section 2 of the next part.

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
    * `actFunc`: the activation function to use (of type `ActFunc`). `ActFunc::Relu` should be used as the default.
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
    * `myBias` should hold `nodeCount` floating-point values randomized in `[-1.0, 1.0]`.
    * `myWeights` should hold `nodeCount` x `weightCount` floating-point values randomized in `[-1.0, 1.0]`.
    * The randomizing itself is written once, in `initParams()`, and called from here; see section 2 of the next part.
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
* Implement every method from the interface except `feedforward()`, `backpropagate()`, `optimize()`, and `initParams()`.
* Follow the descriptions in `include/ml/dense_layer/interface.hpp`:
    * Methods such as `nodeCount()` and `weightCount()` should return the number of nodes and weights per node in the layer, respectively.
    * Getter methods such as `output()` and `error()` should return references to the corresponding member variables.
    * `feedforward()`, both overloads of `backpropagate()`, and `optimize()` return `bool` (see **L03**), so an empty body won't compile. Give each one a placeholder body of `return false;` until you implement it below. `false` rather than `true`, so that a method you forget to finish reports failure instead of quietly claiming success.
    * `initParams()` returns nothing, so an empty body compiles fine. Leave it empty until section 2 of the next part, where the constructor starts calling it. That's the body the stub keeps for good; for `Dense` it's a placeholder.

---

## Implementing the Dense Layer
You'll complete the class `Dense` you just declared by implementing the methods `initParams()`, `feedforward()`, `backpropagate()`, and `optimize()`, then change `Shallow::train()` to report the precision it reached. See [appendix A](./a_theory.md) for an overview of how
the math from **L03** maps onto the code below.

---

### 1. Helper functions
`Dense` needs three helpers that don't belong to any one class. Add them to `ml/utils.hpp` and
`source/ml/utils.cpp`, the pair you created in **L04**, in the namespace `ml`. Declare each one in
the header and define it in the `.cpp`, alongside the `initRandGen()` already there.

* `randomStartVal()`: function to generate and return a random number in the range `[-1.0, 1.0]`, closed at both ends.
    * **Implementation:**
        * First generate a random number in the range `[0.0, 1.0]` by calling `std::rand()`, which generates a random number in `[0, RAND_MAX]`, and dividing by `RAND_MAX`.
        * One of the operands must be converted to a floating-point number to avoid integer division, e.g. `static_cast<double>(RAND_MAX)`.
        * Then rescale the number to the range `[-1.0, 1.0]` by multiplying by `2.0` and subtracting `1.0`. A draw of exactly `0` then gives exactly `-1.0`, and a draw of exactly `RAND_MAX` gives exactly `1.0`.
    * Should be marked `[[nodiscard]]` and `noexcept`.
    * **Note:** the start values must be able to go negative. If every weight and bias starts positive, the input `{1, 1}` starts out giving the largest weighted sum in every node, the opposite of what XOR asks for, and training must first drive some of the weights negative before the network gets anywhere. Start values symmetric around zero also start `Tanh` layers in the sensitive part of the function rather than in saturation, where the derivative is close to zero.
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
Randomize all bias values and weights in `initParams()`, the interface method `Dense` overrides
(see section 4 of the previous part). It takes no arguments, returns nothing, and should be marked
`override` and `noexcept`:
* Iterate through every node in the layer with a for loop: `for (std::size_t i{}; i < nodeCount(); ++i)`.
* For each node `i`:
    * Assign a random number to its bias value by calling `randomStartVal()`: `myBias[i] = randomStartVal()`.
    * Iterate through the node's weights with a nested for loop: `for (std::size_t j{}; j < weightCount(); ++j)`.
    * For each weight `j`:
        * Assign a random number to the weight by calling `randomStartVal()`: `myWeights[i][j] = randomStartVal()`.

In the constructor, first call `initRandGen()` (from `ml/utils.hpp`) to seed the random number
generator, then call `initParams()` to fill the two containers. A layer therefore starts out
randomized exactly as before; the randomization has simply moved into a method the network can call
again.

**Who calls it, and why the network needs it.** `Shallow::train()` calls `initParams()` on both
its layers before the first epoch (see **L04**), so every call trains a network from scratch rather
than continuing the one the previous call left behind. That matters most when a run ends badly:
training on top of a network that has settled into a bad set of parameters mostly keeps it there,
while a fresh start draws new ones and can converge on the second attempt. Without a method on the
interface, the network couldn't do it at all, since the bias and weights are private to the layer.

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

### 7. Reporting the precision from `train()`
In **L04**, `Shallow::train()` returns `true` whenever its arguments are valid, whether the network
learned anything or not. With two stubs that was all there was to say, since nothing could be
learned. Now that the network actually trains, the caller needs to know how well it did, for
instance to decide whether to train again (see section 8). Change `train()` to return the precision
it reached instead.

**`std::optional`.** `train()` now has two kinds of result: a precision, or no result at all when
training couldn't be carried out. `std::optional<double>`, from the header `<optional>`, holds
exactly that, either a `double` or nothing:
* Returning a `double` from a function declared to return `std::optional<double>` stores it, so
  `return currPrecision;` works as written.
* `return std::nullopt;` returns an empty optional, i.e. no value.
* The caller checks for a value with `has_value()` (or simply `if (precision)`), and reads it with
  `*precision` or `precision.value()`. Only read it after checking: `*precision` on an empty
  optional is undefined behavior, while `value()` throws an exception.

**Why not return `-1.0` on invalid arguments?** The precision is `1.0 - MAE`, and the mean absolute
error has no upper bound. A network whose predictions miss by `2.0` on average has a precision of
exactly `-1.0`, so any negative number could be a real result. An empty optional can't be mistaken
for a precision.

**Declaration:** in `ml/neural_network/shallow.hpp`, change the declaration of `train()` to

```cpp
[[nodiscard]] std::optional<double> train(std::size_t epochCount,
                                          double precisionThreshold = 0.999999) noexcept;
```

* Include `<optional>` in `shallow.hpp`.
* Mark it `[[nodiscard]]`: a caller that ignores the result has no idea whether training worked.
* Update the `@return` line of its doc comment to match.

**Implementation:** in `source/ml/neural_network/shallow.cpp`, update the return type of the
definition, then change each `return`:
* Replace every `return false;` with `return std::nullopt;`: the two argument checks, and the three
  layer calls inside the epoch loop. Each of them means training couldn't be carried out at all.
* At the early stop, return the precision you just measured, `currPrecision`, instead of `true`.
  The early stop line printed there stays as it is.
* After the epoch loop, replace `return true;` with `return precision();`. Measure once more rather
  than returning the last evaluated precision: that value is up to 99 epochs old, and a run of 100
  epochs or fewer has never been evaluated at all.

**Return value:** `std::nullopt` on invalid arguments or a failed layer call. Otherwise the
precision the network reached: at least `precisionThreshold` if training stopped early, and whatever
the final measurement gives if the epochs ran out.

---

### 8. Compiling and testing
Update `main.cpp` to use your new layer:
* Replace the two `ml::dense_layer::Stub` instances with `ml::dense_layer::Dense`, keeping the same
  wiring: the output layer's weight count must still match the hidden layer's node count.
* Make the hidden layer wider than the task strictly needs, 15 nodes, and leave its activation
  function at the default `ActFunc::Relu`. Give the output layer `ActFunc::Tanh`. A ReLU output
  node that dies takes the whole network with it, since every prediction is then `0`. Keep the
  2-bit XOR training data from **L04**.
* Give the training parameters named constants in `trainAndTest()`: an epoch count of `100000`, a
  precision threshold of `0.99`, and an attempt count of `10`.
* Pass the threshold to `train()` explicitly. The default of `0.999999` is practically never
  reached, so every attempt would run all 100000 epochs and still fall short of it.

Then train in a loop, at most once per attempt:
* Call `train()` and keep the returned optional: `const auto precision{network.train(epochCount, precisionThreshold)}`.
* If it holds no value, print `"Training failed!"` and return `false`. The arguments are invalid or
  the network is wired wrong, so another attempt would fail the same way.
* If the precision reaches the threshold, print the predictions after training as before and return
  `true`.
* Otherwise, print the attempt number and the precision it reached, e.g.
  `"Attempt 1 reached precision 0.75, retrying!"`, and go round again.
* If every attempt falls short, print `"Training failed after 10 attempts!"` (using the constant
  rather than a literal) and return `false`.

There's no learning rate to pass: the network sets its own, and stops as soon as it reaches the
precision threshold, so the epoch count is an upper bound rather than a number to tune. Each call to
`train()` resets both layers first (see **L04**), so every attempt starts from newly drawn
parameters rather than the ones the previous attempt got stuck in.

**Don't delete `stub.hpp`.** `Dense` replaces the stub in your *network*, not in your codebase. The
test suite still compiles both, and still uses the stub to test `Shallow`, since a layer that
computes nothing is what makes the network's wiring visible. See section 9.

Compile and test-run the program. Unlike **L04**, where every prediction was the stub's fixed
output, each prediction should now land within a few hundredths of its training target. Your exact
numbers will differ from run to run, because the bias and weights start randomized.

The early stop line from `train()` should appear above the predictions, reporting a precision and
an epoch count well short of the 100000 you asked for. Both vary from run to run: the starting
parameters are random, and so is the training order. A run that stops after a few hundred epochs
and one that needs tens of thousands are both normal; the average is around 2500.

**When an attempt doesn't converge.** Roughly two attempts in a hundred settle into a set of
parameters they can't improve on, typically with one or two of the four predictions well off target
however long they train. That's what the attempt loop is for. A second attempt starts from scratch
and nearly always gets there: in 1000 runs of the program, not one used up all ten attempts. That's
the reset from **L04** earning its keep, and it's what the test suite's convergence test relies on
as well.

**Worth trying:** switch the output layer to `ActFunc::Relu` as well and shrink the hidden layer to
2 nodes. Roughly six attempts in seven then fail to learn XOR at all, however long they train, and
even with ten attempts about one run of the program in five ends with `"Training failed after 10
attempts!"`. A node whose weighted sum is negative for every input has a derivative of exactly zero:
it stops learning and can't recover. Since `randomStartVal()` draws from `[-1.0, 1.0]`, a node can
start out that way, and with only three ReLU nodes in the network, about two networks in three
contain one before training has even begun. That's the dying ReLU problem from **L03**, and it's
why the test suite trains with `Tanh`.

---

### 9. Running the tests
The final test suite is available in [exercises/test](../exercises/test/). It's cumulative: it
carries the **L04** stub tests over unchanged, updates the **L04** network tests for the new return
value of `train()` (see section 7), and adds unit tests for `Dense` and for the helpers in
`ml/utils`.

Once you've carried your code forward into this lecture's exercises directory, build and run it:

```bash
make -C test
```

All 64 test cases should pass. Because `Dense` randomizes its own bias and weights, none of its
output can be predicted from the constructor arguments, so the tests recover the bias from the
layer's output instead. Without an activation function, a vector of zeros is enough: every weight
term drops out and the weighted sum is the bias alone. `Tanh` tests feed the same zeros and invert
the output with `std::atanh()`. ReLU would clamp a negative bias to zero, so the ReLU tests instead
lift each node's weighted sum by a known amount, far above any start value, and subtract it again.
That makes every expected value exact rather than approximate.

`InitParamsDrawsNewValues` is the first test in the course where `initParams()` has anything to do.
The stub tests only check that calling it changes nothing, which is all an empty override can be
asked for; here the values it writes are read straight back out of `weights()` and out of a
zero-input feedforward.

The one to read is `BackpropagateUsesPreActivationDerivative`. It checks the **Note!** in sections
4 and 5 above, and only `ActFunc::Tanh` can catch that mistake: for `Relu` the two values agree,
since `max(0.0, s)` is positive exactly when `s` is.

See the [test suite's README](../exercises/test/README.md) for more information, including how the
convergence threshold was chosen.

---
