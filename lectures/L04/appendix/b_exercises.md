# Appendix B - Exercises
You'll extend the codebase from **L03**, first declaring the neural network class, then
implementing its training method. The network trains with an adaptive learning rate, following the
same rule `ml::lin_reg::Adaptive` uses in **L02**, so no learning rate has to be chosen anywhere in
the program.

---

## The Neural Network Class
You'll extend the codebase from last lecture with an interface and a class for a simple neural network consisting of a hidden layer and an output layer, built on top of the `dense_layer::Interface`/`Stub` pair you already have.

---

### 1. Directory structure
Carry last lecture's code forward into this lecture's [exercises](./exercises/) directory, then
extend the structure as follows:

```
exercises/
├── include/
│   └── ml/
│       ├── dense_layer/
│       │   ├── interface.hpp
│       │   └── stub.hpp
│       ├── neural_network/
│       │   ├── interface.hpp
│       │   └── shallow.hpp
│       ├── types.hpp
│       └── utils.hpp
├── source/
│   ├── ml/
│   │   ├── neural_network/
│   │   │   └── shallow.cpp
│   │   └── utils.cpp
│   └── main.cpp
├── test/                <- already here, see section 5 of the next part
└── Makefile
```

Note that the sources now sit under `source/ml/`, mirroring `include/ml/`. Don't forget to add
both `source/ml/neural_network/shallow.cpp` and `source/ml/utils.cpp` to your makefile.

---

### 2. Utility functions
Some helpers don't belong to any one class. The first of them is needed this lecture, and **L05**
adds three more to the same pair of files, so give them a home now.

**The header (`ml/utils.hpp`):**
In the namespace `ml`, declare a single function:
* **`initRandGen()`:**
    * Initializes the random number generator, once per program run.
    * Takes no arguments and returns nothing.
    * Should be marked `noexcept`.

**The implementation (`source/ml/utils.cpp`):**
Define `initRandGen()` in the same namespace:
* Add a static local variable named `initialized` with an initial value of `false`. Being static,
  it keeps its value between calls, which is what makes the seeding happen only once.
* If `initialized` is already `true`, return early.
* Seed the generator with the current time by calling `std::srand(std::time(nullptr))`. Include
  `<cstdlib>` and `<ctime>` for these.
* Set `initialized` to `true` so the next call does nothing.

**Why a shared file rather than an anonymous namespace?** In **L02** the same function lived in an
anonymous namespace inside `fixed.cpp`, which was fine when one class needed it. It doesn't scale:
an anonymous namespace gives every `.cpp` its own private copy, each with its own `initialized`
flag, so the generator gets reseeded once per file that asks for it. Seeding must happen once per
*program*. A single definition in `utils.cpp`, shared through `utils.hpp`, gives exactly that.

---

### 3. Interface for neural networks
In the header file `ml/neural_network/interface.hpp`, add a namespace named `ml::neural_network`. In this namespace, implement an interface named `Interface`:
* **`~Interface()`:** should be set to `default` and marked `virtual` and `noexcept`.
* **`predict(input)`:** pure virtual. `input`: read-only floating-point vector of the input to base the prediction on. Returns a reference to a floating-point vector holding the predicted value. Should be marked `noexcept` (**not** `const`, since the layers' output is updated on every prediction).

---

### 4. The Shallow class - declaration
In the header file `ml/neural_network/shallow.hpp`, add the namespace `ml::neural_network`. Implement a subclass named `Shallow` that inherits `Interface` via public inheritance. The class should be marked `final`.

* **`Shallow()`:** takes `hiddenLayer` and `outputLayer` (the network's hidden layer and output layer, `ml::dense_layer::Interface&`), plus `trainInput` and `trainOutput` (read-only, two-dimensional floating-point vectors holding the training data's inputs and outputs). Should be marked `explicit` and `noexcept`.
* **`~Shallow()`:** should be marked `default`, `noexcept`, and `override`.
* **`predict()`:** overrides the corresponding method in the interface. Should be marked `noexcept` and `override`.
* **`train(epochCount, precisionThreshold = 0.999999)`:** trains the network (implemented in full later this lecture). `epochCount`: number of epochs to train (unsigned integer). `precisionThreshold`: precision at which training stops early (floating-point number). Default value: `0.999999` (99.9999 %). Returns `true` if training was carried out, `false` otherwise. Should be marked `noexcept`.

There's no `learningRate` argument. The network picks its own rate while training and revises it as
it goes, exactly as `ml::lin_reg::Adaptive` does in **L02**, so there's nothing for the caller to
guess at. The rule itself is written in the second part of this appendix.

The class should also have the following private methods, used to train in a randomized order:
* **`initTrainOrder()`:** fills `myTrainOrder` (see below) with the indices `0, 1, 2 ... N-1`, where
  `N` is the number of complete training sets. Takes that set count as its only argument, returns
  nothing, and should be marked `noexcept`.
* **`randomizeTrainOrder()`:** shuffles the contents of `myTrainOrder` into a random order. For each
  index `i`, pick a random index `r` and swap `myTrainOrder[i]` and `myTrainOrder[r]`. Takes no
  arguments, returns nothing, and should be marked `noexcept`.

One more private method is needed by the training method, implemented in the second part of this
appendix:
* **`precision()`:** computes the network's precision over the whole training data. Takes no
  arguments and returns a floating-point number. Should be marked `[[nodiscard]]` and `noexcept`,
  but **not** `const`, unlike its **L02** namesake: it calls `predict()`, which feeds both layers
  forward and therefore changes their output.

Feel free to add more (private) methods as needed.

---

### 5. Removed constructors and operators
Delete the class's default constructor, copy and move constructors, and the corresponding operators.

---

### 6. Private member variables
Add the following private member variables to `Shallow`:
* **`myHiddenLayer`:** reference to the network's hidden layer, obtained via the constructor.
* **`myOutputLayer`:** reference to the network's output layer, obtained via the constructor.
* **`myTrainInput`:** reference to the training data's input, obtained via the constructor.
* **`myTrainOutput`:** reference to the training data's output, obtained via the constructor.
* **`myTrainOrder`:** the indices of the training sets as unsigned integers (`ml::MatrixU32`), in the order they'll be trained in. Initialized by the constructor and reshuffled once per epoch.

No member is needed for the learning rate either. It's a local variable inside `train()`, so every
call starts from the same initial rate rather than inheriting whatever the previous call ended on.

No separate member is needed for the training set count. `myTrainOrder` holds one index per training set, so `myTrainOrder.size()` is that count, exactly as `myTrainOrder` replaced `mySetCount` in **L02**.

---

### 7. Constructor and prediction
Implement the following in `source/ml/neural_network/shallow.cpp`:

**The constructor:**
* Initialize all member variables as described above.
* Print an error message and call `std::terminate()` (from `<exception>`) if the output layer's weight count doesn't match the hidden layer's node count. The two layers can never be connected in that case, and the constructor has no way to return a failure code to the caller.
* Compute the number of complete training sets, i.e. the smaller of `trainInput.size()` and `trainOutput.size()`. Print an error message and call `std::terminate()` if it's 0, as in **L01** and **L02**: a network without training data can never be trained.
* Fill `myTrainOrder` by calling `initTrainOrder()` with that set count.
* Call `initRandGen()` (from `ml/utils.hpp`, see section 2), so the generator is seeded before the first shuffle. Doing it here rather than inside `randomizeTrainOrder()` keeps the one-time setup out of a method called once per epoch.

**The method `predict()`:**
* Perform feedforward through the entire network:
    1. Call `myHiddenLayer.feedforward(input)` with the given input.
    2. Call `myOutputLayer.feedforward(myHiddenLayer.output())` with the hidden layer's output as input.
* Return `myOutputLayer.output()` (a reference; no separate storage variable is needed in `Shallow`).

---

### 8. Training method (placeholder)
Implement a temporary version of `train()` in `source/ml/neural_network/shallow.cpp` that simply returns `true`. The full implementation (feedforward, backpropagation, and optimization for each training set and epoch, plus the precision check that sets the learning rate) follows later this lecture.

Give `precision()` a temporary body too, returning `0.0`, so the header and the source file stay in step until section 1 of the next part fills it in.

---

### 9. Compiling and testing
Write a `main` function in `main.cpp` that:
* Creates one `ml::dense_layer::Stub` instance for the hidden layer and one for the output layer (e.g. 3 nodes/2 weights per node, and 1 node/3 weights per node respectively - the number of weights in the output layer should match the number of nodes in the hidden layer).
* Creates an `ml::neural_network::Shallow` instance from these two layers and training data of your choice (e.g. a 2-bit XOR pattern).
* Performs a prediction for each training set's input, and prints the input and predicted output in the terminal.

Compile and test-run the program. You should get the following output (the dense layers are still stubs, so the prediction is always 0.5):

```
--------------------------------------------------------------------------------
Input: 0 0, predicted output: 0.5
Input: 0 1, predicted output: 0.5
Input: 1 0, predicted output: 0.5
Input: 1 1, predicted output: 0.5
--------------------------------------------------------------------------------
```

---

## Training the Neural Network
You'll complete the class `ml::neural_network::Shallow` you just built by implementing a full
training method: first the precision it measures itself by, then the rule that turns that
measurement into a learning rate, and finally the training loop that uses both.

---

### 1. Precision calculation
Implement the private method `precision()` in `source/ml/neural_network/shallow.cpp`. It measures
how close the network currently is to its training data, the same mean absolute error (MAE) the
models in **L02** measure, computed over vectors rather than single values:
* Declare an error sum and a value count, both starting at zero.
* For each training set index `x` in `myTrainOrder`:
    * Call `predict(myTrainInput[x])` and keep the returned reference.
    * For every output node `i`, add `std::abs(myTrainOutput[x][i] - prediction[i])` to the error
      sum and raise the value count by one.
* Compute the MAE by dividing the error sum by the value count.
* Return `1.0 - MAE`.

Include `<cmath>` in `shallow.cpp` for `std::abs`.

**Why the value count rather than the set count?** **L02** divides by the number of training sets,
because a linear regression model produces exactly one value per set. Here a set holds one value
per output node, so dividing by the set count alone would report an error that grows with the width
of the output layer. Dividing by the number of values added gives the average error *per value*,
which is what `1.0 - MAE` is meant to be read against, and the two are identical for the
single-output network you're building.

**Why indexing `myTrainOutput[x][i]` per output node is safe.** `train()` only calls `precision()`
at the end of a complete epoch, and every epoch has already passed each training output through
`myOutputLayer.backpropagate()`, which returns `false` when the row's length doesn't match the
layer's node count. A row of the wrong length therefore ends training before `precision()` is ever
reached.

---

### 2. The learning rate rule
The learning rate is the one training parameter with no sensible default, so the network sets its
own, using the precision it just computed. Add a function named `updateLearningRate()` in an
anonymous namespace in `shallow.cpp`, exactly as **L02** does in `adaptive.cpp`:

```cpp
void updateLearningRate(double& learningRate, double& prevPrecision,
                        double currentPrecision) noexcept;
```

It takes the current learning rate and the previous precision by reference, since it updates both,
and the precision just measured by value. Give each of the four values it needs a named constant: a
maximum learning rate of `0.25`, a minimum of `0.01`, a smallest expected improvement of `0.1`, and
a step of `0.05`.

The rule works on the difference between the two precisions, i.e. how much the network improved
since the last evaluation:
* **Improved by at least the expected minimum:** leave the learning rate alone. It's working.
* **Improved, but by less than that:** raise the learning rate by the step, and clamp it to the maximum. Progress has stalled, so take bigger steps.
* **Didn't improve at all:** lower the learning rate by the step, and clamp it to the minimum. The network is overshooting, so take smaller steps.

Store the precision just measured in `prevPrecision` before returning, so the next call has
something to compare against.

**The minimum is the one value that differs from L02**, which uses `1e-6`. The purpose the minimum
is given there is that "a model that keeps missing can still creep towards the line rather than
stopping dead", and in a network `1e-6` doesn't deliver it: the parameter updates it allows are far
too small to measure, so the precision drifts *down* by a fraction of a millionth from one
evaluation to the next, the rule reads that as another failure to improve, and the rate stays at
the floor for the rest of the run. Two unlucky evaluations in a row are enough to reach it from the
initial rate, and the network never learns anything afterwards. Measured on the 2-bit XOR pattern,
once the real `Dense` layer of **L05** has replaced the stub, a floor of `1e-6` kills roughly one
run in six that way. With `0.01` the rate recovers on its own, since a network that is still moving
improves sooner or later, and the rule raises the rate again the moment it does.

---

### 3. Training method
Replace the temporary version of `train()` in `source/ml/neural_network/shallow.cpp` with a full implementation:

**Input validation:**
* Return `false` if `epochCount == 0`, or if `precisionThreshold` falls outside the range `(0.0, 1.0)`:
    * A threshold of `1.0` or more can never be reached, since the precision is `1.0 - MAE` and the mean absolute error can't be negative. Training would always run the full epoch count.
    * A threshold of `0.0` or less accepts a network whose mean absolute error is `1.0` or worse, which defeats the purpose of the check.
* There's no learning rate to validate, since the caller no longer supplies one. The layers still check the rate they're handed on every call to `optimize()`, and the rule above keeps it between `0.01` and `0.25`, well inside the `(0.0, 1.0)` the layers demand.
* The training set count doesn't need checking here. The constructor already guaranteed at least one complete set, so `myTrainOrder` is never empty.
* As in **L01** and **L02**, `train()` reports invalid arguments through its return value. Only a constructor calls `std::terminate()`, because it has no way to return a failure code to the caller.

**Starting from scratch:**
* Once the arguments are known to be valid, and before the first epoch, reset both layers by calling `myHiddenLayer.initParams()` and `myOutputLayer.initParams()`. The method is part of `ml::dense_layer::Interface` from **L03**, so the network can call it without knowing which kind of layer it holds: `Stub` implements it as an empty method, and `Dense` implements it in **L05** by drawing new values.
* Training therefore always starts from freshly initialized parameters. Calling `train()` a second time trains a new network rather than continuing the old one, which is what makes retraining a network that ended up badly trained worth doing at all: with the parameters carried over, a second call would resume from the same bad state it just failed in.
* This is a deliberate departure from `ml::lin_reg::Fixed` in **L01**, whose `train()` continues from the current parameters so that a model can be trained in stages. That only makes sense while the caller sets the learning rate. Here the schedule restarts from the initial rate on every call, so the rate the network picks would no longer match the parameters it's applied to.
* Do the reset **after** the argument checks, so a rejected call leaves a trained network exactly as it was.

**Training:**
* Give the evaluation interval (`100`) and the initial learning rate (`0.1`) named constants.
* Declare the learning rate and the previous precision as local variables, initialized to the initial learning rate and `0.0`. Being local means every call starts from the initial rate again, rather than inheriting whatever the previous call ended on.
* Iterate the desired number of epochs with a for loop: `for (std::size_t epoch{}; epoch < epochCount; ++epoch)`.
* At the start of every epoch, call `randomizeTrainOrder()`. As in **L02**, reshuffling each epoch keeps the network from learning anything from the order the training data happens to be stored in.
* For each epoch, iterate through the training sets in the order `myTrainOrder` gives rather than sequentially, e.g. with a range-based for loop over `myTrainOrder`. Note that the loop variable is the index into the training data, not the counter itself.
* For each training set index `x`, perform the following three steps:
    1. **Feedforward:** call `predict(myTrainInput[x])`. This performs feedforward through both the hidden layer and the output layer.
    2. **Backpropagation:**
        * Compute the error in the output layer: `myOutputLayer.backpropagate(myTrainOutput[x])`.
        * Compute the error in the hidden layer from the output layer's error and weights: `myHiddenLayer.backpropagate(myOutputLayer)`.
    3. **Optimization:**
        * Optimize the hidden layer: `myHiddenLayer.optimize(myTrainInput[x], learningRate)`, passing the current learning rate.
        * Optimize the output layer based on the hidden layer's output: `myOutputLayer.optimize(myHiddenLayer.output(), learningRate)`.

Each of the four layer calls above returns `bool` (see **L03**). Return `false` as soon as any of them fails: a dimension mismatch means the network is wired wrong, and training on from there would only produce meaningless numbers.

**Evaluating the training progress:**
* Every hundredth epoch, except the first, compute the precision once and keep it in a local variable:
    * Stop training and return `true` if it reaches or exceeds `precisionThreshold`, printing the achieved precision and the number of epochs it took. Report the number of epochs completed, not the loop index, so the first epoch reads as 1 rather than 0.
    * Otherwise, pass it to `updateLearningRate()`.
* **L02** evaluates every tenth epoch, since a linear regression model reaches its threshold in tens of epochs. A network needs thousands, so a tenth-epoch evaluation would scan the whole training data hundreds of times for a picture that has barely changed. Every hundredth keeps that overhead down without letting the learning rate go unexamined for long.
* Note that the precision is computed once per evaluation and then used twice. Calling `precision()` a second time inside `updateLearningRate()` would feed every training set through the network again for a value that's already in hand.

**Return value:** `false` on invalid arguments or a failed layer call, otherwise `true` once the threshold is reached or the epochs run out.

---

### 4. Compiling and testing
Update `main.cpp` so `train()` is called (with an epoch count of your choice, and the default precision threshold) before prediction is performed. Check the return value of `train()`: print an error message and exit the program with an error code if training fails.

Keep using `ml::dense_layer::Stub` for the hidden layer and output layer until a concrete dense layer implementation exists (see **L05**).

Compile and test-run the program. Don't worry if the network doesn't predict correctly; the dense layers are still stubs and therefore don't actually train. What matters is that the code compiles and runs without errors; a concrete implementation of `Dense` is added in **L05**.

The early-stop line stays absent for the same reason: the stub reports the same output whatever it's fed, so the precision sits at whatever that fixed output happens to score and never moves. Every run trains for the full epoch count. Your `train()` isn't wrong; there's simply nothing there to learn until **L05**.

---

### 5. Running the tests
An updated test suite is available in [exercises/test](./exercises/test/). It's cumulative: it
carries the **L03** stub tests over unchanged and adds component tests for `Shallow` on top, so it
supersedes the L03 suite entirely.

Once you've carried your code forward into this lecture's exercises directory, build and run it:

```bash
make -C test
```

All 27 test cases should pass. The ones worth reading check that a prediction reads the output
layer *live* rather than from a stored copy, that training performs exactly one feedforward per
training set per epoch, and that the precision is evaluated on every hundredth epoch and no other.
Nothing else can tell a correct training loop from one that runs a single pass: both line up
dimensionally and both return `true`.

See the [test suite's README](./exercises/test/README.md) for more information, including the
mistakes this suite deliberately can't catch and what it would take to catch them.

---
