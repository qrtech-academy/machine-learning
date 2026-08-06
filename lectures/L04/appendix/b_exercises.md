# Appendix B - Exercises
You'll extend the codebase from **L03**, first declaring the neural network class, then
implementing its training method.

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
├── test/                <- already here, see section 3 of the next part
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
* **`train(epochCount, learningRate = 0.01)`:** trains the network (implemented in full later this lecture). `epochCount`: number of epochs to train (unsigned integer). `learningRate`: learning rate (floating-point number). Returns `true` if training was carried out, `false` otherwise. Should be marked `noexcept`.

The class should also have the following private methods, used to train in a randomized order:
* **`initTrainOrder()`:** fills `myTrainOrder` (see below) with the indices `0, 1, 2 ... N-1`, where
  `N` is the number of complete training sets. Takes that set count as its only argument, returns
  nothing, and should be marked `noexcept`.
* **`randomizeTrainOrder()`:** shuffles the contents of `myTrainOrder` into a random order. For each
  index `i`, pick a random index `r` and swap `myTrainOrder[i]` and `myTrainOrder[r]`. Takes no
  arguments, returns nothing, and should be marked `noexcept`.

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
* Return `myOutputLayer.output()` (a reference - no separate storage variable is needed in `Shallow`).

---

### 8. Training method (placeholder)
Implement a temporary version of `train()` in `source/ml/neural_network/shallow.cpp` that simply returns `true`. The full implementation (feedforward, backpropagation, and optimization for each training set and epoch) follows later this lecture.

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
You'll complete the class `ml::neural_network::Shallow` you just built by implementing a full training method.

---

### 1. Training method
Replace the temporary version of `train()` in `source/ml/neural_network/shallow.cpp` with a full implementation:

**Input validation:**
* Return `false` if `epochCount == 0`, or if `learningRate <= 0.0 || learningRate >= 1.0`.
* The training set count doesn't need checking here. The constructor already guaranteed at least one complete set, so `myTrainOrder` is never empty.
* As in **L01** and **L02**, `train()` reports invalid arguments through its return value. Only a constructor calls `std::terminate()`, because it has no way to return a failure code to the caller.

**Training:**
* Iterate the desired number of epochs with a for loop: `for (std::size_t epoch{}; epoch < epochCount; ++epoch)`.
* At the start of every epoch, call `randomizeTrainOrder()`. As in **L02**, reshuffling each epoch keeps the network from learning anything from the order the training data happens to be stored in.
* For each epoch, iterate through the training sets in the order `myTrainOrder` gives rather than sequentially, e.g. with a range-based for loop over `myTrainOrder`. Note that the loop variable is the index into the training data, not the counter itself.
* For each training set index `x`, perform the following three steps:
    1. **Feedforward:** call `predict(myTrainInput[x])`. This performs feedforward through both the hidden layer and the output layer.
    2. **Backpropagation:**
        * Compute the error in the output layer: `myOutputLayer.backpropagate(myTrainOutput[x])`.
        * Compute the error in the hidden layer from the output layer's error and weights: `myHiddenLayer.backpropagate(myOutputLayer)`.
    3. **Optimization:**
        * Optimize the hidden layer: `myHiddenLayer.optimize(myTrainInput[x], learningRate)`.
        * Optimize the output layer based on the hidden layer's output: `myOutputLayer.optimize(myHiddenLayer.output(), learningRate)`.

Each of the four layer calls above returns `bool` (see **L03**). Return `false` as soon as any of them fails: a dimension mismatch means the network is wired wrong, and training on from there would only produce meaningless numbers.

**Return value:** `false` on invalid arguments or a failed layer call, otherwise `true` once training is complete.

---

### 2. Compiling and testing
Update `main.cpp` so `train()` is called (with an epoch count and learning rate of your choice) before prediction is performed. Check the return value of `train()`: print an error message and exit the program with an error code if training fails.

Keep using `ml::dense_layer::Stub` for the hidden layer and output layer until a concrete dense layer implementation exists (see **L05**).

Compile and test-run the program. Don't worry if the network doesn't predict correctly; the dense layers are still stubs and therefore don't actually train. What matters is that the code compiles and runs without errors; a concrete implementation of `Dense` is added in **L05**.

---

### 3. Running the tests
An updated test suite is available in [exercises/test](./exercises/test/). It's cumulative: it
carries the **L03** stub tests over unchanged and adds component tests for `Shallow` on top, so it
supersedes the L03 suite entirely.

Once you've carried your code forward into this lecture's exercises directory, build and run it:

```bash
make -C test
```

All 26 test cases should pass. The ones worth reading check that a prediction reads the output
layer *live* rather than from a stored copy, and that training performs exactly one feedforward per
training set per epoch. Nothing else can tell a correct training loop from one that runs a single
pass: both line up dimensionally and both return `true`.

See the [test suite's README](./exercises/test/README.md) for more information, including the four
mistakes this suite deliberately can't catch and what it would take to catch them.

---
