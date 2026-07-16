# Appendix B - Exercises
You'll extend the codebase from **L03**, first declaring the neural network class, then
implementing its training method.

---

## The Neural Network Class
You'll extend the codebase from last lecture with an interface and a class for a simple neural network consisting of a hidden layer and an output layer, built on top of the `dense_layer::Interface`/`Stub` pair you already have.

---

### 1. Directory structure
Extend last lecture's directory structure as follows:

```
ml/
├── include/
│   └── ml/
│       ├── dense_layer/
│       │   ├── interface.hpp
│       │   └── stub.hpp
│       ├── neural_network/
│       │   ├── interface.hpp
│       │   └── shallow.hpp
│       └── types.hpp
├── source/
│   ├── neural_network/
│   │   └── shallow.cpp
│   └── main.cpp
└── Makefile
```

Don't forget to add `source/neural_network/shallow.cpp` to your makefile.

---

### 2. Interface for neural networks
In the header file `ml/neural_network/interface.hpp`, add a namespace named `ml::neural_network`. In this namespace, implement an interface named `Interface`:

* **`~Interface()`:** should be set to `default` and marked `virtual` and `noexcept`.
* **`predict(input)`:** pure virtual. `input`: read-only floating-point vector of the input to base the prediction on. Returns a reference to a floating-point vector holding the predicted value. Should be marked `noexcept` (**not** `const`, since the layers' output is updated on every prediction).

---

### 3. The Shallow class - declaration
In the header file `ml/neural_network/shallow.hpp`, add the namespace `ml::neural_network`. Implement a subclass named `Shallow` that inherits `Interface` via public inheritance. The class should be marked `final`.

* **`Shallow()`:** takes `hiddenLayer` and `outputLayer` (the network's hidden layer and output layer, `ml::dense_layer::Interface&`), plus `trainInput` and `trainOutput` (read-only, two-dimensional floating-point vectors holding the training data's inputs and outputs). Should be marked `explicit` and `noexcept`.
* **`~Shallow()`:** should be marked `default`, `noexcept`, and `override`.
* **`predict()`:** overrides the corresponding method in the interface. Should be marked `noexcept` and `override`.
* **`train(epochCount, learningRate = 0.01)`:** trains the network (implemented in full later this lecture). `epochCount`: number of epochs to train (unsigned integer). `learningRate`: learning rate (floating-point number). Returns `true` if training was carried out, `false` otherwise. Should be marked `noexcept`.

Feel free to add more (private) methods as needed.

---

### 4. Removed constructors and operators
Delete the class's default constructor, copy and move constructors, and the corresponding operators.

---

### 5. Private member variables
Add the following private member variables to `Shallow`:
* **`myHiddenLayer`:** reference to the network's hidden layer, obtained via the constructor.
* **`myOutputLayer`:** reference to the network's output layer, obtained via the constructor.
* **`myTrainInput`:** reference to the training data's input, obtained via the constructor.
* **`myTrainOutput`:** reference to the training data's output, obtained via the constructor.
* **`myTrainSetCount`:** constant unsigned integer holding the number of complete training sets (i.e. the smaller of `myTrainInput.size()` and `myTrainOutput.size()`).

---

### 6. Constructor and prediction
Implement the following in `source/neural_network/shallow.cpp`:

**The constructor:**
* Initialize all member variables as described above.

**The method `predict()`:**
* Perform feedforward through the entire network:
    1. Call `myHiddenLayer.feedforward(input)` with the given input.
    2. Call `myOutputLayer.feedforward(myHiddenLayer.output())` with the hidden layer's output as input.
* Return `myOutputLayer.output()` (a reference - no separate storage variable is needed in `Shallow`).

---

### 7. Training method (placeholder)
Implement a temporary version of `train()` in `source/neural_network/shallow.cpp` that simply returns `true`. The full implementation (feedforward, backpropagation, and optimization for each training set and epoch) follows later this lecture.

---

### 8. Compiling and testing
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
Replace the temporary version of `train()` in `source/neural_network/shallow.cpp` with a full implementation:

**Input validation:**
* Print an error message and call `std::terminate()` (from `<exception>`) if `myTrainSetCount == 0`, `epochCount == 0`, or `learningRate <= 0.0 || learningRate >= 1.0`.

**Training:**
* Iterate the desired number of epochs with a for loop: `for (std::size_t epoch{}; epoch < epochCount; ++epoch)`.
* For each epoch, iterate through all training sets: `for (std::size_t x{}; x < myTrainSetCount; ++x)`.
* For each training set `x`, perform the following three steps:
    1. **Feedforward:** call `predict(myTrainInput[x])`. This performs feedforward through both the hidden layer and the output layer.
    2. **Backpropagation:**
        * Compute the error in the output layer: `myOutputLayer.backpropagate(myTrainOutput[x])`.
        * Compute the error in the hidden layer from the output layer's error and weights: `myHiddenLayer.backpropagate(myOutputLayer)`.
    3. **Optimization:**
        * Optimize the hidden layer: `myHiddenLayer.optimize(myTrainInput[x], learningRate)`.
        * Optimize the output layer based on the hidden layer's output: `myOutputLayer.optimize(myHiddenLayer.output(), learningRate)`.

**Return value:** `true` once training is complete.

---

### 2. Compiling and testing
Update `main.cpp` so `train()` is called (with an epoch count and learning rate of your choice) before prediction is performed. Check the return value of `train()`: print an error message and exit the program with an error code if training fails.

Keep using `ml::dense_layer::Stub` for the hidden layer and output layer until a concrete dense layer implementation exists (see **L05**).

Compile and test-run the program. Don't worry if the network doesn't predict correctly; the dense layers are still stubs and therefore don't actually train. What matters is that the code compiles and runs without errors; a concrete implementation of `Dense` is added in **L05**.

---
