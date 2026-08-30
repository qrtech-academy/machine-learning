# Appendix A - Theory
This appendix covers the neural network class you'll build this lecture, and how its training loop
maps onto the theory from L03.

---

## 1. Neural Network Architecture

### Overview
Last lecture you built a `dense_layer::Interface` and a placeholder implementation of it,
`dense_layer::Stub`. This lecture adds the piece that actually uses them: a small neural network
class, `neural_network::Shallow`, completing the structure:

```
ml::dense_layer::Interface   (built last lecture)
└── ml::dense_layer::Stub    (built last lecture)

ml::neural_network::Interface   (Interface for neural networks)
└── ml::neural_network::Shallow (Neural network with a single hidden layer)
        ├── ml::dense_layer::Interface& myHiddenLayer
        └── ml::dense_layer::Interface& myOutputLayer
```

`ml::dense_layer::Stub` continues to serve as a placeholder throughout this lecture. Once a
concrete implementation is created in **L05**, the stub is replaced without needing to change any
of the code you write today.

---

### The Network's Structure
`Shallow` holds references to two dense layers and connects them during prediction:

```
input → myHiddenLayer.feedforward(input)
               ↓
       myHiddenLayer.output()
               ↓
       myOutputLayer.feedforward(...)
               ↓
       myOutputLayer.output()  →  prediction
```

`Shallow` doesn't own the layers; it receives them as references via the constructor. This makes it easy to swap out layers without changing the network class, which is exactly what happens in **L05** when the stub is replaced with a real implementation.

---

## 2. Training Loop

### Overview
You'll implement the method `train()` in `ml::neural_network::Shallow`. The method carries out the complete training process, repeated for every training set and epoch:
* Feedforward:
  * Computes outputs for every node in the network.
  * The output from the last layer forms the network's prediction.
  * Must be computed forward (from input to output).
* Backpropagate:
  * Computes gradients (simplified error) for every node in the network.
  * Must be computed backward (from output to input).
* Optimization:
  * Adjusts the trainable parameters (weights and biases) based on the computed error, together with the learning rate the network currently holds.
  * Computation can happen in either direction.

---

### Structure of the Training Loop

The training loop can be summarized as follows:

```
Reset both layers' trainable parameters

For each epoch:
    For each training set x:
        1. Feedforward: compute predictions
        2. Backpropagation: compute error
        3. Optimization: update weights and biases

    Every hundredth epoch:
        4. Measure the precision, and stop early or set a new learning rate from it
```

The order matters:
* Backpropagation depends on the result of feedforward.
* Optimization depends on the result of backpropagation.

The reset at the top is what makes a second call to `train()` train a new network rather than
continue the old one, and step 4 is what frees the caller from having to pick a learning rate. Both
are covered in [appendix B](./b_exercises.md).

---

### Step 1 - Feedforward
Call `predict(myTrainInput[x])`. This performs feedforward through both layers and returns the output layer's output.

```
predict(input)
  → myHiddenLayer.feedforward(input)
  → myOutputLayer.feedforward(myHiddenLayer.output())
```

---

### Step 2 - Backpropagation
Compute the error backward: always start with the output layer.

**The output layer** compares its output against the reference value:
```
myOutputLayer.backpropagate(myTrainOutput[x])
```

**The hidden layer** computes its error from the output layer's error and weights:
```
myHiddenLayer.backpropagate(myOutputLayer)
```

The order is crucial: the hidden layer's calculation depends on the output layer having already computed its error.

---

### Step 3 - Optimization
Update the weights and biases in both layers using the errors computed in step 2.

```
myHiddenLayer.optimize(myTrainInput[x], learningRate)
myOutputLayer.optimize(myHiddenLayer.output(), learningRate)
```

The hidden layer's input is the original input. The output layer's input is the hidden layer's output (the one computed during feedforward). `learningRate` is the network's own, a local variable in `train()` rather than an argument, revised in step 4 below.

---

### Step 4 - Evaluation
Once every hundredth epoch, measure how close the network is to its training data and act on the
number:

```
precision = 1.0 - MAE (over every value of every training set)

precision >= precisionThreshold  ->  stop training, the network is done
otherwise                        ->  updateLearningRate(): raise the rate when progress has
                                     stalled, lower it when the network is overshooting
```

The learning rate is therefore never supplied by the caller. It starts at `0.1` and is revised from
one evaluation to the next, exactly as in `ml::lin_reg::Adaptive` from **L02**, which evaluates
every tenth epoch instead: a linear regression model converges in tens of epochs, a network in
thousands.

---

### Connection to L03
These three steps map directly onto the theory from **L03** (see [appendix A](../../L03/appendix/a_theory.md)):

| Step | Theory (L03) | Code |
|---|---|---|
| Feedforward | $y = \sigma(b + \sum w_i x_i)$ | `feedforward(input)` |
| Backpropagation | $\Delta e = \delta \cdot y'$ | `backpropagate(...)` |
| Optimization | $w = w + \Delta e \cdot L \cdot x$ | `optimize(input, learningRate)` |

---

---

### Input Validation
Check that training is possible before the training loop starts. Print an error message and return `false` immediately if any of the following conditions hold:

| Condition | Explanation |
|---|---|
| `myTrainSetCount == 0` | Training can't be carried out without training data. |
| `epochCount == 0` | Training must run for at least one epoch. |
| `precisionThreshold <= 0.0 \|\| precisionThreshold >= 1.0` | Invalid threshold: must be in the range `(0.0, 1.0)`. A threshold of `1.0` or more can never be reached; `0.0` or less is reached by any network at all. |

There's no learning rate among them: the network sets its own, and the layers check the rate they're
handed on every call to `optimize()`.

---
