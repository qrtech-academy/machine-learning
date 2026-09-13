# Appendix B - Exercises

## Hand-Training a Neural Network - Example 1
The neural network below should be trained to predict a high output Y when an odd number of the inputs X1 and X2 are high, and output Y = 0 otherwise:

![](./images/network1.png)

The first training set: inputs X1X2 = 01, Y = 1.

**Starting parameters:**

```
b1 = 0.2,  b2 = 0.4,  b3 = 0.6
w1 = 0.5,  w2 = 0.6,  w3 = 0.3
w4 = 0.8,  w5 = 0.1,  w6 = 0.9
```

Learning rate: LR = 0.1. Activation function: ReLU for all nodes.

Train the network on the first training set: perform feedforward, backpropagation, and
optimization. Then feed the same input forward again with the new parameters, and compare the
deviation against the one you started with.

The full solution is in [appendix C](./c_solutions.md#example-1).

---

## Hand-Training a Neural Network - Example 2
Consider the neural network below:

![](./images/network2.png)

The network should be trained to predict outputs Y1Y2 that form the inverse of inputs X1X2:

| X1X2 | Y1Y2 |
|:----:|:----:|
|  00  |  11  |
|  01  |  10  |
|  10  |  01  |
|  11  |  00  |

First training set: X1X2 = 10, Y1Y2 = 01.

**Starting parameters - hidden layer:**

```
b1 = 0.1,  b2 = 0.1,  b3 = 0.7
w1 = 0.2,  w2 = 0.9,  w3 = 0.3
w4 = 0.8,  w5 = 0.5,  w6 = 0.4
```

**Starting parameters - output layer:**

```
b4 = 0.1,  b5 = 0.6
w7  = 0.1,  w8  = 0.1,  w9  = 1.0
w10 = 0.3,  w11 = 0.0,  w12 = 0.6
```

Learning rate: LR = 0.1. Activation function: ReLU for all nodes.

Train the network on the first training set: perform feedforward, backpropagation, and
optimization. Then feed the same input forward again with the new parameters, and compare the
deviations against the ones you started with.

The full solution is in [appendix C](./c_solutions.md#example-2).

---

## Hand-Training a Neural Network - Example 3
Consider the neural network below:

![](./images/network3.png)

The network should be trained to predict a high output Y when an odd number of the inputs X1 and X2 are high, and output Y = 0 otherwise.

Training has already been carried out for one epoch, and the parameters have the following starting values:

**Starting parameters:**

```
b1 = 0.1,   b2 = -0.4,  b3 = -0.2
w1 = 0.5,   w2 = 0.4,   w3 = -0.2
w4 = 0.1,   w5 = 0.7,   w6 = 0.8
```

Training is now to be carried out with the training set X1X2 = 11, Y = 0.

Learning rate: LR = 0.1. Activation function: ReLU for all nodes.

Perform feedforward, backpropagation, and optimization. Then compute the output again. Did the error decrease? If not, what could you have changed to obtain a smaller error?

The full solution is in [appendix C](./c_solutions.md#example-3).

---

## Dense Layer Interface and Stub
You'll start a new `ml` codebase (or extend your existing one, if you'd like to keep the linear
regression code alongside it) with an interface for a dense layer and a placeholder implementation
of it.

---

### 1. Directory structure
Set up the following structure in this lecture's [exercises](../exercises/) directory:

```
exercises/
├── include/
│   └── ml/
│       ├── dense_layer/
│       │   ├── interface.hpp
│       │   └── stub.hpp
│       └── types.hpp
├── source/
│   └── main.cpp
├── test/                <- already here, see section 4
└── Makefile
```

You'll extend this structure further next lecture.

---

### 2. Dense layer interface and stub class
The hidden layer and output layer of a future neural network will be represented by the interface
`ml::dense_layer::Interface`. A concrete implementation isn't created until **L05**. Until then,
you'll implement a simple stub class `ml::dense_layer::Stub` as a placeholder.

**The interface (`ml/dense_layer/interface.hpp`):**
In the namespace `ml::dense_layer`, implement an interface named `Interface`. All methods
(except the destructor) should be declared pure virtual (`= 0`).

* **`~Interface()`:** should be set to `default` and marked `virtual` and `noexcept`.

Getters, all `const`, `noexcept`, and `[[nodiscard]]`:

| Method | Returns |
|---|---|
| `nodeCount()` | Number of nodes in the layer (`std::size_t`). |
| `weightCount()` | Number of weights per node (`std::size_t`). |
| `output()` | Reference to the layer's output (read-only floating-point vector). |
| `error()` | Reference to the layer's error (read-only floating-point vector). |
| `weights()` | Reference to the layer's weights (read-only, two-dimensional floating-point vector). |

Computation methods, all `noexcept` and returning `bool`. Each returns `false` on invalid input
(wrong dimensions or an invalid learning rate) and `true` otherwise, so the caller can decide what
to do about the failure. As in **L01** and **L02**, `std::terminate()` is reserved for the
constructor, which has no way to return a failure code to the caller:
* **`feedforward(input)`:** performs feedforward.
    * `input`: read-only floating-point vector of input data.
* **`backpropagate(reference)`** (output layer): computes error from reference values.
    * `reference`: read-only floating-point vector of reference values.
* **`backpropagate(nextLayer)`** (hidden layer): computes error from the next layer.
    * `nextLayer`: reference to the next layer (`const Interface&`).
* **`optimize(input, learningRate)`:** updates bias and weights.
    * `input`: read-only floating-point vector.
    * `learningRate`: floating-point number.

One method more, `noexcept` and returning nothing:
* **`initParams()`:** resets the layer's trainable parameters, i.e. its bias values and weights, to
  their initial values.
    * Takes no arguments and returns nothing.
    * Pure virtual like the rest, so every layer states what its own reset means, even when the
      answer is "nothing". `Stub` has no trainable parameters to draw again, so its override is
      empty; `Dense` overrides it in **L05** by drawing a new random bias and weight for every
      node, and its constructor calls it too, so the randomization is written once and used from
      both places.

**Who calls it.** The network you write in **L04** does, on both of its layers, before its first
training epoch. That's what makes a second call to `train()` train a new network rather than
continue the one the previous call left behind, which matters when a run ends up badly trained: a
network that has settled into a bad set of parameters mostly stays there, while a fresh start draws
new ones and can converge on the next attempt. The bias and weights are private to the layer, so
without a method on the interface the network couldn't reset them at all.

**The stub class (`ml/dense_layer/stub.hpp`):**
In the namespace `ml::dense_layer`, implement a subclass named `Stub` that inherits `Interface`
via public inheritance. The class should be marked `final`. The stub doesn't perform any real
computation; it exists solely so other code can be compiled, test-run, and unit tested against a
real `dense_layer::Interface` before a concrete `Dense` implementation exists (see **L05**). The
network you build in **L04** is tested entirely against this stub, so it's worth getting right.

* **`Stub()`:**
    * The class's only implemented constructor.
    * Should take the following arguments:
        * `nodeCount`: number of nodes in the layer (unsigned integer).
        * `weightCount`: number of weights per node (unsigned integer).
        * `outputValue`: value every output element is set to (floating-point number).
          Default value: `0.5`.
    * Initializes every element of the output vector to `outputValue`, and the error and weight
      vectors to zeros.
    * Should print an error message and call `std::terminate()` if either `nodeCount` or
      `weightCount` is zero:
        * As in **L01** and **L02**, the constructor is the one place that terminates, since it
          has no way to return a failure code.
        * It also guarantees at least one node and one weight, which is what makes it safe for
          `weightCount()` to read the width of the first row of the weight matrix.
    * `outputValue` is an argument rather than a hard-coded constant so that two stubs can be told
      apart. A network whose layers all report the same output can't show whether a prediction
      came from the output layer or from the hidden one; giving each layer its own value makes
      that visible.
    * Should be marked `explicit` and `noexcept`.
* **`~Stub()`:**
    * Destructor overriding the interface's destructor.
    * Should be marked `default`, `noexcept`, and `override`.
* **Getters** (`nodeCount()`, `weightCount()`, `output()`, `error()`, `weights()`):
    * Override the corresponding methods in the interface.
    * Should be marked `override`, retaining the interface's `[[nodiscard]]`, `const`, and
      `noexcept`.
    * Attributes aren't inherited, so the override needs its own `[[nodiscard]]`. Without it, a
      discarded call such as `stub.nodeCount();` compiles silently whenever it's made on a `Stub`
      rather than through an `Interface&`.
* **`feedforward()`**, both overloads of **`backpropagate()`**, and **`optimize()`**:
    * Perform range checks only:
        * Return `false` when the dimensions don't match, or, for `optimize()`, when the learning
          rate lies outside `(0.0, 1.0)`.
        * Return `true` otherwise.
    * Deliberately compute nothing:
        * The output stays at `outputValue` regardless of what's fed in.
        * The error stays at zero.
    * Should be marked `override` and `noexcept`.
* **`setOutput()`:**
    * Sets every element of the output vector to the given value.
    * Should take a single argument:
        * `outputValue`: value to set every output element to (floating-point number).
    * Returns nothing, and should be marked `noexcept`.
    * **Not** part of `Interface`. It exists on the stub alone.
    * It's what lets a test drive the output of a whole network in **L04**:
        * The network stores its layers by reference, so calling `setOutput()` on the layer a
          network was built with changes what that network predicts.
        * That reveals whether the network reads its output layer live or kept a stale copy of
          it. Keeping a stale copy is exactly what the note in **L04** about not needing a
          separate storage variable is there to prevent.
* **`feedforwardCount()`** and **`clearFeedforwardCount()`:**
    * `feedforwardCount()` returns the number of times `feedforward()` has been called on this
      layer as a `std::size_t`, counting **every** call, not just the ones that passed the range
      check. Return `std::size_t` rather than `int`: the tests compare the tally against
      `std::size_t` values, and a signed return trips `-Wsign-compare`, which the suite builds
      with `-Werror`.
        * Increment the counter (`myFeedforwardCount`) at the very top of `feedforward()`, before
          the input size is checked, so a rejected call raises it just as an accepted one does.
        * The tally records how often the layer was *asked* to feed forward, not how often it
          agreed to.
        * Should be marked `[[nodiscard]]`, `const`, and `noexcept`.
    * `clearFeedforwardCount()` resets that tally to zero.
        * Returns nothing, and should be marked `noexcept`.
    * Neither is part of `Interface`.
    * `train()` in **L04** performs one feedforward per training set per epoch, so this is what
      lets a test pin down the training loop:
        * Nothing else can: a loop that runs a single pass instead of every epoch still lines up
          dimensionally and still returns `true`, so without a count it's indistinguishable from
          a correct one.

* **`initParams()`:**
    * Overrides the corresponding method in the interface, and should be marked `override` and
      `noexcept`.
    * Its body is empty: the stub has no bias, and its weights never move from the zeros it was
      constructed with, so there's nothing to draw again.
    * An empty body is the implementation here, not a placeholder. It's also the only one the
      network in **L04** needs: that network calls `initParams()` on both of its layers before
      every training run, and against a stub the call has to compile, run, and change nothing.

For this class, the default constructor and the copy and move constructors (and corresponding
operators) should be deleted.

Add private member variables for the output, the error, the weights, and the feedforward count.

The number of nodes and the number of weights per node don't need member variables of their own.
Both are readable from the vectors you already have: `nodeCount()` is the size of the output
vector, and `weightCount()` is the width of the weight matrix.

The bias doesn't need one at all. It isn't part of `Interface`, and this stub never optimizes
anything, so there'd be nothing to put in it and no way to read it back out.

**Note!** Leaving it out is a stub-only simplification. The real `Dense` layer you write in **L05**
does need a bias vector: it's a trainable parameter, adjusted alongside the weights on every call
to `optimize()`.

---

### 3. A quick compile check
There's no network to run yet, that's next lecture, once `neural_network::Shallow` exists to make
use of this interface. For now, just confirm your `Interface`/`Stub` pair compiles: in `main.cpp`,
create a `ml::dense_layer::Stub` with a few nodes and weights, and print its `nodeCount()`,
`weightCount()`, and `output()` to the terminal. You should see `nodeCount()`/`weightCount()`
match what you constructed it with, and every value in `output()` equal to the default
`outputValue` of `0.5`.

Create a second stub with an explicit `outputValue`, and check that its `output()` reports that
value instead. Also feed both stubs a vector of the wrong size and check that `feedforward()`
returns `false` rather than terminating.

---

### 4. Running the tests
A test suite for the stub is available in [exercises/test](../exercises/test/). It's already in
place next to the code you just wrote. Build and run it from the exercises directory:

```bash
make -C test
```

The test framework is checked out as a submodule at the root of this repository. If `libs/test` is
empty, fetch it once with `git submodule update --init`.

All 14 test cases should pass. **L03 starts a new `ml` codebase**, so this suite replaces nothing
and stands on its own.

Testing a class that computes nothing may look pointless, but the range checks are the only real
logic the stub has, and the network you write in **L04** is tested entirely against them. A stub
whose `feedforward()` accepted an input of any size would let every L04 test pass without proving
anything.

See the [test suite's README](../exercises/test/README.md) for more information, including which of
these tests carry over unchanged to your real `Dense` layer in **L05**.

---
