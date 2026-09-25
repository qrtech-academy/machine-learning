# L04 Exercise Solutions
Solutions and notes for the exercises in [appendix B](../appendix/b_exercises.md). This directory
holds the **L03** codebase extended with
* An interface for a neural network.
* A shallow network, one hidden layer and one output layer, with a full training method.
* A pair of files for utility functions shared across the codebase.

---

## What L04 Builds
**A network on top of the layer interface.** `ml::neural_network::Shallow` holds two
`dense_layer::Interface&` and drives them through feedforward, backpropagation, and optimization.
It never learns what kind of layer it holds, so the stubs from **L03** stand in for real layers
this lecture, and the `Dense` layer from **L05** replaces them without the network changing.

**A training method that sets its own learning rate.** `train()` takes an epoch count and a
precision threshold, but no learning rate. The network starts from a fixed rate and revises it
every hundredth epoch from the precision it measures, following the adaptive rule of
`ml::lin_reg::Adaptive` in **L02**.

**A home for shared helpers.** `ml::initRandGen()` moves into `ml/utils.hpp` and
`source/ml/utils.cpp`, the pair of files **L05** adds three more helpers to.

---

## Files

| File | Contents |
|---|---|
| [include/ml/types.hpp](./include/ml/types.hpp) | Unchanged from **L03**. `MatrixU32` is back in use, for the network's training order. |
| [include/ml/dense_layer/interface.hpp](./include/ml/dense_layer/interface.hpp) | Unchanged from **L03**: `ml::dense_layer::Interface`. |
| [include/ml/dense_layer/stub.hpp](./include/ml/dense_layer/stub.hpp) | Unchanged from **L03**: `ml::dense_layer::Stub`, including the `setOutput()` and `feedforwardCount()` aids the tests rely on. |
| [include/ml/utils.hpp](./include/ml/utils.hpp) | Declares `ml::initRandGen()`. |
| [source/ml/utils.cpp](./source/ml/utils.cpp) | Defines `initRandGen()`, seeding `std::rand()` with the current time once per program run. |
| [include/ml/neural_network/interface.hpp](./include/ml/neural_network/interface.hpp) | `ml::neural_network::Interface`: a virtual destructor and the pure virtual `predict()`, `[[nodiscard]]` and `noexcept` but not `const`. |
| [include/ml/neural_network/shallow.hpp](./include/ml/neural_network/shallow.hpp) | `ml::neural_network::Shallow`, marked `final`: the constructor, `predict()`, `train()`, the deleted default, copy, and move operations, the private methods `initTrainOrder()`, `randomizeTrainOrder()`, `feedforward()`, `backpropagate()`, `optimize()`, and `precision()`, and five member variables. |
| [source/ml/neural_network/shallow.cpp](./source/ml/neural_network/shallow.cpp) | The implementation of `Shallow`, plus three helpers in an anonymous namespace: `updateLearningRate()`, `isEvaluationEpoch()`, which decides whether an epoch is evaluated, and `averageAbs()`. |
| [source/main.cpp](./source/main.cpp) | Demonstration: builds a hidden layer stub with 3 nodes and 2 weights per node and an output layer stub with 1 node and 3 weights, wires them into a `Shallow` with the 2-bit XOR pattern as training data, and prints a prediction for every training input before and after training for 100 epochs. Returns -1 if training fails. |
| [Makefile](./Makefile) | Builds `source/main.cpp`, `source/ml/neural_network/shallow.cpp`, and `source/ml/utils.cpp` into `neural_network` with `g++ -Wall -Werror -std=c++17 -Iinclude`. Targets: `build`, `run`, `clean`, with `build run` as the default. |

The [test](./test/) directory holds this lecture's test suite and is documented in its
[own README](./test/README.md).

---

## Building and Running

```bash
make
```

This builds and runs the application, which prints the predictions before and after training:

```
--------------------------------------------------------------------------------
Predictions before training:
Input: 0 0, predicted output: 0.5
Input: 0 1, predicted output: 0.5
Input: 1 0, predicted output: 0.5
Input: 1 1, predicted output: 0.5
--------------------------------------------------------------------------------
Predictions after training:
Input: 0 0, predicted output: 0.5
Input: 0 1, predicted output: 0.5
Input: 1 0, predicted output: 0.5
Input: 1 1, predicted output: 0.5
--------------------------------------------------------------------------------
```

Both rounds print `0.5`, the stub's default output value: a stub reports what it was constructed
with, whatever it's fed, so there's nothing to learn until **L05**. There's no early-stop line
either. With 100 epochs the loop never reaches an evaluation, since the first one comes after the
101st epoch, and with more epochs the stub's fixed output would keep the precision at `0.5`, far
below the default threshold of `0.999999`.

## Running the Tests

```bash
make -C test
```

All 30 test cases pass: the 14 stub tests carried over from **L03** and 16 for the network.

---

## Notes on the Implementation

### The header only knows the layers by name
[shallow.hpp](./include/ml/neural_network/shallow.hpp) declares `ml::dense_layer::Interface` with a
forward declaration instead of including its header. The class only stores references to layers
and names them in the constructor's parameter list, and neither needs the full definition. Only
[shallow.cpp](./source/ml/neural_network/shallow.cpp), which calls methods on the layers, includes
`ml/dense_layer/interface.hpp`. A file that includes `shallow.hpp` therefore doesn't pull in the
layer interface along with it.

### The constructor checks the wiring and the data
Two conditions make a network useless from the start, and the constructor checks both before
doing anything else:
* The output layer's weight count must equal the hidden layer's node count, since the output layer
  takes one input per hidden node. Otherwise no input could ever pass through the network.
* There must be at least one complete training set, the smaller of the two training data sizes.

Either failure prints a message and calls `std::terminate()`, since a constructor has no return
value to report it through. `train()` relies on both checks: it never needs to test the set count
or the layer shapes again.

The set count isn't stored. `initTrainOrder()` sizes `myTrainOrder` to it, so
`myTrainOrder.size()` is the count from then on, the same reasoning that replaced `mySetCount` in
**L02**.

### `feedforward()` returns `bool`, and `predict()` ignores it
Both `predict()` and `train()` need a pass through both layers, so that pass lives in the private
method `feedforward()`. It returns `false` as soon as a layer rejects its input, and doesn't feed
the output layer at all when the hidden layer has already failed.

`train()` checks that result and stops on a failure. `predict()` has no way to pass it on, since it
returns the output layer's output, so it ignores it. That's why `feedforward()` isn't marked
`[[nodiscard]]`, and why `train()` doesn't call `predict()` instead: `predict()` is
`[[nodiscard]]`, so `train()` would have to discard its value, which `-Werror` rejects.

`predict()` returns a reference to `myOutputLayer.output()` rather than a copy. `Shallow` stores no
prediction of its own, so the value can't go stale: the tests change the output layer's output
behind the network's back with `setOutput()`, and the next prediction reports the new value.

### Three steps, three private methods
Each training set passes through the network in three steps, and each step drives both layers:

| Method | First call | Second call |
|---|---|---|
| `feedforward(input)` | `myHiddenLayer.feedforward(input)` | `myOutputLayer.feedforward(myHiddenLayer.output())` |
| `backpropagate(reference)` | `myOutputLayer.backpropagate(reference)` | `myHiddenLayer.backpropagate(myOutputLayer)` |
| `optimize(input, learningRate)` | `myHiddenLayer.optimize(input, learningRate)` | `myOutputLayer.optimize(myHiddenLayer.output(), learningRate)` |

Each method skips the second call when the first one fails. Backpropagation is the one step that
runs from the output layer backwards, because the hidden layer's error is computed from the output
layer's error and weights, so the output layer has to go first.

With the six layer calls behind three methods, the inner loop of `train()` reads as the three steps
of the theory in [appendix A](../appendix/a_theory.md), and a dimension mismatch in any of them
ends training with `false`.

### `train()` validates, then resets, then trains
`train()` returns `false` for an epoch count of zero or a precision threshold outside `(0.0, 1.0)`.
Only once both arguments are known to be valid does it call `initParams()` on both layers, so a
rejected call leaves a trained network exactly as it was.

The reset means every call to `train()` trains a new network rather than continuing the old one.
Against a stub it changes nothing, since `Stub::initParams()` is empty, but it's what makes it
worth calling `train()` again on a network that got stuck. **L05**'s convergence test relies on it.

### The training state lives in `train()`
The learning rate and the previous precision are local variables in `train()`, starting at `0.01`
and `0.0`. Neither is a member of `Shallow`, so every call to `train()` starts from the initial
learning rate again, rather than from whatever the previous call ended on.

After the inner loop, once per epoch, the helper `isEvaluationEpoch(epoch)` decides whether this
epoch gets evaluated: every hundredth epoch except the first. If so, `precision()` is called once
and its value used twice:
* If it reaches the precision threshold, `train()` prints the precision and the number of epochs,
  and returns `true`. The count is `epoch + 1U`, the number of epochs completed, rather than the
  zero-based loop index.
* Otherwise it's handed to `updateLearningRate()`.

Measuring the precision a second time for the learning rate would feed every training set through
the network again for a value that's already in hand.

### The evaluation interval is 100 epochs
**L02** evaluates every tenth epoch, because a linear regression model reaches its threshold in
tens of epochs. The network needs thousands, so evaluating ten times as often would scan the whole
training data again and again for a precision that has barely moved. Every hundredth epoch keeps
that cost down while the learning rate still gets revised regularly.

The test suite depends on the interval: an evaluation feeds every training set forward once more,
so runs of 1, 101, and 201 epochs must produce 0, 1, and 2 extra passes over the training data.

### The learning rate rule
`updateLearningRate()` sits in the anonymous namespace in `shallow.cpp`, as in **L02**, since it
touches nothing but its arguments. It takes the learning rate and the previous precision by
reference, since it updates both, and compares the precision just measured with the previous one:

| Change in precision | Action |
|---|---|
| Improved by at least `0.1` | Leave the rate alone; it's working. |
| Improved, but by less than `0.1` | Raise the rate by `0.05`; progress has stalled. |
| Didn't improve | Halve the rate; the network is overshooting. |

The limits are applied once afterwards with `std::clamp()`, between `0.01` and `0.25`, and the
precision just measured becomes the previous precision for the next call.

**Why halve?** **L02** lowers the rate by the same step it raises it by. Raising by a fixed step and
cutting by a fixed factor instead is known as *additive increase, multiplicative decrease* (AIMD),
the rule TCP uses to adjust how fast it sends data. The cut scales with the rate, so a high rate
eases down over several evaluations rather than dropping to the floor after one bad measurement,
and a single raise more than makes up for a single halving at the low end.

**Why a floor of `0.01`?** **L02** uses `1e-6`. In a network, a rate that small changes the
parameters too little to measure, so the precision drifts down by a tiny amount between two
evaluations, the rule reads that as another failure to improve, and the rate never leaves the
floor. With `0.01` the network keeps moving, and the rule raises the rate again as soon as it
improves.

The initial rate is `0.01` too, where **L02** starts at `0.1`, so the network starts cautiously and
only speeds up once it has shown it's making progress.

### Precision is an average of averages
`precision()` returns `1.0 - MAE`, where the mean absolute error is computed in two steps. The
helper `averageAbs()` averages the absolute differences within one training set, and
`precision()` averages those per-set values over all sets.

Averaging within each set first keeps the error on the same scale however many output nodes the
network has, where summing the raw differences and dividing by the set count would report an error
that grows with the output layer's width. Every set holds exactly one value per output node, so
the mean of the per-set means equals the mean over every value.

That uniform width isn't luck. `precision()` is only reached after a full epoch, and during that
epoch every training output went through `myOutputLayer.backpropagate()`, which rejects a row of
the wrong length. `averageAbs()` still compares only up to the shorter of its two vectors, so it
can't read out of range even when called without that guarantee.

`precision()` walks the training sets by index, `0` to `N - 1`, rather than in `myTrainOrder`'s
shuffled order. The order doesn't change a sum.

### Training in random order
`initTrainOrder()` fills `myTrainOrder` with `0, 1, 2 ... N - 1`, and `randomizeTrainOrder()`
shuffles it at the start of every epoch by swapping each index with a randomly chosen one. The
inner loop visits the training sets in that order, so the network can't learn anything from the
order the data happens to be stored in.

The random generator is seeded by `initRandGen()`, called once in the constructor rather than in
`randomizeTrainOrder()`, which runs once per epoch. `initRandGen()` seeds only on its first call,
guarded by a `static` flag, and it lives in [utils.cpp](./source/ml/utils.cpp) rather than in an
anonymous namespace: an anonymous namespace would give every `.cpp` that needs it its own copy and
its own flag, so the generator would be reseeded once per file instead of once per program.

### `main.cpp` uses the interface where it can
`printPredictions()` takes an `ml::neural_network::Interface&`, since a prediction is all it needs,
so it works unchanged for any network the course builds later. `trainAndTest()` takes a
`Shallow&`, because `train()` isn't part of the interface: each kind of network trains
differently, but every network predicts the same way.

`train()` is called with the epoch count alone, so it uses the default precision threshold, and
without a learning rate, since the network picks its own.

---

## Where This Goes Next
**L05** adds `ml::dense_layer::Dense`, a real implementation of `dense_layer::Interface` with
weights, biases, and an activation function. Swapped in for the two stubs, it gives the same
`Shallow` a network that actually learns the XOR pattern, and the early-stop line appears once the
precision reaches the threshold. The network code doesn't change, and the stub stays in the
codebase as what the network's tests still run against.

---
