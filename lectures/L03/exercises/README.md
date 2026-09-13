# L03 Exercise Solutions
Solutions and notes for the exercises in [appendix B](../appendix/b_exercises.md). The three
hand-training examples are solved step by step in [appendix C](../appendix/c_solutions.md), so this
directory holds the code only: the start of a new `ml` codebase, consisting of
* An interface for a dense layer.
* A stub that implements the interface without computing anything.
* A small program that checks the pair works.

---

## What L03 Builds
**A new codebase, not an extension of L02.** The linear regression models stay behind in **L02**.
What's carried over is the shape of the code rather than the code itself: an interface in its own
namespace, a concrete class marked `final`, and `std::terminate()` reserved for the constructor.
Only [types.hpp](./include/ml/types.hpp) comes along unchanged.

**An interface for a dense layer.** `ml::dense_layer::Interface` is the contract every dense layer
in the course fulfills: five getters, four computation methods that report failure through their
return value, and `initParams()` for resetting the trainable parameters. Everything is pure virtual,
so the interface says what a layer must be able to do and nothing about how.

**A stub in place of a real layer.** `ml::dense_layer::Stub` implements that contract with range
checks and nothing else. It exists so the network in **L04** can be written, built, and tested
against a real `dense_layer::Interface` a lecture before the real `Dense` layer arrives in **L05**,
and it carries three extra methods that exist purely to make that network testable.

---

## Files

| File | Contents |
|---|---|
| [include/ml/types.hpp](./include/ml/types.hpp) | Unchanged from **L02**: `Matrix1d`, `Matrix2d`, and `Matrix3d` of `double`, plus `MatrixU32`. Only `Matrix1d` and `Matrix2d` are used in this lecture; `MatrixU32` returns in **L04** for the network's training order, and `Matrix3d` is there for the layered models built later in the course. |
| [include/ml/dense_layer/interface.hpp](./include/ml/dense_layer/interface.hpp) | `ml::dense_layer::Interface`: a virtual destructor, the getters `output()`, `error()`, `weights()`, `nodeCount()`, and `weightCount()`, all `[[nodiscard]]`, `const`, and `noexcept`, the computation methods `feedforward()`, both overloads of `backpropagate()`, and `optimize()`, all returning `bool`, and `initParams()`. Every method but the destructor is pure virtual. |
| [include/ml/dense_layer/stub.hpp](./include/ml/dense_layer/stub.hpp) | `ml::dense_layer::Stub`, header only: the constructor with its two zero checks, every override from the interface, the stub-only `setOutput()`, `feedforwardCount()`, and `clearFeedforwardCount()`, the deleted default, copy, and move operations, and four member variables. |
| [source/main.cpp](./source/main.cpp) | Demonstration: builds a hidden layer stub with 3 nodes and 2 weights per node on the default output value, and an output layer stub with 1 node and 3 weights on an explicit output value of `0.75`, and prints each one's counts and output. It then checks that each layer reports the output value it was built with, and that `feedforward()` accepts an input of the right size and rejects one a single element too long, on both layers. Returns -1 if a check fails. |
| [Makefile](./Makefile) | Builds `source/main.cpp` into `dense_layer` with `g++ -Wall -Werror -std=c++17 -Iinclude`. Targets: `build`, `run`, `clean`, with `build run` as the default. |

The [test](./test/) directory holds this lecture's test suite and is documented in its
[own README](./test/README.md).

---

## Building and Running

```bash
make
```

This builds and runs the application, which prints both layers followed by the results of the
checks:

```
Node count:	3
Weight count:	2
Output:		0.5, 0.5, 0.5

Node count:	1
Weight count:	3
Output:		0.75

Output values work as expected!
Feedforward operations work as expected!
```

The hidden layer reports the default `outputValue` of `0.5` and the output layer the `0.75` it was
given, however the layers are used afterwards: the stub reports whatever it was constructed with.
Giving the two layers different values is what lets a network built from them show which layer a
prediction came from, which is how the **L04** tests use the stub.

The shapes aren't arbitrary either. The hidden layer takes 2 inputs into 3 nodes, and the output
layer has 3 weights per node, one for each hidden node, which is exactly the wiring the network in
**L04** requires and checks in its constructor.

## Running the Tests

```bash
make -C test
```

All 14 test cases pass.

---

## Notes on the Implementation

### The stub is header only
Every method of `Stub` is defined inside the class body in
[stub.hpp](./include/ml/dense_layer/stub.hpp), so there's no `stub.cpp`, and `main.cpp` is the only
file the [Makefile](./Makefile) compiles. The test suite's `SOURCE_FILES` list is empty for the
same reason. Both lists grow in **L04**, when the
network and the shared utilities bring the first `.cpp` files under `source/ml/`.

A class this small gains nothing from a separate source file, and every method being visible in
one place makes it easy to confirm that none of them computes anything.

### The counts are read from the vectors
The stub has no `myNodeCount` or `myWeightCount`. `nodeCount()` returns the size of the output
vector, and `weightCount()` returns the size of the first row of the weight matrix. Storing the
counts as well would give each of them two sources of truth, the same reasoning that replaced
`mySetCount` with `myTrainOrder.size()` in **L02**.

Reading row 0 is only safe because the layer is guaranteed at least one node, which is the other
job of the constructor's zero checks: a stub with no nodes would have no row to read.

### The constructor checks before it allocates
The member initializer list leaves all three vectors empty. The node count and the weight count are
checked first, each with its own message, and only once both have passed are the vectors sized:
the output filled with `outputValue`, the error and the weights with `0.0`. A stub constructed with
a zero count therefore terminates without having allocated anything.

There's no bias vector at all. The bias isn't part of `Interface`, and a layer that never optimizes
has nothing to put in one. That's a simplification for the stub only; the `Dense` layer in **L05**
needs a bias vector as much as it needs its weights.

### The range checks are the only logic the stub has
Each computation method compares one size against one count and returns the result:

| Method | Accepts when |
|---|---|
| `feedforward(input)` | `input.size() == weightCount()` |
| `backpropagate(reference)` | `reference.size() == nodeCount()` |
| `backpropagate(nextLayer)` | `nextLayer.weightCount() == nodeCount()` |
| `optimize(input, learningRate)` | the learning rate lies inside `(0.0, 1.0)`, and `input.size() == weightCount()` |

Nothing else happens. The output stays at `outputValue`, and the error and the weights stay at
zero, whether a call is accepted or rejected. The checks still matter, because the network in
**L04** is tested against nothing but this stub: a `feedforward()` that accepted an input of any
size would let a network that hands a vector to the wrong layer pass every one of its tests.

### The learning rate is checked as "inside", not as "not outside"
`optimize()` builds the check as a positive condition:

```cpp
const bool lrValid{(0.0 < learningRate) && (1.0 > learningRate)};
return lrValid && (input.size() == weightCount());
```

For every ordinary number that's equivalent to rejecting the rate when
`(0.0 >= learningRate) || (1.0 <= learningRate)`. The two part ways on NaN: every comparison
involving NaN is false, so NaN fails `lrValid` and is rejected here, while the inverted form would
find neither bound violated and let it through. The test suite only feeds `optimize()` ordinary
numbers, so it can't tell the two forms apart.

### `feedforward()` counts before it checks
`++myFeedforwardCount` is the first statement in `feedforward()`, ahead of the size check, so a
rejected call raises the tally just as an accepted one does. The count records how often the layer
was *asked* to feed forward, which is the question the **L04** tests need answered: a training loop
that runs one pass instead of one per epoch still lines up dimensionally and still returns `true`,
and the count is the only thing that gives it away.

### The test aids aren't part of the interface
`setOutput()`, `feedforwardCount()`, and `clearFeedforwardCount()` exist on `Stub` alone. Code that
holds a layer through a `dense_layer::Interface&`, as the network in **L04** does, can't call them,
so only a test holding the concrete `Stub` can. That's the intended split: the network can't come
to depend on a testing aid, and the tests get two things the interface alone can't give them.
`setOutput()` changes a layer's output behind the network's back, which shows whether the network
reads its output layer live or kept a copy, and the feedforward count shows how often the network
drove each layer.

### `initParams()` is empty, and that's the implementation
`initParams()` is pure virtual in the interface, so `Stub` has to override it, and the override has
an empty body. There's nothing to reset: the stub has no bias, and its weights never move from the
zeros it was constructed with. The empty body isn't a placeholder waiting for **L05**. The network
in **L04** calls `initParams()` on both of its layers before every training run, and against a stub
that call has to compile, run, and change nothing, which is exactly what it does.

### Overrides repeat `[[nodiscard]]`
The getters and `feedforwardCount()` all carry `[[nodiscard]]` on the stub as well as on the
interface. Attributes aren't inherited, so without it a discarded call made on the concrete type,
such as `stub.nodeCount();`, would compile without a word. With it, `g++` rejects that line under
`-Werror` ("ignoring return value of ... declared with attribute 'nodiscard'"), whether the call is
made on a `Stub` or through an `Interface&`. The computation methods don't carry the attribute in
the interface, and their overrides don't add it.

### `main.cpp` goes through the interface
`printLayerInfo()` and `testOutputValue()` take a `const dense_layer::Interface&`, and
`testFeedforward()` takes a non-`const` one, since `feedforward()` changes the layer it's called
on. None of the three knows it was handed a stub, which is the point of the exercise: the printing
and the feedforward check work unchanged on the `Dense` layer from **L05**. The output value check
is the exception, since a real layer computes its output instead of reporting a fixed one.

`testOutputValue()` compares every output element with `!=` rather than within a tolerance. That's
safe here because nothing is computed: the stub copies the value it was constructed with into each
element, and a copied `double` compares exactly equal to its source. The same comparison on a
`Dense` layer's output would be a mistake.

Both test functions are `[[nodiscard]]`, so a result can't be dropped by accident. Within each kind
of check, `main()` combines the two layers' results with `&&`, so a failure in the hidden layer
skips the output layer for that check. The two kinds of check run independently of each other,
though, and each prints its own success line: a stub that ignores its `outputValue` still gets its
feedforward checked, and the program returns -1 if either kind fails.

---

## Where This Goes Next
Nothing in this directory is thrown away. **L04** builds `ml::neural_network::Shallow` on top of
the interface, wires two stubs together as its hidden and output layers, and tests the network
through `setOutput()` and `feedforwardCount()`. **L05** adds `ml::dense_layer::Dense`, a real
implementation of the same interface, and swaps it in for the stubs without the network changing
at all. The stub stays in the codebase even then, because it's still what the network's tests run
against.

---
