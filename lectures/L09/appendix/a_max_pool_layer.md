# Appendix A - Implementing the Real Max Pooling Layer (`ml::conv_layer::MaxPool`)
You'll continue extending `cnn_work` (from L08), implementing a concrete class `MaxPool`,
replacing the temporary stub class `ml::conv_layer::MaxPoolStub` currently used for the max
pooling layer in `factory.cpp`. This is the same logic you already worked through by hand in
[L06's appendix B](../../L06/appendix/b_exercises.md) and implemented as a standalone
struct in [L07's `ml::MaxPoolLayer`](../../L07/appendix/b_max_pool_layer.md): here you're adapting
that same logic to satisfy `cnn_work`'s polymorphic `conv_layer::Interface` instead. `MaxPool` uses
the *same* interface as `Conv` (from L08); a max pooling layer just has no trainable parameters and
no activation function.

---

## Declaring the Max Pooling Layer

### 1. The `MaxPool` class - declaration
In the header file `include/ml/conv_layer/max_pool.hpp` (currently a TODO placeholder), add a class
named `MaxPool` that inherits `ml::conv_layer::Interface` (the same interface `Conv` implements):
* Use public inheritance and mark the class `final`.
* Override every method from the interface, including the destructor (can be `= default`).

---

### 2. Removed constructors and operators
Delete the default constructor, copy constructor, move constructor, copy assignment operator, and
move assignment operator.

---

### 3. Private member variables
* **`myInput`**: the most recently fed-forward input, needed later by `backpropagate()` to
  re-locate which position held the max value in each pool (`Matrix2d`).
* **`myInputGradients`**: input gradients; what `inputGradients()` returns (`Matrix2d`).
* **`myOutput`**: the pooled output; what `output()` returns (`Matrix2d`).

Note this differs from the standalone `ml::MaxPoolLayer` struct you wrote in **L07**, which cached
the max *index* for each pool. Here we cache the input and re-find the max during backpropagation,
which keeps the member list smaller. Both give the same answer, and both break ties toward the
first occurrence in row-major order.

Feel free to add a private helper method that computes the pool size from `myInput`/`myOutput`
(`myInput.size() / myOutput.size()`) rather than storing it separately: that's what the reference
implementation does, since it's always derivable from the two matrices you already have.

---

### 4. Constructor
`explicit MaxPool(std::size_t inputSize, std::size_t poolSize) noexcept`:
* Print an error message and call `std::terminate()` if `inputSize` is `0`, `poolSize` is `0`,
  `inputSize < poolSize`, or `inputSize` isn't evenly divisible by `poolSize`.
* Size `myInput`/`myInputGradients` to `inputSize`, and `myOutput` to `inputSize / poolSize` (see
  `ml::initMatrix()`).

---

### 5. `feedforward()`
`bool feedforward(const Matrix2d& input) noexcept`:
* Return `false` if `input`'s size doesn't match `myInput`'s size (`ml::matchDimensions()`), or if
  `input` isn't square (`ml::isMatrixSquare()`).
* For each output position `(i, j)`, find the largest value in the corresponding
  `poolSize × poolSize` block of `input` (block starting at `(i * poolSize, j * poolSize)`) and
  store it in `myOutput[i][j]`. If a block has more than one occurrence of the max value, it doesn't
  matter which one `feedforward()` treats as "the" max; but `backpropagate()` (below) must be
  consistent with whichever one it picks, so route to the *first* occurrence (matches the L06
  hand-training exercise's convention).
* Store `input` in `myInput` (needed by `backpropagate()`).
* Return `true`.

---

### 6. `backpropagate()`
`bool backpropagate(const Matrix2d& outputGradients) noexcept`:
* Return `false` if `outputGradients`'s size doesn't match `myOutput`'s size, or if it isn't square.
* Reset `myInputGradients` to zero.
* For each output position `(i, j)`: re-locate which position in the corresponding block of
  `myInput` held the max value (first occurrence, same rule as `feedforward()`), and add
  `outputGradients[i][j]` to `myInputGradients` at that position. Every other position in the block
  gets no gradient (stays `0`).
* Return `true`.

---

### 7. `optimize()`
`bool optimize(double learningRate) noexcept`:
* No trainable parameters, so there's nothing to update: this is a genuine no-op, not a
  placeholder. Ignore `learningRate` and return `true`.

---

### 8. Trivial accessors
* **`inputSize()`**: returns `myInputGradients.size()`.
* **`outputSize()`**: returns `myOutput.size()`.
* **`output()`**: returns `myOutput`.
* **`inputGradients()`**: returns `myInputGradients`.

---

## Wiring It In
1. Add `source/ml/conv_layer/max_pool.cpp` to `cnn_work`'s `Makefile` `SRC_FILES`.
2. In `source/ml/factory/factory.cpp`, find `Factory::maxPoolLayer()`: it currently returns a
   `conv_layer::MaxPoolStub` (marked with a `//! @todo` comment). Replace that with your new class:
   ```cpp
   return std::make_unique<conv_layer::MaxPool>(inputSize, poolSize);
   ```
3. Run `make`. Both `Conv` and `MaxPool` are now real; only the flatten layer is still stubbed, so
   `predict()`'s output still won't be meaningful yet. That's next lecture.

---

## Running the tests
The test suite is available in [exercises/test](./exercises/test/). It's cumulative: it carries
L08's conv layer tests over unchanged and adds unit tests for `MaxPool`.

Build and run it from this lecture's `appendix` directory:

```bash
make -C exercises/test
```

There's nothing to copy: `ML_DIR` already points back at the `cnn_work` codebase you started in
L08. All 44 test cases should pass.

Because the pooling layer randomizes nothing, its expected values are written out in full rather
than recomputed the way the conv layer's are; and the main worked case is the hand-training
example from [L06's appendix B](../../L06/appendix/b_exercises.md), the same pooling and gradient
routing you did on paper.

The test worth knowing about before you start is `FeedforwardHandlesNegativeValues`. It feeds an
all-negative image, which catches starting the search for the max at `0.0` instead of at the
block's first value. Every non-negative input passes either way, so it's an easy bug to ship, and
it bites for real once the conv layer in front is using `Tanh`.

See the [test suite's README](./exercises/test/README.md) for more information.

---
