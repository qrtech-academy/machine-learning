# Appendix A - Implementing the Real Flatten Layer (`ml::flatten_layer::Flatten`) & Wiring the CNN Together
You'll finish `cnn_work` (from L08/L09), implementing a concrete class `Flatten`, replacing the
temporary stub class `ml::flatten_layer::Stub` currently used for the flatten layer in
`factory.cpp`. This is the same logic you already worked through by hand in
[L06's appendix B](../../L06/appendix/b_exercises.md) and implemented as a standalone struct in
[L07's `ml::FlattenLayer`](../../L07/appendix/c_flatten_layer.md): here you're adapting that same
logic to satisfy `cnn_work`'s polymorphic `flatten_layer::Interface`. Once this is done, every layer
in `cnn_work` is real, and you'll wire the whole pipeline together for the first time.

---

## Declaring the Flatten Layer

### 1. The `Flatten` class - declaration
In the header file `include/ml/flatten_layer/flatten.hpp` (currently a TODO placeholder), add a
class named `Flatten` that inherits `ml::flatten_layer::Interface`:
* Use public inheritance and mark the class `final`.
* Override every method from the interface, including the destructor (can be `= default`).

**Note**: unlike `Conv` and `MaxPool`, `flatten_layer::Interface` declares **no `optimize()`
method at all**: not because it was left out, but because a flatten layer has nothing to
optimize and the interface simply doesn't ask for it. Don't add one.

---

### 2. Removed constructors and operators
Delete the default constructor, copy constructor, move constructor, copy assignment operator, and
move assignment operator.

---

### 3. Private member variables
* **`myInputGradients`**: the unflattened gradients; what `inputGradients()` returns (`Matrix2d`).
* **`myOutput`**: the flattened output; what `output()` returns (`Matrix1d`, not `Matrix2d`; note
  the interface's `output()` returns a 1D vector here, unlike `Conv`/`MaxPool`).

---

### 4. Constructor
`explicit Flatten(std::size_t inputSize) noexcept`:
* Print an error message and call `std::terminate()` if `inputSize` is `0`.
* Size `myInputGradients` to `inputSize` (a square `inputSize × inputSize` matrix), and `myOutput`
  to `inputSize * inputSize` (see `ml::initMatrix()`).

---

### 5. `feedforward()`
`bool feedforward(const Matrix2d& input) noexcept`:
* Return `false` if `input`'s size doesn't match `myInputGradients`'s size
  (`ml::matchDimensions()`), or if `input` isn't square (`ml::isMatrixSquare()`).
* Row-major flatten: for every position `(i, j)` in `input`, write it to
  `myOutput[i * inputSize + j]`, where `inputSize = myInputGradients.size()`.
* Return `true`.

---

### 6. `backpropagate()`
`bool backpropagate(const Matrix1d& outputGradients) noexcept`:
* Return `false` if `outputGradients`'s size doesn't match `myOutput`'s size.
* The exact inverse of `feedforward()`: for every position `(i, j)`, set
  `myInputGradients[i][j] = outputGradients[i * inputSize + j]`.
* Return `true`.

---

### 7. Trivial accessors
* **`inputSize()`**: returns `myInputGradients.size()`.
* **`outputSize()`**: returns `myOutput.size()`.
* **`inputGradients()`**: returns `myInputGradients`.
* **`output()`**: returns `myOutput`.

---

## Wiring It In
1. Add `source/ml/flatten_layer/flatten.cpp` to `cnn_work`'s `Makefile` `SRC_FILES`.
2. In `source/ml/factory/factory.cpp`, find `Factory::flattenLayer()`: it currently returns a
   `flatten_layer::Stub` (marked with a `// TODO`). Replace that with your new class:
   ```cpp
   return std::make_unique<flatten_layer::Flatten>(inputSize);
   ```
3. Run `make`. Every layer in `cnn_work` (`Conv`, `MaxPool`, `Flatten`, and `Dense`, already done
   back in L05) is now real. `predict()`'s output is finally meaningful, and `train()` actually
   trains a working CNN end to end.
4. Compare notes against [`cnn_demo`](../cnn_demo), the finished reference implementation sitting
   right alongside this lecture's material: if your `cnn_work` doesn't train the way you expect,
   this is a good place to check your work against.

---
