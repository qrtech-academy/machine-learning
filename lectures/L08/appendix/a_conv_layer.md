# Appendix A - Implementing the Real Conv Layer (`ml::conv_layer::Conv`)
You'll extend the `cnn_work` codebase, implementing a concrete class `Conv`, replacing
the temporary stub class `ml::conv_layer::ConvStub` currently used for the conv layer in
`factory.cpp`. This is the same math you already worked through by hand in
[L06's appendix B](../../L06/appendix/b_exercises.md) and implemented as a standalone struct in
[L06/L07's `ml::ConvLayer`](../../L06/appendix/c_conv_layer.md): here you're adapting that same
logic to satisfy `cnn_work`'s polymorphic `conv_layer::Interface` instead.

---

## Declaring the Conv Layer

### 1. The `Conv` class - declaration
In the header file `include/ml/conv_layer/conv.hpp` (currently a TODO placeholder), add a class
named `Conv` that inherits `ml::conv_layer::Interface` (`include/ml/conv_layer/interface.hpp`):
* Use public inheritance and mark the class `final` so it can't be inherited further.
* Override every method from the interface, including the destructor.

**Tip**: copy the entire contents of the interface and paste it into the new file, then adapt it (no
`virtual`/`= 0`, add `override` everywhere; including on `inputGradients()`, which is easy to miss).

---

### 2. Removed constructors and operators
Delete the default constructor, copy constructor, move constructor, copy assignment operator, and
move assignment operator.

---

### 3. Private member variables
Add the following private member variables to `Conv`:
* **`myInputPadded`**: the zero-padded input (`Matrix2d`).
* **`myInputGradientsPadded`**: input gradients, padded (`Matrix2d`).
* **`myInputGradients`**: input gradients, unpadded; what `inputGradients()` returns (`Matrix2d`).
* **`myKernel`**: the kernel weights (`Matrix2d`).
* **`myKernelGradients`**: kernel gradients (`Matrix2d`).
* **`myOutput`**: the layer's output (feature map); what `output()` returns (`Matrix2d`).
* **`myPreActivationOutput`**: the sum before the activation function is applied, needed later by
  `backpropagate()` to compute the activation's derivative (`Matrix2d`).
* **`myBias`**: the bias value (`double`).
* **`myBiasGradient`**: the bias gradient (`double`).
* **`myActFunc`**: pointer to the activation function (`ActFuncPtr`, see `ml/act_func/interface.hpp`).

Feel free to add private helper methods as needed: the reference implementation uses one to
validate the constructor's parameters, one to (re-)build the zero-padded input, and one to strip the
padding back off the input gradients, but you're free to structure this however you like.

---

### 4. Constructor
`explicit Conv(std::size_t inputSize, std::size_t kernelSize, act_func::Type actFunc = act_func::Type::None) noexcept`:
* Print an error message and call `std::terminate()` if `kernelSize` is outside `[1, 11]`, or if
  `inputSize < kernelSize`.
* Compute the padding offset (`kernelSize / 2`) and the padded size (`inputSize + 2 * offset`).
* Size `myInputPadded`/`myInputGradientsPadded` to the padded size, `myInputGradients`/`myOutput`/
  `myPreActivationOutput` to `inputSize`, and `myKernel`/`myKernelGradients` to `kernelSize` (see
  `ml::initMatrix()` in `ml/utils.hpp`).
* Randomize the bias and every kernel weight using `ml::randomStartVal()`.
* Get the activation function: `factory::Factory factory{}; myActFunc = factory.actFunc(actFunc);`
  (see `ml/factory/factory.hpp`; this is the same pattern the already-finished `Dense` layer uses
  in `dense.cpp`, if you want a working example to compare against).

---

### 5. `feedforward()`
`bool feedforward(const Matrix2d& input) noexcept`:
* Return `false` if `input`'s size doesn't match `myOutput`'s size (`ml::matchDimensions()`), or if
  padding the input fails (see below).
* Zero-pad `input` into `myInputPadded`: for every position `(i, j)` in `input`, write it to
  `myInputPadded[i + offset][j + offset]`, where `offset = myKernel.size() / 2`. Return `false` if
  `input` isn't square (`ml::isMatrixSquare()`).
* For each output position `(i, j)`: sum `myBias` plus the element-wise product of `myKernel` and
  the corresponding `kernelSize × kernelSize` window of `myInputPadded` starting at `(i, j)`. Store
  the sum in `myPreActivationOutput[i][j]`, then store `myActFunc->output(sum)` in `myOutput[i][j]`.
* Return `true`.

---

### 6. `backpropagate()`
`bool backpropagate(const Matrix2d& outputGradients) noexcept`:
* Return `false` if `outputGradients`'s size doesn't match `myOutput`'s size, or if it isn't square.
* Reset `myInputGradientsPadded`, `myInputGradients`, `myKernelGradients` to zero, and
  `myBiasGradient` to `0.0`.
* For each output position `(i, j)`: compute `delta = myActFunc->delta(myPreActivationOutput[i][j])`
  and `outGrad = outputGradients[i][j] * delta`. Add `outGrad` to `myBiasGradient`. Then, for each
  kernel position `(ki, kj)`: add `myInputPadded[i+ki][j+kj] * outGrad` to
  `myKernelGradients[ki][kj]`, and add `myKernel[ki][kj] * outGrad` to
  `myInputGradientsPadded[i+ki][j+kj]`.
* Strip the padding border back off `myInputGradientsPadded` into `myInputGradients` (the exact
  inverse of the padding step in `feedforward()`).
* Return `true`.

---

### 7. `optimize()`
`bool optimize(double learningRate) noexcept`:
* Return `false` if the learning rate is invalid (`ml::checkLearningRate()`).
* `myBias += myBiasGradient * learningRate`.
* For every kernel position: `myKernel[i][j] += myKernelGradients[i][j] * learningRate`.
* Return `true`.

---

### 8. Trivial accessors
* **`inputSize()`**: returns `myInputGradients.size()`.
* **`outputSize()`**: returns `myOutput.size()`.
* **`output()`**: returns `myOutput`.
* **`inputGradients()`**: returns `myInputGradients`.

Add two more accessors, which are **not** part of `conv_layer::Interface`:
* **`kernel()`**: returns `myKernel` (`const Matrix2d&`).
* **`bias()`**: returns `myBias` (`double`).

These two exist for the test suite. Because the constructor randomizes the bias and the kernel,
nothing the layer computes can be predicted from the constructor arguments alone; the tests read
these back and redo the arithmetic by hand to check the layer got the same answer. They're declared
on `Conv` rather than on the interface because the max pooling layer implements that same interface
in L09 and has neither a kernel nor a bias to report.

---

## Wiring It In
1. Make sure `source/ml/conv_layer/conv.cpp` is listed in `cnn_work`'s `Makefile` under `SRC_FILES`.
2. In `source/ml/factory/factory.cpp`, find `Factory::convLayer()`: it currently returns a
   `conv_layer::ConvStub` (marked with a `// TODO`). Replace that with your new class:
   ```cpp
   return std::make_unique<conv_layer::Conv>(inputSize, kernelSize, actFunc);
   ```
3. Run `make`. The conv layer in `cnn_work` is now real; though the rest of the pipeline (max
   pooling, flatten) is still stubbed, so `predict()`'s output won't be meaningful yet. That
   happens once every layer is real, at the end of L10.

---

## Running the tests
A test suite for `Conv` is available in [exercises/test](./exercises/test/). It's the first suite of
part II and covers this lecture's layer only; L09 and L10 add their own.

Build and run it from that directory:

```bash
make -C exercises/test
```

You don't need to copy anything: unlike L01 - L05, the suite's `ML_DIR` already points at
`cnn_work`, where you're writing the code. All 22 test cases should pass.

Until `Conv` exists the build fails with `'Conv' in namespace 'ml::conv_layer' does not name a
type` - the suite is the specification, so it doesn't compile until the class does. Note also that
the tests construct `Conv` directly, so they pass whether or not you've done the `factory.cpp`
wiring step above; run `cnn_work` itself to check that.

See the [test suite's README](./exercises/test/README.md) for more information, including why the
layer needs `kernel()` and `bias()` at all.

---
