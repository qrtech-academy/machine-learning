# L01 Exercise Solutions
Solutions to the exercises in [appendix B](../appendix/b_exercises.md). The hand-training question is
answered below; the C++ model asked for in sections 1 to 13 is implemented in this directory and
described further down, together with the tuning asked for in section 15.

---

## Linear Regression by Hand
Five training sets defined by `y = 3x + 2`:

| x | y  |
|:-:|:--:|
| 0 |  2 |
| 1 |  5 |
| 2 |  8 |
| 3 | 11 |
| 4 | 14 |

The model starts out with both parameters at zero, and is trained for one epoch with a learning
rate of 10%:

$$\begin{cases} k = 0 \\ m = 0 \end{cases}, LR = 0.1$$

The update rule is the one from [appendix A](../appendix/a_theory.md), applied to every training set,
including the one at $x = 0$:

$$\delta = y_{ref} - y_p, \quad \Delta e = \delta * LR, \quad m = m + \Delta e, \quad k = k + \Delta e * x$$

The training sets are used one at a time, in the order listed above.

### Training set 1
From the first training set we get input $x = 0$ and reference value $y_{ref} = 2$:

$$\begin{cases} x = 0 \\ y_{ref} = 2 \end{cases}$$

Both parameters are zero at the start, so the predicted output $y_p$ is zero as well:

$$y_p = k * x + m = 0 * 0 + 0 = 0$$

The deviation $δ$ is therefore two:

$$\delta = y_{ref} - y_p = 2 - 0 = 2$$

For a learning rate $LR$ of 10%, the adjustment amount $Δe$ is 0.2:

$$\Delta e = \delta * LR = 2 * 0.1 = 0.2$$

The m-value is increased directly by the adjustment amount $Δe$:

$$m = m + \Delta e = 0 + 0.2 = 0.2$$

The k-value is increased by $Δe$ multiplied by x, which when $x = 0$ leaves it untouched:

$$k = k + \Delta e * x = 0 + 0.2 * 0 = 0$$

After the first round of training:

$$\begin{cases} k = 0 \\ m = 0.2 \end{cases}$$

### Training set 2
$$\begin{cases} x = 1 \\ y_{ref} = 5 \end{cases}$$

$$y_p = k * x + m = 0 * 1 + 0.2 = 0.2$$

$$\delta = y_{ref} - y_p = 5 - 0.2 = 4.8$$

$$\Delta e = \delta * LR = 4.8 * 0.1 = 0.48$$

$$m = m + \Delta e = 0.2 + 0.48 = 0.68$$

$$k = k + \Delta e * x = 0 + 0.48 * 1 = 0.48$$

After the second round of training:

$$\begin{cases} k = 0.48 \\ m = 0.68 \end{cases}$$

### Training set 3
$$\begin{cases} x = 2 \\ y_{ref} = 8 \end{cases}$$

$$y_p = k * x + m = 0.48 * 2 + 0.68 = 1.64$$

$$\delta = y_{ref} - y_p = 8 - 1.64 = 6.36$$

$$\Delta e = \delta * LR = 6.36 * 0.1 = 0.636$$

$$m = m + \Delta e = 0.68 + 0.636 = 1.316$$

$$k = k + \Delta e * x = 0.48 + 0.636 * 2 = 1.752$$

After the third round of training:

$$\begin{cases} k = 1.752 \\ m = 1.316 \end{cases}$$

The deviation has grown for three sets in a row, simply because the reference values grow faster
than the parameters have managed to so far.

### Training set 4
$$\begin{cases} x = 3 \\ y_{ref} = 11 \end{cases}$$

$$y_p = k * x + m = 1.752 * 3 + 1.316 = 6.572$$

$$\delta = y_{ref} - y_p = 11 - 6.572 = 4.428$$

$$\Delta e = \delta * LR = 4.428 * 0.1 = 0.4428$$

$$m = m + \Delta e = 1.316 + 0.4428 = 1.7588$$

$$k = k + \Delta e * x = 1.752 + 0.4428 * 3 = 3.0804$$

After the fourth round of training:

$$\begin{cases} k = 3.0804 \\ m = 1.7588 \end{cases}$$

The deviation has decreased for the first time, and the k-value has passed its target of 3.

### Training set 5
$$\begin{cases} x = 4 \\ y_{ref} = 14 \end{cases}$$

$$y_p = k * x + m = 3.0804 * 4 + 1.7588 = 14.0804$$

$$\delta = y_{ref} - y_p = 14 - 14.0804 = -0.0804$$

The deviation is negative for the first time: the prediction overshoots the reference value, so
both parameters are adjusted downwards.

$$\Delta e = \delta * LR = -0.0804 * 0.1 = -0.00804$$

$$m = m + \Delta e = 1.7588 - 0.00804 = 1.75076$$

$$k = k + \Delta e * x = 3.0804 - 0.00804 * 4 = 3.04824$$

After the fifth and final round of training:

$$\begin{cases} k = 3.04824 \\ m = 1.75076 \end{cases}$$

After a single epoch, the model predicts according to the following formula:

$$y_p = 3.04824 * x + 1.75076$$

### Prediction
Predicted output $y_p$ and reference values $y_{ref}$ for input $x$ in the range [-5, 5], with the
predictions rounded to two decimal places:

| $x$ | $y_p$  | $y_{ref}$ |
|:--:|:------:|:---------:|
| -5 | -13.49 | -13       |
| -4 | -10.44 | -10       |
| -3 | -7.39  | -7        |
| -2 | -4.35  | -4        |
| -1 | -1.30  | -1        |
|  0 |  1.75  |  2        |
|  1 |  4.80  |  5        |
|  2 |  7.85  |  8        |
|  3 | 10.90  | 11        |
|  4 | 13.94  | 14        |
|  5 | 16.99  | 17        |

The weight is already within 2% of its target after one epoch, while the bias is still 0.25 short,
which is what tilts every prediction in the table. The bias converges more slowly here because it's
only ever adjusted by $Δe$, whereas the weight is adjusted by $Δe * x$, i.e. up to four times as
much per training set.

The `x = 0` shortcut used by the code in [section 12](../appendix/b_exercises.md#12-optimization) would have
assigned `m = 2` directly on the first training set, and the run would have ended somewhere else
entirely. That's why the exercise text asks for the appendix A rule here: it's the rule the hand
calculations in appendix A demonstrate, and it converges on both parameters from the training data
alone.

---

## Implementation

### Files

| File | Contents |
|---|---|
| [include/ml/types.hpp](./include/ml/types.hpp) | The matrix aliases in namespace `ml`: `Matrix1d`, `Matrix2d`, and `Matrix3d`, i.e. one, two, and three dimensional vectors of `double`. Only `Matrix1d` is used in this lecture; the other two are there for the layered models built later in the course. |
| [include/ml/lin_reg/interface.hpp](./include/ml/lin_reg/interface.hpp) | The `ml::lin_reg::Interface` class: a virtual destructor and the pure virtual `predict()`. That's the whole contract a caller needs in order to predict without knowing which model it was handed. |
| [include/ml/lin_reg/fixed.hpp](./include/ml/lin_reg/fixed.hpp) | Declaration of `ml::lin_reg::Fixed`, the model trained with a fixed learning rate: the constructor, `predict()`, `train()`, the private `optimize()`, the deleted default, copy, and move operations, and the five member variables. |
| [source/ml/lin_reg/fixed.cpp](./source/ml/lin_reg/fixed.cpp) | Definitions for the above: member initialization and the empty-data check in the constructor, `y = kx + m` in `predict()`, argument validation and the epoch loop in `train()`, and the parameter update in `optimize()`. |
| [source/main.cpp](./source/main.cpp) | Demonstration: builds a model over the five training sets from `y = 3x + 2`, trains it, and prints one prediction per training input. Returns -1 if training fails. The two constants at the top of `main()` hold the tuned pair from [section 15](../appendix/b_exercises.md#15-tuning-the-training) rather than the 1000 epochs at 10% the listing starts out with. |
| [Makefile](./Makefile) | Builds the two source files into the `linreg_demo` application with `g++ -Wall -Werror -std=c++17`, and runs it. Targets: `build`, `run`, `clean`, with `build run` as the default. |

The [test](./test/) directory came with the exercise and is documented in its
[own README](./test/README.md).

### Building and running

```bash
make
```

This builds and runs the application, which prints:

```
Input: 0, prediction: 2
Input: 1, prediction: 5
Input: 2, prediction: 8
Input: 3, prediction: 11
Input: 4, prediction: 14
```

That's the output asked for in [section 13](../appendix/b_exercises.md#13-compiling-and-running), reached in
5 epochs at a learning rate of 0.23 rather than the 1000 epochs at 0.1 the listing starts out with;
see the tuning solution below.

### Running the tests

```bash
make -C test
```

All 15 test cases pass against this implementation.

### Tuning the training
[Section 15](../appendix/b_exercises.md#15-tuning-the-training) asks for the fewest epochs that still
reproduce the output above. The table below gives the answer for each learning rate, i.e. the
smallest `epochCount` whose predictions all match the six digits `%g` prints. The numbers come from
running this implementation one epoch at a time and checking the printed output after each:

| Learning rate | Epochs needed |
|:-------------:|:-------------:|
| 0.01          | 45            |
| 0.05          | 8             |
| 0.10          | 6             |
| 0.15          | 10            |
| 0.19          | 11            |
| 0.21          | 8             |
| 0.225         | 5             |
| **0.228**     | **3**         |
| 0.23          | 5             |
| 0.25          | 12            |
| 0.27          | 32            |
| 0.28          | 84            |
| 0.285         | 308           |
| 0.29 and up   | diverges      |

`main.cpp` settles on 5 epochs at 0.23, which is the answer a student is likely to arrive at by
hand. Four things in the table are worth a closer look:
* **Raising the learning rate doesn't monotonically lower the epoch count.** Going from 0.10 to
  0.19 makes the model *slower*, not faster: each pass overshoots the line, and the next has to
  correct back. It improves again above 0.20, bottoms out just before 0.23, then collapses.
* **The true minimum is 3 epochs, and it's luck rather than tuning.** It's only reached between
  0.228 and 0.2283, a window 0.0003 wide, where the third pass happens to land on the line. One
  step outside it costs an epoch, and 0.23 costs two. A setting that narrow says more about these
  five training sets than about the model, which is why 0.23 is the better answer to quote.
* **The upper limit isn't the 1.0 the range check allows.** From 0.29 each pass overshoots by more
  than the previous error, so the parameters grow without bound: 50 epochs at 0.3 predict
  `-2.60711e+06`, and 0.5 reaches `inf` after 361 epochs, after which every prediction is `nan`.
  Curiously, the rates between 0.626 and 0.640 converge again, in as few as 4 epochs, because the
  `x = 0` shortcut resets the bias exactly on every pass and pulls the oscillation back in. It's a
  quirk of this training data, not a setting to use.
* **All of it is specific to this training data.** None of these numbers are properties of the
  algorithm, and neighbouring rates differing by a factor of ten is exactly why the models later in
  the course measure their own precision instead of trusting a hand-tuned epoch count.

---

## Notes on the Implementation
* **The constructor terminates on missing training data.** A constructor has no return value, so
  there's no way to hand the caller an error code; and a model with no complete training set can
  never be trained. The alternative would be to let the object exist in an unusable state and check
  for it in `train()` instead, which just moves the problem.
* **The model keeps references to the training data, it doesn't copy it.** The vectors therefore
  have to outlive the model. Passing a temporary compiles cleanly and then reads freed memory, so
  the training data is declared as named variables in `main()`, above the model that uses it.
* **`optimize()` treats `x = 0` as a special case,** assigning `myBias = output` and leaving
  `myWeight` alone. Since `y = k * 0 + m = m`, the reference value *is* the bias for that input,
  and no value of `k` can change a prediction made with an input of zero. It's a shortcut rather
  than a better rule; see the note at the end of the hand-training solution above, and the last
  section of [appendix A](../appendix/a_theory.md).
* **`train()` continues from the current parameters,** it doesn't reset them. Calling it twice for
  one epoch each gives the same model as one call for two epochs, which is what makes it possible
  to train in stages and inspect the model in between.
* **`predict()` is the only virtual method.** `evaluateModel()` in `main.cpp` takes a
  `const ml::lin_reg::Interface&` and therefore works with any model that can predict, without
  knowing which one it was handed; the `Adaptive` variant mentioned in
  [section 6](../appendix/b_exercises.md#6-subclass-fixed) would slot straight in. Training is deliberately
  left out of the interface, since how a model is trained is exactly what differs between the
  variants, while all of them predict with the same `y = kx + m`.

---
