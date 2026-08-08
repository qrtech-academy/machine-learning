# Machine Learning, Written Examination, Paper B

**Time:** 4 hours. **Closed book.** A basic calculator is permitted. **Total: 100 marks.**

This paper exists to test your own skills and knowledge. It is not a qualification and it gates
nothing. It draws on all ten lectures, so **it is meant to be taken once the course is over**, after
L10.

---

## Rubric

**Sign convention.** The course computes a deviation $\delta = y_{ref} - y_p$ and *adds* the
resulting adjustment to every parameter. Every update in this paper therefore reads
`parameter += gradient * learningRate`, with no minus sign. An answer that flips the sign
consistently and says so is not penalised twice.

**The activation derivative is evaluated at the weighted sum $s$**, never at the output $y$. This
matters in more than one question.

$$\sigma_{ReLU}(s) = \begin{cases} s & s > 0 \\ 0 & s \leq 0\end{cases} \qquad
\sigma'_{ReLU}(s) = \begin{cases} 1 & s > 0 \\ 0 & s \leq 0\end{cases}$$

$$\sigma_{Tanh}(s) = \tanh(s) \qquad \sigma'_{Tanh}(s) = 1 - \tanh^2(s) \qquad
\sigma_{None}(s) = s \qquad \sigma'_{None}(s) = 1$$

**Convolutional layers zero-pad** with `pad = kernelSize / 2` (integer division), so the output has
the same size as the input.

**Max pooling breaks ties toward the first occurrence** in row-major order, and **flatten is
row-major**.

**Code answers** may be written in C++17 or in unambiguous pseudocode. Nothing on this paper is
marked on syntax; a missing semicolon costs nothing and a missing bounds check costs everything.

**Show the arithmetic.** Where a question asks you to compute, an unsupported number scores
nothing, and a correct method with a slip in it scores most of the marks.

### Supplied constants

| Quantity          | Value    |
| ----------------- | -------- |
| $\tanh(1.5)$      | 0.905148 |
| $\tanh(1.456072)$ | 0.896887 |
| $\tanh(0.905148)$ | 0.718796 |
| $2^{16}$          | 65536    |
| 1 KB              | 1024 B   |

### Marks

| Question | 1   | 2   | 3   | 4   | 5   | 6   | 7   | 8   | 9   |
| -------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Marks    | 10  | 10  | 14  | 12  | 12  | 14  | 12  | 8   | 8   |

---

## Question 1, The model, and the rate that destroys it (10 marks)

**(a)** Traditional programming and machine learning both involve input, rules and output, and
differ in which two are supplied and which one is produced. State the arrangement for each.

Then state the property of image data that makes the traditional arrangement impractical for image
classification. (2 marks)

**(b)** A linear regression model starts at $k = 0$, $m = 0$ and is trained with $LR = 0.5$ on data
following $y = 2x + 1$. Three optimization steps are performed, in this order:

$$(x, y_{ref}) = (3, 7), \quad (4, 9), \quad (3, 7)$$

Compute $y_p$, $\delta$, $\Delta e$, $k$ and $m$ after each step, and state what happens to
$\lvert\delta\rvert$ across the three. (3 marks)

**(c)** For a single training set $(x, y_{ref})$, let $\delta$ be the deviation before an
optimization step and $\delta'$ the deviation for the same input immediately after it.

Derive $\delta'$ as a multiple of $\delta$, in terms of $LR$ and $x$. Hence give the condition on
$LR$ under which the deviation shrinks, and evaluate it at $x = 4$. (3 marks)

**(d)** `train()` returns `false` when the learning rate falls outside $(0.0, 1.0)$. Give the reason
the course states for each end of that range.

Then use your answer to (c) to state why passing that check does not guarantee the model converges.
(2 marks)

---

## Question 2, Order, precision, and knowing when to stop (10 marks)

**(a)** The training order is reshuffled at the start of every epoch, not once at construction.
State what the model risks learning if it is not, and what the shuffle is therefore protecting.
(2 marks)

**(b)** `initRandGen()` uses a `static` local variable. State what that variable is for, and what
would happen without it.

In L02 the function lived in an anonymous namespace inside `fixed.cpp`; in L04 it moved to a shared
`ml/utils.cpp`. State exactly what breaks in the L02 arrangement once a second `.cpp` file needs the
generator. (3 marks)

**(c)** The model's precision is computed as $1.0 - MAE$. A model predicts as follows:

| $x$ | $y_{ref}$ | $y_p$ |
| :-: | :-------: | :---: |
|  1  |     2     | 1.85  |
|  2  |     4     |  3.9  |
|  3  |     6     |  6.3  |
|  4  |     8     |  8.1  |

Compute the precision.

The precision threshold must lie strictly inside $(0.0, 1.0)$. Give the reason for each end
separately. Then give a set of predictions for the same reference values that makes the precision
negative, and state what that reveals about the metric. (3 marks)

**(d)** The precision is computed every tenth epoch rather than every epoch. State what that saves,
and give the two prices paid for it. (2 marks)

---

## Question 3, Two output nodes (14 marks)

The network below has two inputs, two hidden nodes ($y_1$, $y_2$) and two output nodes ($y_3$,
$y_4$). Every node uses ReLU, and the learning rate is $LR = 0.1$.

**Hidden layer:** $b_1 = 0.2$, $b_2 = 0.5$, with $w_1 = 0.3$ and $w_2 = 0.6$ feeding node 1 from
$x_1$ and $x_2$, and $w_3 = -0.4$ and $w_4 = 0.2$ feeding node 2 from $x_1$ and $x_2$.

**Output layer:** $b_3 = -0.1$, $b_4 = 0.4$, with $w_5 = 0.7$ and $w_6 = 0.5$ feeding node 3 from
$y_1$ and $y_2$, and $w_7 = -0.2$ and $w_8 = 0.9$ feeding node 4 from $y_1$ and $y_2$.

Training set: $X_1X_2 = 1\ 0$, $Y_1Y_2 = 0\ 1$.

**(a)** Perform feedforward. Give $s$ and $y$ for all four nodes. (3 marks)

**(b)** Perform backpropagation. Give $\delta$ and $e$ for both output nodes and both hidden nodes.

State how many terms enter each hidden node's $\delta$, and why. (4 marks)

**(c)** Perform optimization. Give all four updated biases and all eight updated weights. (4 marks)

**(d)** Feed the same input forward again with the updated parameters. Give both new outputs and
both new deviations, and state whether the step helped. (2 marks)

**(e)** Two of the eight weights come out of (c) with exactly the values they went in with. Name
both and give the reason. (1 mark)

---

## Question 4, The network class, and what a stub can prove (12 marks)

**(a)** `Shallow` stores its two layers as references, not by value. Give two consequences of that
choice.

Then state what `predict()` returns and why the class needs no member variable to hold it.
(3 marks)

**(b)** `train()` performs three steps per training set. Name them in order, and state the two
separate ordering constraints that make that the only correct sequence. (3 marks)

**(c)** The stub carries two methods that `dense_layer::Interface` does not declare:
`setOutput()` and `feedforwardCount()`. State what each one makes testable, and why nothing else in
the interface can establish the same thing.

`feedforwardCount()` is incremented before the input size is checked, so a rejected call raises it
too. State the argument for that choice. (4 marks)

**(d)** The constructor calls `std::terminate()` on bad arguments; `train()` returns `false`
instead. State the rule the course applies and the reason behind it. (2 marks)

---

## Question 5, The dense layer, and the mistake only Tanh can catch (12 marks)

**(a)** `Dense` has six member variables. Name each one, give its size, and give the quantity from
the L03 theory it corresponds to. (3 marks)

**(b)** A `Dense` layer with `ActFunc::Tanh` has one node and two weights:

$$b = 0.3, \quad w_0 = 0.6, \quad w_1 = 0.6$$

It is fed `input = {1.0, 1.0}` with `reference = {0.5}` and a learning rate of 0.2.

Perform feedforward, backpropagation and optimization: give $s$, $y$, $\delta$, $\sigma'(s)$,
$\Delta e$, the updated bias and both updated weights. Then feed the same input forward again and
give the new deviation. (5 marks)

**(c)** The layer is now changed to compute
`myError[i] = err * actFuncDelta(myActFunc, myOutput[i])`.

Recompute $\Delta e$ for the node in (b) under that change and give the ratio between the wrong
value and the right one.

State why the same mistake is invisible when the layer uses `ActFunc::Relu`, and state what the
consequence of that is for how the mistake has to be tested. (4 marks)

---

## Question 6, Padding, sharing, and the kernel's gradient (14 marks)

A conv layer takes a 4 x 4 input, has one 3 x 3 kernel, stride 1, zero-padding with
`pad = kernelSize / 2`, and `ActFunc::None`. The learning rate is 0.01.

**Input image:**

```text
0 1 1 0
1 1 1 1
1 1 1 1
0 1 1 0
```

**Kernel:**

```text
 1   0  -1
 1   0  -1
 1   0  -1
```

**Bias:** 0.0

**(a)** Give the padded input and compute the layer's full 4 x 4 output.

State what feature this kernel responds to, and what the sign of an output value tells you.
(3 marks)

**(b)** The same 4 x 4 input is now given to a layer with a **2 x 2** kernel instead.

Give the padded size in both cases. Then show that with the 2 x 2 kernel, one row and one column of
the padded input are never read by `feedforward()` at all, and state what that implies about how the
output lines up with the input. (3 marks)

**(c)** Returning to the 3 x 3 kernel, the following output gradients arrive:

```text
0  2  0  0
0  0  0  4
6  0  0  0
0  0  8  0
```

Compute the bias gradient and the full 3 x 3 matrix of kernel gradients, then give the updated
kernel and bias. (4 marks)

**(d)** Give the number of trainable parameters in this conv layer, and the number a dense layer
producing the same 16 outputs from the same 16 inputs would have.

The course says the gradients "from all those positions must be accumulated before the kernel's
weights are updated". Using your answer to (c), state how many separate output positions
contributed to the centre kernel gradient, and state why the layer nevertheless applies exactly one
update to it. (4 marks)

---

## Question 7, A pooling layer that runs and returns true (12 marks)

`MaxPool::feedforward()` is correct: it scans each block, keeps a value only when it is strictly
greater than the best so far, and therefore records the **first** occurrence of the maximum.
`backpropagate()` is written as follows:

```cpp
bool MaxPool::backpropagate(const Matrix2d& outputGradients) noexcept
{
    if (!ml::matchDimensions(outputGradients, myOutput)) { return false; }

    const auto poolSize{myInput.size() / myOutput.size()};

    for (std::size_t i{}; i < myOutput.size(); ++i)
    {
        for (std::size_t j{}; j < myOutput.size(); ++j)
        {
            auto maxValue{myInput[i * poolSize][j * poolSize]};
            std::size_t maxRow{i * poolSize};
            std::size_t maxCol{j * poolSize};

            for (std::size_t r{}; r < poolSize; ++r)
            {
                for (std::size_t c{}; c < poolSize; ++c)
                {
                    const auto value{myInput[i * poolSize + r][j * poolSize + c]};

                    if (value >= maxValue)
                    {
                        maxValue = value;
                        maxRow   = i * poolSize + r;
                        maxCol   = j * poolSize + c;
                    }
                }
            }
            myInputGradients[maxRow][maxCol] = outputGradients[i][j];
        }
    }
    return true;
}
```

**(a)** There are two independent faults in this method. Name both, and for each state exactly what
the conv layer behind this one receives that it should not. (4 marks)

**(b)** The layer's most recent `feedforward()` input was:

```text
 0.5   0.5  -0.1  -0.4
 0.2   0.3  -0.6  -0.9
 1.0   0.4   0.7   0.2
 0.1   0.9   0.3   0.8
```

The gradients $\begin{bmatrix} 10 & 20 \\ 30 & 40\end{bmatrix}$ arrive. Give the matrix this code
produces and the matrix it should have produced, and name the block in which they differ. (3 marks)

**(c)** The same class implements `optimize()` as follows:

```cpp
bool MaxPool::optimize(const double learningRate) noexcept
{
    if (!ml::checkLearningRate(learningRate)) { return false; }

    for (std::size_t i{}; i < myInput.size(); ++i)
    {
        for (std::size_t j{}; j < myInput.size(); ++j)
        {
            myInput[i][j] += myInputGradients[i][j] * learningRate;
        }
    }
    return true;
}
```

State what is wrong with this, give the correct body, and give the general rule about layers with no
trainable parameters.

`flatten_layer::Interface` handles the same situation differently. State how, and state which of the
two designs is the stronger one. (3 marks)

**(d)** Every fault above leaves a network that compiles, runs, and returns `true` from every call.
State the symptom a user of the finished CNN actually sees, and name the L10 test that would fail.
(2 marks)

---

## Question 8, Every layer, side by side (8 marks)

**(a)** Complete the table below for all five layer types built during the course. For each, give
the trainable parameters, what `feedforward()` computes, what `backpropagate()` computes, and what
`optimize()` does.

| Layer     | Trainable parameters | `feedforward()` | `backpropagate()` | `optimize()` |
| --------- | -------------------- | --------------- | ----------------- | ------------ |
| `Fixed`   |                      |                 |                   |              |
| `Dense`   |                      |                 |                   |              |
| `Conv`    |                      |                 |                   |              |
| `MaxPool` |                      |                 |                   |              |
| `Flatten` |                      |                 |                   |              |

(4 marks)

**(b)** "A conv layer is a dense layer with two constraints added." Name both constraints, and state
in one sentence each what they buy and what they cost. (2 marks)

**(c)** A single training example is passed through `input -> Conv -> MaxPool -> Flatten -> Dense`.

Name the method called on each layer, in order, for all three phases of the training step, and state
what each call is given as its argument. (2 marks)

---

## Question 9, Off the laptop (8 marks)

**(a)** A CNN is built as follows:

```text
input (32x32) -> conv (one 3x3 kernel, same padding) -> max pooling (2x2, stride 2)
              -> flatten -> dense (64 nodes)
```

Compute the total trainable-parameter count, then the footprint stored as `double`, as `float`, and
as `int8`.

The target has 64 KB of RAM available. State which of the three fits, and show the comparison in
bytes. (3 marks)

**(b)** The same network is to run on a target without a hardware FPU, using Q16.16 fixed-point.

Convert $0.75$ and $-2.5$ to Q16.16. Multiply them the way `fixedMultiply()` does, giving the
product before and after the shift, and the value that represents. Show that the intermediate does
not fit in a 32-bit integer.

Give the resolution of Q16.16. (3 marks)

**(c)** Name two constraints of resource-constrained targets other than memory footprint and the
absence of an FPU.

Then compute the number of multiply-accumulate operations the conv layer in (a) performs per
feedforward pass, and state why the *worst-case* execution time matters more than the average on
such a target. (2 marks)

---
