# Machine Learning, Written Examination, Paper A

**Time:** 4 hours. **Closed book.** A basic calculator is permitted. **Total: 100 marks.**

This paper exists to test your own skills and knowledge. It is not a qualification and it gates
nothing. It draws on all ten lectures, so **it is meant to be taken once the course is over**,
after L10.

---

## Rubric
**Sign convention.** The course computes a deviation $\delta = y_{ref} - y_p$ and *adds* the
resulting adjustment to every parameter. Every update in this paper therefore reads
`parameter += gradient * learningRate`, with no minus sign. An answer that flips the sign
consistently and says so is not penalized twice.

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

| Quantity                | Value    |
| ----------------------- | -------- |
| $\tanh(1.5)$            | 0.905148 |
| $\tanh(0.905148)$       | 0.718795 |
| 1 KB                    | 1024 B   |

### Marks

| Question | 1   | 2   | 3   | 4   | 5   | 6   | 7   | 8   | 9   |
| -------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Marks    | 12  | 12  | 14  | 10  | 12  | 10  | 14  | 8   | 8   |

---

## Question 1, Learning the rules instead of writing them (12 marks)

**(a)** Traditional programming and machine learning both involve the same three things: input,
rules, and output. They differ in which two are supplied and which one is produced. State the
arrangement for each of the two.

Then name the pre-machine-learning paradigm the course contrasts with, give one class of task it
handled well and one class it did not, and state the property of the input that separates the two.
(3 marks)

**(b)** A linear regression model is to be trained on the training data below, which follows
$y = 4x - 1$:

| $x$ | $y$  |
| :-: | :--: |
|  0  |  -1  |
|  1  |   3  |
|  2  |   7  |

Both parameters start at zero, the learning rate is $LR = 0.1$, and one epoch is run with the sets
taken in the order shown.

Using the general update rules from L01, $\Delta e = \delta \cdot LR$ applied to both parameters,
with no special case for any input, compute $y_p$, $\delta$, $\Delta e$, $k$ and $m$ for each of
the three training sets. (4 marks)

**(c)** Give the trained model's prediction at $x = 5$ and compare it against the reference value.

The worked example in L01 (which trained on $y = 2x + 1$) landed within a few hundredths of its
target after a single epoch. State why this model did not, quoting the specific numbers from (b)
that account for the difference. (2 marks)

**(d)** `ml::lin_reg::Fixed::optimize()` treats $x = 0$ as a special case: it assigns the reference
value straight to the bias and leaves the weight untouched.

Redo the first training set under that rule, carry the result through the remaining two sets, and
compare the final parameters against (b). Then state one situation in which the special case is a
bad idea. (3 marks)

---

## Question 2, The training loop that trains nothing (12 marks)

The training method below compiles, runs, and returns `true`. It is the L04 training loop, and it is
wrong in several independent ways.

```cpp
bool Shallow::train(const std::size_t epochCount, const double precisionThreshold) noexcept
{
    constexpr std::size_t evaluationInterval{100U};
    constexpr double initialLearningRate{0.1};

    double learningRate{initialLearningRate};
    double prevPrecision{};

    if ((0U == epochCount) || (0.0 >= precisionThreshold) || (1.0 <= precisionThreshold))
    {
        return false;
    }
    myHiddenLayer.initParams();
    myOutputLayer.initParams();

    randomizeTrainOrder();

    for (std::size_t epoch{}; epoch < epochCount; ++epoch)
    {
        for (std::size_t x{}; x < myTrainOrder.size(); ++x)
        {
            predict(myTrainInput[x]);

            myHiddenLayer.backpropagate(myOutputLayer);
            myOutputLayer.backpropagate(myTrainOutput[x]);

            myHiddenLayer.optimize(myTrainInput[x], learningRate);
            myOutputLayer.optimize(myHiddenLayer.output(), learningRate);
        }
        const bool evaluate{(0U != epoch) && (0U == (epoch % evaluationInterval))};

        if (evaluate)
        {
            const auto currentPrecision = precision();

            if (precisionThreshold <= currentPrecision) { return true; }
            updateLearningRate(learningRate, prevPrecision, currentPrecision);
        }
    }
    return true;
}
```

The parameter reset, the evaluation every hundredth epoch and the learning rate rule are all as
specified; the faults below are elsewhere.

**(a)** The loop was supposed to visit the training sets in a different random order every epoch.
Two separate faults prevent it. Name both, and state what order the sets are actually visited in.
(3 marks)

**(b)** Backpropagation is performed in the wrong order. State the correct order, state precisely
what the hidden layer computes its error from as the code stands, and state what value it computes
on the very first training set of the very first epoch. (3 marks)

**(c)** Each of the four layer calls returns `bool` and every one of those return values is
discarded. State what the four calls report through that value, name the failure it was meant to
catch, and state what this function returns when that failure occurs. (2 marks)

**(d)** The L03 stub exposes three methods that are not part of `dense_layer::Interface`. Two of them are:
`feedforwardCount()` and `setOutput()`. State what each one makes testable.

Then state, for each of (a), (b) and (c) in turn, whether a test built on those two methods catches
the fault described there, with the reason. Finally, name one thing you would have to add to the
stub to catch the fault in (b). (4 marks)

---

## Question 3, A network trained by hand (14 marks)

The network below has two inputs, two hidden nodes and one output node. Every node uses ReLU, and
the learning rate is $LR = 0.1$.

**Hidden layer:** $b_1 = 0.3$, $b_2 = -0.2$, with $w_1 = 0.4$ and $w_2 = -0.5$ feeding node 1 from
$x_1$ and $x_2$, and $w_3 = 0.2$ and $w_4 = 0.3$ feeding node 2 from $x_1$ and $x_2$.

**Output layer:** $b_3 = 0.1$, with $w_5 = 0.6$ from $y_1$ and $w_6 = -0.7$ from $y_2$.

Training set: $X_1X_2 = 1\ 1$, $Y = 1$.

**(a)** Perform feedforward. Give the weighted sum $s$ and the output $y$ for all three nodes.
(3 marks)

**(b)** Perform backpropagation. Give $\delta_3$, $e_3$, $\delta_1$, $\delta_2$, $e_1$ and $e_2$.
State why the output layer has to be done before the hidden layer. (3 marks)

**(c)** Perform optimization. Give all three updated biases and all six updated weights. (4 marks)

**(d)** Feed the same input forward again with the updated parameters. Give the new output and the
new deviation, and state by how much the deviation shrank. (2 marks)

**(e)** The **original** parameters are now used to train on a different set: $X_1X_2 = 0\ 0$,
$Y = 0$. Show that node 2's bias and both of its weights leave this step completely unchanged, name
the phenomenon, and name the activation function the course recommends as a countermeasure.
(2 marks)

---

## Question 4, The activation function, and the argument it is given (10 marks)

**(a)** `ml::actFuncOutput()` and `ml::actFuncDelta()` each handle exactly three cases. Give the
output and the derivative for all three, and state why `ActFunc::None` is the `default` case of the
switch rather than an error case. (3 marks)

**(b)** `actFuncDelta()` must be given the weighted sum $s$, not the output $y$. A `Tanh` node has
$s = 1.5$.

Compute the derivative the correct way and the derivative the wrong way, and give the ratio between
them. Then show that for a `Relu` node the two ways agree at $s = 1.5$ and at $s = -0.8$, and state
what that agreement costs a test suite. (4 marks)

**(c)** `ml::randomStartVal()` returns values in the range $(0, 1)$ only. Explain how that interacts
with ReLU to leave a hidden node unable to learn ever again, and state the two countermeasures the
course names. (3 marks)

---

## Question 5, The dense layer, written out (12 marks)

**(a)** Write `Dense::feedforward()`. Include the input validation.

One member variable is written by this method, is needed later, and is returned by no getter on the
interface. Name it and state what it is for. (4 marks)

**(b)** Write the hidden-layer overload, `Dense::backpropagate(const Interface& nextLayer)`.

The next layer's weight matrix is read as `nextLayer.weights()[j][i]`. State what $j$ ranges over
and what $i$ ranges over, and state what the layer would compute if the two indices were swapped.
(4 marks)

**(c)** Write `Dense::optimize()`.

Then state which vector `Shallow::train()` must pass as `input` when it optimizes the **output**
layer, and state what would go wrong if it passed the network's own training input instead.
(4 marks)

---

## Question 6, Why an image is not just a long vector (10 marks)

**(a)** Give the three reasons the course sets out for why a network built only from dense layers
handles images badly. (3 marks)

**(b)** A 1024 x 1024 greyscale image is flattened and fed into a single hidden layer of 1000 nodes.

Compute the number of trainable parameters in that one layer, and its memory footprint stored as
`double`. Give the footprint in bytes and in GB. (3 marks)

**(c)** Consider a 64 x 64 feature map. A conv layer with one 3 x 3 kernel and same-size padding
produces a 64 x 64 output from it. A dense layer producing the same 64 x 64 outputs from the same
64 x 64 inputs is the alternative.

Compute both parameter counts and the ratio between them.

Conv adds two constraints to a dense layer. Name both, then split the ratio you computed into the
factor due to each. (4 marks)

---

## Question 7, A convolutional layer, traced by hand (14 marks)

A conv layer takes a 4 x 4 input, has one 2 x 2 kernel, stride 1, zero-padding with
`pad = kernelSize / 2`, and `ActFunc::Relu`. It is followed by a max pooling layer (pool size 2,
stride 2) and a flatten layer. The learning rate is 0.01.

**Input image** (a hand-drawn 7):

```text
1 1 1 1
0 0 0 1
0 0 1 0
0 1 0 0
```

**Kernel:**

```text
0.5   1.0
1.0  -0.5
```

**Bias:** $-0.2$

**(a)** Give the padded input, then compute the conv layer's pre-activation matrix and its output.
(4 marks)

**(b)** Give the max pooling layer's output and the flatten layer's output. For each pooled value,
name the position in the conv output it came from.

Two of the four pooling blocks need the tie-breaking rule. Name them and state the rule. (3 marks)

**(c)** The dense layer sends back the gradients $[4, -2, 6, 8]$. Reshape them, route them back
through the pooling layer, and give the 4 x 4 gradient matrix that arrives at the conv layer's
output. (3 marks)

**(d)** Compute the bias gradient and all four kernel gradients, and give the updated kernel and
bias.

One of the four gradients from (c) contributes nothing whatsoever to any of those five numbers.
Identify it and give the reason. (4 marks)

---

## Question 8, Max pooling, and the value it forgets (8 marks)

A max pooling layer (pool size 2, stride 2) receives the following 4 x 4 feature map:

```text
 0.6   0.2   0.3   0.3
 0.1   0.4   0.1  -0.5
 0.8   0.2  -0.1  -0.6
 0.5   0.9  -0.4  -0.3
```

**(a)** Give the layer's output and, for each value, the position it came from. One block contains a
tie; state which, and which position wins. (3 marks)

**(b)** The gradients $\begin{bmatrix} 5 & -3 \\ 2 & 7\end{bmatrix}$ arrive from the layer in front.
Give the 4 x 4 gradient matrix the layer passes back, on the first call after construction.
(2 marks)

**(c)** A student writes `double maxValue{0.0};` before the search, leaves `maxRow` and `maxCol`
default-initialized, and ends the block with
`myInputGradients[maxRow][maxCol] = outputGradients[i][j];`.

Give the 4 x 4 matrix this produces for the same data. Name the **two** distinct things that go
wrong with it, and state why a test suite whose feature maps are all non-negative passes anyway.
(3 marks)

---

## Question 9, The finished network, and the target it has to run on (8 marks)

A CNN is built as follows:

```text
input (16x16) -> conv (one 5x5 kernel, stride 1, same padding, ReLU)
              -> max pooling (4x4, stride 4)
              -> flatten
              -> dense (4 nodes)
```

**(a)** Give the shape of the data after each of the four layers. (2 marks)

**(b)** Give the trainable-parameter count for each of the four layers and the total for the
network. Two of the four contribute nothing; name them and state why. (3 marks)

**(c)** A different network contains a dense layer of 128 nodes over 256 inputs.

Compute its trainable-parameter count, then its footprint stored as `double`, as `float`, and as
`int8`. State which of the three fits into a 256 KB flash budget, and show the comparison in bytes
rather than in KB. (3 marks)

---
