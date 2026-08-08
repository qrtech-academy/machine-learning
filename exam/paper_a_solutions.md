# Paper A, Solutions

Marks are shown per part. Method carries them: a correct rule with an arithmetic slip in it is worth
more than a correct number with no working, and later parts consume earlier answers, so an error
should be followed through rather than penalised twice.

Code answers are marked on the algorithm, not on the syntax. A candidate who writes correct
pseudocode loses nothing.

---

## Question 1, Learning the rules instead of writing them (12 marks)

### (a) 3 marks

**Traditional programming** is given the **input** and the **rules**, and produces the **output**.
The rules are written by hand: the course's ventilation example is a chain of `if`/`else if` on a
temperature reading.

**Machine learning** is given the **input** and the **output**, and produces the **rules**. The same
ventilation behaviour becomes a table of temperatures paired with fan settings, and the machine
finds the relationship itself.
*(1 mark for both arrangements)*

**The paradigm** is **symbolic AI**, the dominant approach before machine learning.
*(1 mark)*

**Where it worked and where it did not.** It worked on complex but *well-defined* tasks: chess is
the course's example, where the legal moves and the goal can be stated exactly. It failed on
computer vision, speech recognition and language translation.

**The property that separates them.** How widely the input can vary. Chess has a finite, exactly
describable set of positions; two photographs of the same object share almost no pixel values. A
task whose input varies without bound cannot be covered by an enumeration of conditions, however
long, because there is no finite enumeration to write.
*(1 mark)*

### (b) 4 marks

$y_p = kx + m$, $\delta = y_{ref} - y_p$, $\Delta e = \delta \cdot LR$, then $m \mathrel{+}= \Delta e$
and $k \mathrel{+}= \Delta e \cdot x$.

**Set 1** ($x = 0$, $y_{ref} = -1$):

$$y_p = 0 \cdot 0 + 0 = 0, \quad \delta = -1 - 0 = -1, \quad \Delta e = -0.1$$
$$m = 0 + (-0.1) = -0.1, \qquad k = 0 + (-0.1)(0) = 0$$

**Set 2** ($x = 1$, $y_{ref} = 3$):

$$y_p = 0 \cdot 1 + (-0.1) = -0.1, \quad \delta = 3 - (-0.1) = 3.1, \quad \Delta e = 0.31$$
$$m = -0.1 + 0.31 = 0.21, \qquad k = 0 + 0.31(1) = 0.31$$

**Set 3** ($x = 2$, $y_{ref} = 7$):

$$y_p = 0.31(2) + 0.21 = 0.83, \quad \delta = 7 - 0.83 = 6.17, \quad \Delta e = 0.617$$
$$m = 0.21 + 0.617 = \mathbf{0.827}, \qquad k = 0.31 + 0.617(2) = \mathbf{1.544}$$

*(1 mark per set, 1 mark for the correct rules stated or clearly used, including $k$ not moving on
set 1 because $x = 0$)*

### (c) 2 marks

$$y_p(5) = 1.544(5) + 0.827 = \mathbf{8.547} \qquad \text{against } y_{ref} = 19$$

*(1 mark)*

**Why this epoch achieved so much less.** The deviations were $-1$, $3.1$, $6.17$: they **grew** in
magnitude across the whole epoch, where L01's example saw its deviation turn over and start
shrinking by the fourth set. Two things cause it.

The target slope is 4 rather than 2, so $k$ has twice as far to travel and every prediction lags
further behind. And the target bias is $-1$ while every deviation after the first is positive, so
sets 2 and 3 drive $m$ *upwards*, from $-0.1$ to $+0.827$, away from where it has to end up. The
model can only reach $m = -1$ after $k$ has overshot far enough for the deviations to turn negative,
and one epoch of three sets is nowhere near enough for that.
*(1 mark; credit any answer that quotes the growing $\delta$ sequence and identifies either the
larger slope or the wrong-direction bias movement)*

### (d) 3 marks

The special case: `if (0 == input) { myBias = output; return; }`.

**Set 1** ($x = 0$): $m$ is assigned $-1$ directly. $k$ is untouched, so $k = 0$.
*(1 mark)*

**Set 2** ($x = 1$): $y_p = 0(1) + (-1) = -1$, $\delta = 4$, $\Delta e = 0.4$, so $m = -0.6$ and
$k = 0.4$.

**Set 3** ($x = 2$): $y_p = 0.4(2) + (-0.6) = 0.2$, $\delta = 6.8$, $\Delta e = 0.68$, so
$m = \mathbf{0.08}$ and $k = \mathbf{1.76}$.
*(1 mark)*

Against (b)'s $k = 1.544$, $m = 0.827$: the special case ends with a larger weight and a bias still
near zero rather than well above it, because it never let the first set push the bias the short way
and then spent the rest of the epoch undoing it. Both models are still far from $(4, -1)$.

**When it is a bad idea.** When the training data is noisy, or contains more than one set at
$x = 0$. The bias is a property of the whole data set; the special case snaps it onto a single
sample and discards everything every other sample had contributed to it. With more than one $x = 0$
sample the final bias is simply whichever of them was visited last, and since L02 randomizes the
order, the trained model then depends on the shuffle.
*(1 mark; any one of: noise, duplicate $x = 0$ samples, order dependence)*

---

## Question 2, The training loop that trains nothing (12 marks)

### (a) 3 marks

**Fault 1: `randomizeTrainOrder()` is called once, outside the epoch loop.** It belongs at the top of
the epoch body, so a new order is drawn each epoch. As written the order is drawn once and reused
for the entire run.
*(1 mark)*

**Fault 2: the inner loop indexes with the counter, not with the order.** It should read
`myTrainInput[myTrainOrder[x]]` (and likewise for `myTrainOutput`). As written, `myTrainOrder` is
used only for its `size()`; its contents are never read at all.
*(1 mark)*

**What actually happens.** The sets are visited in strict storage order $0, 1, 2, \ldots$, the same
every epoch. Fault 2 makes fault 1 invisible: shuffling a vector nobody reads changes nothing, so
fixing only fault 1 changes nothing either. Both have to be fixed for the shuffle to have any
effect.
*(1 mark)*

### (b) 3 marks

**The correct order** is output layer first, hidden layer second:

```cpp
myOutputLayer.backpropagate(myTrainOutput[x]);
myHiddenLayer.backpropagate(myOutputLayer);
```

*(1 mark)*

**What the hidden layer reads as written.** `myHiddenLayer.backpropagate(myOutputLayer)` computes
$\delta_i = \sum_j \Delta e_j \cdot w_{ji}$ from `myOutputLayer.error()` and
`myOutputLayer.weights()`. Called first, it reads the error vector the output layer computed on the
**previous training set**, against a different reference value, and from a different feedforward
pass than the one that just ran. The blame is real, but it belongs to another example.
*(1 mark)*

**The very first set of the very first epoch.** The output layer has never backpropagated, so its
error vector still holds the zeros it was constructed with. Every hidden $\delta$ is therefore $0$,
every hidden $\Delta e$ is $0$, and `myHiddenLayer.optimize()` adds zero to every bias and every
weight. The hidden layer's first update is a no-op.
*(1 mark)*

### (c) 2 marks

All four report **failure of a range check** through the return value: `false` when the dimensions
do not match, and additionally, for `optimize()`, when the learning rate lies outside $(0.0, 1.0)$.
*(1 mark)*

The failure they exist to catch is training data whose shape does not match the layers, namely a
training input row that is not `weightCount()` wide, or a training output row that is not
`nodeCount()` long.
(The layer-to-layer mismatch is already caught: `Shallow`'s constructor terminates when the output
layer's weight count does not equal the hidden layer's node count.)

With the values discarded the loop runs to completion regardless, the layers are left holding
whatever they held before, and the function **returns `true`**, reporting a successful training run
that never happened.
*(1 mark)*

### (d) 4 marks

**`feedforwardCount()`** makes the *number* of feedforward passes testable: one per training set per
epoch. Nothing else can establish it. A loop that runs a single pass instead of every epoch lines up
dimensionally and returns `true` exactly as a correct one does, so without a tally the two are
indistinguishable.

**`setOutput()`** makes it testable that `predict()` reads the output layer **live** rather than
returning a stored copy. The network holds its layers by reference, so changing a layer's output
after the network was built must change what the network predicts; if it does not, the network kept
a stale copy.
*(1 mark for both)*

**Fault (a): not caught.** The count is one feedforward per set per epoch whatever order the sets are
visited in, and neither method records the *arguments* a call was made with. A count cannot see an
order.
*(1 mark)*

**Fault (b): not caught.** `backpropagate()` is called the right number of times with the right
arguments; only the sequence is wrong. Against the stub it is doubly invisible, because the stub's
error stays at zero no matter when it is called.
*(1 mark)*

**Fault (c): not caught.** With correctly sized training data every call returns `true`, so
discarding the values changes nothing a passing test can observe. It only matters on data that is
wrongly shaped, which a green test run does not supply.

**What the stub would need to catch (b).** A record of the *order* in which its methods were called,
not just how often; a call log the test can read back, or a per-method sequence number, so a test
can assert that the output layer's `backpropagate()` ran before the hidden layer's. Equivalently,
letting the stub report the arguments it was handed would expose fault (a) as well.
*(1 mark)*

---

## Question 3, A network trained by hand (14 marks)

### (a) 3 marks

$x_1 = 1$, $x_2 = 1$.

$$s_1 = b_1 + x_1w_1 + x_2w_2 = 0.3 + 0.4 - 0.5 = 0.2 \Rightarrow y_1 = \sigma(0.2) = \mathbf{0.2}$$
$$s_2 = b_2 + x_1w_3 + x_2w_4 = -0.2 + 0.2 + 0.3 = 0.3 \Rightarrow y_2 = \sigma(0.3) = \mathbf{0.3}$$
$$s_3 = b_3 + y_1w_5 + y_2w_6 = 0.1 + 0.2(0.6) + 0.3(-0.7) = 0.1 + 0.12 - 0.21 = 0.01$$
$$y_3 = \sigma(0.01) = \mathbf{0.01}$$

*(1 mark per node)*

### (b) 3 marks

$$\delta_3 = Y - y_3 = 1 - 0.01 = 0.99, \qquad e_3 = \delta_3 \cdot \sigma'(s_3) = 0.99 \cdot 1 = \mathbf{0.99}$$
$$\delta_1 = e_3 w_5 = 0.99(0.6) = 0.594, \qquad e_1 = 0.594 \cdot \sigma'(s_1) = 0.594 \cdot 1 = \mathbf{0.594}$$
$$\delta_2 = e_3 w_6 = 0.99(-0.7) = -0.693, \qquad e_2 = -0.693 \cdot \sigma'(s_2) = \mathbf{-0.693}$$

All three weighted sums are positive, so every ReLU derivative is 1.
*(2 marks)*

**Why the output layer first.** A hidden node's deviation is a sum of the *next* layer's computed
errors, weighted by the connections leaving it: $\delta_i = \sum_j e_j w_{ji}$. Those errors do not
exist until the output layer has compared its own output against the reference value. Reversing the
order does not fail; it silently uses whatever the output layer's error vector happened to hold from
the previous example.
*(1 mark)*

### (c) 4 marks

**Output layer:** $\Delta c_3 = e_3 \cdot LR = 0.99(0.1) = 0.099$

$$b_3 = 0.1 + 0.099 = \mathbf{0.199}$$
$$w_5 = 0.6 + 0.099(y_1) = 0.6 + 0.099(0.2) = \mathbf{0.6198}$$
$$w_6 = -0.7 + 0.099(y_2) = -0.7 + 0.099(0.3) = \mathbf{-0.6703}$$

*(2 marks)*

**Hidden layer:** $\Delta c_1 = 0.594(0.1) = 0.0594$, $\Delta c_2 = -0.693(0.1) = -0.0693$

$$b_1 = 0.3 + 0.0594 = \mathbf{0.3594}, \quad w_1 = 0.4 + 0.0594(1) = \mathbf{0.4594}, \quad w_2 = -0.5 + 0.0594(1) = \mathbf{-0.4406}$$
$$b_2 = -0.2 - 0.0693 = \mathbf{-0.2693}, \quad w_3 = 0.2 - 0.0693(1) = \mathbf{0.1307}, \quad w_4 = 0.3 - 0.0693(1) = \mathbf{0.2307}$$

*(2 marks)*

Note that the weights are scaled by the *input to that layer*, $y_1$ and $y_2$ for the output
layer, $x_1$ and $x_2$ for the hidden layer, while the biases are not. A candidate who used $x$ for
the output layer's weights has made the mistake Question 5(c) asks about.

### (d) 2 marks

$$s_1 = 0.3594 + 0.4594 - 0.4406 = 0.3782 \Rightarrow y_1 = 0.3782$$
$$s_2 = -0.2693 + 0.1307 + 0.2307 = 0.0921 \Rightarrow y_2 = 0.0921$$
$$s_3 = 0.199 + 0.3782(0.6198) + 0.0921(-0.6703) = 0.199 + 0.234408 - 0.061735 = 0.371673$$
$$Y = y_3 \approx \mathbf{0.3717}, \qquad \delta_3 = 1 - 0.3717 = \mathbf{0.6283}$$

*(1 mark)*

The deviation fell from $0.99$ to $0.6283$, a reduction of $0.3617$, or about 37%. One step of one
training set moved the output roughly a third of the way to its target, which is what a learning
rate of 0.1 on a deviation this large looks like.
*(1 mark)*

### (e) 2 marks

With the **original** parameters and $x_1 = x_2 = 0$:

$$s_1 = 0.3 \Rightarrow y_1 = 0.3, \qquad s_2 = -0.2 \Rightarrow y_2 = \sigma(-0.2) = 0$$
$$s_3 = 0.1 + 0.3(0.6) + 0(-0.7) = 0.28 \Rightarrow y_3 = 0.28$$
$$\delta_3 = 0 - 0.28 = -0.28, \qquad e_3 = -0.28$$
$$\delta_2 = e_3 w_6 = -0.28(-0.7) = 0.196, \qquad e_2 = 0.196 \cdot \sigma'(s_2) = 0.196 \cdot 0 = \mathbf{0}$$

Therefore $\Delta c_2 = 0$, and $b_2$, $w_3$ and $w_4$ all keep their original values of $-0.2$,
$0.2$ and $0.3$.
*(1 mark)*

Note that the blame arriving at node 2 was not small, $\delta_2 = 0.196$; it was **annihilated** by
the derivative. And the bias is the tell: $w_3$ and $w_4$ would have stayed put anyway, since both
are multiplied by an input of zero, but $b_2$ has nothing to hide behind and it does not move
either.

**The phenomenon** is the **dying ReLU** problem: once a node's weighted sum is negative, its
derivative is exactly zero, its error is zero, its parameters stop updating, and nothing can drive
the sum back up. The node is dead for the rest of training.

**The countermeasure** the course names is **Leaky ReLU**, which passes a weak signal below zero
($y = ks$ with $k \approx 0.01$) and so keeps the derivative non-zero everywhere.
*(1 mark)*

---

## Question 4, The activation function, and the argument it is given (10 marks)

### (a) 3 marks

| `ActFunc` | `actFuncOutput(f, v)`             | `actFuncDelta(f, v)`         |
| --------- | --------------------------------- | ---------------------------- |
| `Relu`    | `v` if `v > 0.0`, otherwise `0.0` | `1.0` if `v > 0.0`, else `0.0` |
| `Tanh`    | `std::tanh(v)`                    | `1.0 - tanh(v) * tanh(v)`    |
| `None`    | `v` (identity)                    | `1.0`                        |

*(2 marks)*

**Why `None` is the `default` case.** `ActFunc` has exactly three enumerators. `Relu` and `Tanh` are
handled explicitly, so the only case that can reach `default` is `None`; there is no
"cannot happen" branch to write, and inventing one would add an error path that is unreachable and
therefore untestable. Putting `None` there also satisfies the compiler that the function returns on
every path without a redundant trailing `return`.
*(1 mark)*

### (b) 4 marks

**Tanh at $s = 1.5$.**

Correct, the derivative evaluated at the activation's *input*:

$$\sigma'(s) = 1 - \tanh^2(1.5) = 1 - 0.905148^2 = 1 - 0.819293 = \mathbf{0.180707}$$

Wrong, the derivative evaluated at the activation's *output*, $y = \tanh(1.5) = 0.905148$:

$$1 - \tanh^2(0.905148) = 1 - 0.718796^2 = 1 - 0.516668 = \mathbf{0.483332}$$

$$\text{ratio} = \frac{0.483332}{0.180707} = \mathbf{2.675}$$

The wrong value is 2.7x too large, so every parameter this node feeds moves 2.7x too far on every
step. The error grows with $\lvert s \rvert$: the true derivative collapses towards zero as the node
saturates and the mistaken one does not, so the bug does its worst damage exactly where tanh is
supposed to be applying the brakes.
*(2 marks)*

**ReLU.** At $s = 1.5$: $y = 1.5$. From $s$: $1.5 > 0 \Rightarrow 1$. From $y$: $1.5 > 0 \Rightarrow
1$. At $s = -0.8$: $y = 0$. From $s$: $-0.8 > 0$ is false $\Rightarrow 0$. From $y$: $0 > 0$ is
false $\Rightarrow 0$. The two agree in both cases.

They agree in *every* case, and not by luck: ReLU's derivative tests only the predicate
"is this value positive", and $\max(0, s)$ is positive exactly when $s$ is.
*(1 mark)*

**What that costs a test suite.** A suite that only ever exercises `Relu` cannot detect the bug at
all; every assertion passes, and the layer is wrong. Only a `Tanh` test separates the two, which is
what `BackpropagateUsesPreActivationDerivative` in the L05 suite exists to do. The general lesson is
that a test suite has to include the case that can *distinguish* two implementations, not merely a
case each of them handles.
*(1 mark)*

### (c) 3 marks

`randomStartVal()` divides `std::rand()` by `RAND_MAX`, so it returns a value in $(0, 1)$ and never
a negative one. Every bias and every weight therefore starts positive, and every node starts on
ReLU's positive side.
*(1 mark)*

If training then drives some node's weighted sum negative for **every** input in the training data,
$\sigma'(s) = 0$ for that node on every set. Its $\Delta e$ is zero, so its bias and weights never
change again, and because they never change, the sum can never come back above zero. The node is
permanently stuck, and nothing upstream or downstream can rescue it, because it also stops passing
any blame backwards.
*(1 mark)*

**The countermeasures the course names:**

* **Leaky ReLU**, which keeps a small non-zero derivative below zero, so a node driven negative can
  still recover.
* **`Tanh`**, whose derivative is non-zero everywhere. This is what the L05 test suite trains with,
  precisely so the suite does not fail at random.

Also creditable: widening the hidden layer, so that losing a node is survivable. L05 records that
with only two hidden nodes and ReLU, roughly one run in four fails to learn XOR at all, however long
it trains.
*(1 mark)*

---

## Question 5, The dense layer, written out (12 marks)

### (a) 4 marks

```cpp
bool Dense::feedforward(const Matrix1d& input) noexcept
{
    if (input.size() != weightCount())
    {
        std::fprintf(stderr, "Input dimension mismatch: expected %zu, actual: %zu!\n",
            weightCount(), input.size());
        return false;
    }
    for (std::size_t i{}; i < nodeCount(); ++i)
    {
        auto sum{myBias[i]};

        for (std::size_t j{}; j < weightCount(); ++j)
        {
            sum += myWeights[i][j] * input[j];
        }
        myPreActivationOutput[i] = sum;
        myOutput[i]              = actFuncOutput(myActFunc, sum);
    }
    return true;
}
```

*(3 marks: 1 for the validation returning `false` without computing anything, 1 for the weighted sum
starting at the bias, 1 for applying the activation function to it)*

**The member variable is `myPreActivationOutput`.** It holds $s$, the weighted sum for each node
*before* the activation function was applied. Both overloads of `backpropagate()` need it, because
the activation's derivative has to be evaluated at the activation's input, not at its output
(Question 4b). It is not on the interface because no other layer ever reads it: it is this layer's
own bookkeeping, carried from the forward pass to the backward one.
*(1 mark)*

### (b) 4 marks

```cpp
bool Dense::backpropagate(const Interface& nextLayer) noexcept
{
    if (nextLayer.weightCount() != nodeCount())
    {
        std::fprintf(stderr, "Layer dimension mismatch: expected %zu, actual: %zu!\n",
            nodeCount(), nextLayer.weightCount());
        return false;
    }
    for (std::size_t i{}; i < nodeCount(); ++i)
    {
        double err{};

        for (std::size_t j{}; j < nextLayer.nodeCount(); ++j)
        {
            err += nextLayer.error()[j] * nextLayer.weights()[j][i];
        }
        myError[i] = err * actFuncDelta(myActFunc, myPreActivationOutput[i]);
    }
    return true;
}
```

*(3 marks: 1 for the validation, 1 for the sum over the next layer's nodes, 1 for multiplying by the
derivative at `myPreActivationOutput[i]`)*

**The indices.** $j$ ranges over the **next** layer's nodes, $0 \ldots$
`nextLayer.nodeCount() - 1`. $i$ ranges over **this** layer's nodes, $0 \ldots$ `nodeCount() - 1`,
and doubles as the weight index inside each next-layer node, which works precisely because the next
layer has one weight per node in this layer, the condition the validation just checked.
`weights()[j][i]` therefore reads "next-layer node $j$'s weight on this layer's node $i$": the single
connection along which node $i$'s output travelled forward, and along which its share of the blame
must travel back.

**Swapped.** `weights()[i][j]` reads a different entry entirely, the weight matrix transposed. If
the two layers happen to be the same size it still compiles, still stays in bounds and still returns
`true`, while handing every node the blame belonging to some other node; the network then trains on
systematically misattributed gradients and simply never converges. If the sizes differ it indexes
out of bounds. The silent version is the more dangerous of the two.
*(1 mark)*

### (c) 4 marks

```cpp
bool Dense::optimize(const Matrix1d& input, const double learningRate) noexcept
{
    if ((0.0 >= learningRate) || (1.0 <= learningRate))
    {
        std::fprintf(stderr, "Invalid learning rate %g!\n", learningRate);
        return false;
    }
    if (input.size() != weightCount())
    {
        std::fprintf(stderr, "Input dimension mismatch: expected %zu, actual: %zu!\n",
            weightCount(), input.size());
        return false;
    }
    for (std::size_t i{}; i < nodeCount(); ++i)
    {
        myBias[i] += myError[i] * learningRate;

        for (std::size_t j{}; j < weightCount(); ++j)
        {
            myWeights[i][j] += myError[i] * learningRate * input[j];
        }
    }
    return true;
}
```

*(2 marks: 1 for both checks, 1 for the bias update carrying no `input[j]` factor while the weight
update does)*

**What `Shallow::train()` must pass to the output layer.** `myHiddenLayer.output()`, the hidden
layer's output vector, which is exactly what the output layer consumed during feedforward.
*(1 mark)*

**If it passed the network's training input instead.** Usually nothing at all happens: the output
layer has one weight per hidden node, not one per network input, so `input.size() != weightCount()`
and the call returns `false` without touching a parameter. Training then stops with `false` if the
return value is checked, and silently stops updating the output layer if it is not.

The dangerous case is a hidden layer that happens to have as many nodes as the network has inputs.
The check then passes and every weight is scaled by the wrong number. $w_{ij}$ is supposed to move
in proportion to the value that flowed along it, and it would instead move in proportion to an
unrelated one, so the update stops being the gradient with respect to that weight. The network fails
to converge while every call reports success.
*(1 mark)*

---

## Question 6, Why an image is not just a long vector (10 marks)

### (a) 3 marks

**1. Spatial structure is lost.** An image is two- or three-dimensional, and has to be flattened into
a one-dimensional vector before a dense layer can take it. Which pixels sat next to which is
discarded, and no amount of training recovers it.

**2. The parameter count explodes.** Every pixel gets its own weight into every node, so the layer
grows as (pixels x nodes). Part (b) puts a number on it.

**3. Generalization gets worse.** Each pixel is treated as a completely separate input, so the
network has no way of knowing that the same pattern shifted a few pixels across is the same pattern.
It needs far more training data and still transfers badly to new images.

*(1 mark each)*

### (b) 3 marks

$$1024 \times 1024 = 1{,}048{,}576 \text{ inputs}$$

$$\text{weights} = 1{,}048{,}576 \times 1000 = 1{,}048{,}576{,}000, \qquad \text{biases} = 1000$$

$$\text{total} = \mathbf{1{,}048{,}577{,}000 \text{ trainable values}}$$

*(2 marks)*

As `double`, at 8 bytes each:

$$1{,}048{,}577{,}000 \times 8 = 8{,}388{,}616{,}000 \text{ bytes} \approx \mathbf{8.39\ GB}$$

And that is **one hidden layer**. A real network has several.
*(1 mark)*

### (c) 4 marks

**Conv layer:** one 3 x 3 kernel and one bias.

$$3 \times 3 + 1 = \mathbf{10}$$

**Dense layer:** $64 \times 64 = 4096$ inputs and 4096 outputs.

$$4096 \times 4096 + 4096 = 16{,}777{,}216 + 4096 = \mathbf{16{,}781{,}312}$$

$$\text{ratio} = \frac{16{,}781{,}312}{10} \approx \mathbf{1{,}678{,}131 : 1}$$

*(2 marks)*

**The two constraints.**

* **Locality.** Each output looks at a small window of the input rather than at all of it.
* **Weight sharing.** The same kernel is reused at every position, instead of each output/input pair
  owning its own weight.

*(1 mark)*

**Splitting the ratio.** Apply locality alone: each of the 4096 outputs keeps its own private 3 x 3
window of weights plus a bias.

$$4096 \times 9 + 4096 = 40{,}960 \qquad \frac{16{,}781{,}312}{40{,}960} \approx 410\text{x}$$

Now add weight sharing: those 4096 private kernels collapse into one.

$$\frac{40{,}960}{10} = 4096\text{x}$$

$$410 \times 4096 \approx 1{,}678{,}000 \text{, the whole ratio.}$$

Weight sharing is the larger of the two factors, by an order of magnitude. Locality is what makes the
layer look at neighbourhoods; sharing is what makes it cheap.
*(1 mark)*

---

## Question 7, A convolutional layer, traced by hand (14 marks)

### (a) 4 marks

$\text{pad} = 2 / 2 = 1$, so one zero on every side and the padded input is 6 x 6:

```text
0 0 0 0 0 0
0 1 1 1 1 0
0 0 0 0 1 0
0 0 0 1 0 0
0 0 1 0 0 0
0 0 0 0 0 0
```

*(1 mark)*

Each output cell $(i, j)$ is
$s = b + \sum_{a,b} K[a][b] \cdot P[i+a][j+b]$, over the 2 x 2 window whose top-left corner is
$(i, j)$ in the padded matrix. Two worked cells:

$$s(0,0) = 0.5(0) + 1.0(0) + 1.0(0) - 0.5(1) - 0.2 = -0.70$$
$$s(1,1) = 0.5(1) + 1.0(1) + 1.0(0) - 0.5(0) - 0.2 = 1.30$$

**Pre-activation:**

```text
-0.70   0.30   0.30   0.30
 0.80   1.30   1.30   0.80
-0.20  -0.20  -0.70   1.80
-0.20  -0.70   1.80   0.30
```

*(2 marks)*

**Output** (ReLU: every negative entry becomes 0):

```text
0.00   0.30   0.30   0.30
0.80   1.30   1.30   0.80
0.00   0.00   0.00   1.80
0.00   0.00   1.80   0.30
```

*(1 mark)*

### (b) 3 marks

| Block        | Values                   | Max  | Position |
| ------------ | ------------------------ | ---- | -------- |
| top left     | 0.00, 0.30, 0.80, 1.30   | 1.30 | (1, 1)   |
| top right    | 0.30, 0.30, 1.30, 0.80   | 1.30 | (1, 2)   |
| bottom left  | 0.00, 0.00, 0.00, 0.00   | 0.00 | (2, 0)   |
| bottom right | 0.00, 1.80, 1.80, 0.30   | 1.80 | (2, 3)   |

**Max pooling output:**

```text
1.3   1.3
0.0   1.8
```

**Flatten output:** $[1.3,\ 1.3,\ 0.0,\ 1.8]$
*(2 marks)*

**The tie blocks are the bottom-left and the bottom-right.** The bottom-left block is four identical
zeros, and the bottom-right block holds 1.80 twice, at $(2,3)$ and $(3,2)$.

**The rule:** on a tie, the first occurrence in row-major order wins. What matters is not which cell
is chosen but that `feedforward()` and `backpropagate()` choose the *same* one. If they disagree the
gradient is routed to a cell that did not produce the output, and the conv layer behind learns from
the wrong pixel.
*(1 mark)*

### (c) 3 marks

Reshaped row-major into the pooling output's 2 x 2 shape:

$$\begin{bmatrix} 4 & -2 \\ 6 & 8 \end{bmatrix}$$

*(1 mark)*

Each gradient goes to the position its block's maximum came from, and every other position gets 0:

```text
0    0    0   0
0    4   -2   0
6    0    0   0
0    0    0   0
```

*(2 marks: 4 to (1,1), -2 to (1,2), 6 to (2,0), 8 to (2,3))*

### (d) 4 marks

The conv layer first multiplies each incoming gradient by $\sigma'(s)$ at that position, using the
pre-activation matrix from (a):

| Position | $s$   | $\sigma'(s)$ | gradient in | `outGrad` |
| -------- | ----- | ------------ | ----------- | --------- |
| (1, 1)   | 1.30  | 1            | 4           | 4         |
| (1, 2)   | 1.30  | 1            | -2          | -2        |
| (2, 0)   | -0.20 | **0**        | 6           | **0**     |
| (2, 3)   | 1.80  | 1            | 8           | 8         |

$$\text{biasGradient} = 4 - 2 + 0 + 8 = \mathbf{10}$$

*(1 mark)*

$$\text{kernelGradient}[a][b] = \sum_{i,j} P[i+a][j+b] \cdot \text{outGrad}(i,j)$$

Three positions contribute: $(1,1)$ with 4, $(1,2)$ with $-2$ and $(2,3)$ with 8. Each kernel entry
therefore has three terms, one per contributing position, reading the padded input at the offset
that entry sits at:

$$k_{00} = P[1][1](4) + P[1][2](-2) + P[2][3](8) = 1(4) + 1(-2) + 0(8) = \mathbf{2}$$
$$k_{01} = P[1][2](4) + P[1][3](-2) + P[2][4](8) = 1(4) + 1(-2) + 1(8) = \mathbf{10}$$
$$k_{10} = P[2][1](4) + P[2][2](-2) + P[3][3](8) = 0(4) + 0(-2) + 1(8) = \mathbf{8}$$
$$k_{11} = P[2][2](4) + P[2][3](-2) + P[3][4](8) = 0(4) + 0(-2) + 0(8) = \mathbf{0}$$

**Kernel gradients:**

```text
 2   10
 8    0
```

*(2 marks)*

**Updated parameters** at $LR = 0.01$:

$$\text{bias} = -0.2 + 10(0.01) = \mathbf{-0.10}$$

```text
0.52   1.10
1.08  -0.50
```

*(1 mark)*

**The gradient that contributes nothing is the 6.** Max pooling routed it to conv output position
$(2,0)$, whose pre-activation was $-0.20$. ReLU's derivative there is exactly zero, so `outGrad` at
that position is $0 \times 6 = 0$, and the gradient is annihilated before it reaches the bias
gradient, any kernel gradient, or the input gradients. It is worth seeing *why* that position was
chosen at all: its whole pooling block had been flattened to zero by ReLU, and a block ReLU has
flattened passes nothing back.

A candidate who instead answers "the $-2$, because $k_{11}$ came out as 0" has found a different
fact. $k_{11}$ is zero because the corresponding image pixels are zero, not because a gradient died.

---

## Question 8, Max pooling, and the value it forgets (8 marks)

### (a) 3 marks

| Block        | Values                     | Max   | Position |
| ------------ | -------------------------- | ----- | -------- |
| top left     | 0.6, 0.2, 0.1, 0.4         | 0.6   | (0, 0)   |
| top right    | 0.3, 0.3, 0.1, -0.5        | 0.3   | (0, 2)   |
| bottom left  | 0.8, 0.2, 0.5, 0.9         | 0.9   | (3, 1)   |
| bottom right | -0.1, -0.6, -0.4, -0.3     | -0.1  | (2, 2)   |

**Output:**

```text
 0.6    0.3
 0.9   -0.1
```

*(2 marks)*

**The tie is in the top-right block**, where 0.3 appears at both $(0,2)$ and $(0,3)$. The first
occurrence in row-major order wins, so $(0,2)$ is recorded.
*(1 mark)*

### (b) 2 marks

```text
 5    0   -3    0
 0    0    0    0
 0    0    7    0
 0    2    0    0
```

*(2 marks)*

### (c) 3 marks

With `maxValue` starting at 0.0 and `maxRow`/`maxCol` starting at 0:

* top left: 0.6 beats 0.0; $(0,0)$ receives 5.
* top right: 0.3 beats 0.0; $(0,2)$ receives -3.
* bottom left: 0.9 beats 0.0; $(3,1)$ receives 2.
* bottom right: every value is negative, so **nothing ever beats 0.0**. `maxRow` and `maxCol` keep
  their default value of 0, and 7 is written to $(0,0)$.

```text
 7    0   -3    0
 0    0    0    0
 0    0    0    0
 0    2    0    0
```

*(1 mark)*

**The two distinct things that go wrong:**

**1. The bottom-right block's gradient never reaches the bottom-right block.** Position $(2,2)$
produced the pooled output and receives nothing, so the conv layer behind gets no gradient at all for
that quarter of the image and never learns from it.

**2. The gradient is written into a different block, destroying what was there.** $(0,0)$ belongs to
the top-left block and had already, correctly, been given 5. The assignment (`=`, not `+=`)
overwrites it with 7. So one gradient is lost and another is fabricated at a position that did not
earn it, and the top-left block's legitimate gradient disappears as collateral damage.

*(1 mark; both halves needed)*

**Why an all-non-negative test suite passes.** After a ReLU conv layer, every value in the feature
map is $\geq 0$, so every block contains at least one value that beats the initial 0.0, the search
always finds a real maximum, and `maxRow`/`maxCol` are always assigned. The defect is invisible.

It only surfaces where the layer in front can produce negative values, which is exactly what
happens once the conv layer uses `Tanh`. The fix is to initialise the search from the block's own
first value rather than from 0.0. The same defect appears in `feedforward()`, where the L09 suite
catches it with `FeedforwardHandlesNegativeValues`; the backward half needs the same all-negative
input to be exposed.
*(1 mark)*

---

## Question 9, The finished network, and the target it has to run on (8 marks)

### (a) 2 marks

| After           | Shape   |
| --------------- | ------- |
| conv layer      | 16 x 16 |
| max pooling     | 4 x 4   |
| flatten         | 1 x 16  |
| dense (4 nodes) | 1 x 4   |

The conv layer preserves the size because it zero-pads; the pooling layer divides by the pool size,
$16 / 4 = 4$; the flatten layer produces $4 \times 4 = 16$ values.
*(2 marks)*

### (b) 3 marks

| Layer       | Trainable parameters              |
| ----------- | --------------------------------- |
| conv        | $5 \times 5 + 1 = \mathbf{26}$    |
| max pooling | $\mathbf{0}$                      |
| flatten     | $\mathbf{0}$                      |
| dense       | $4 \times 16 + 4 = \mathbf{68}$   |
| **total**   | $\mathbf{94}$                     |

*(2 marks)*

**The two that contribute nothing are max pooling and flatten**, and both are purely structural. Max
pooling selects a value that was already present and stores no weights of its own; flatten changes
the shape of the data and computes nothing at all. Neither has anything an `optimize()` could
adjust, which is why the course's `MaxPool::optimize()` is a genuine no-op rather than a
placeholder, and why `flatten_layer::Interface` declares no `optimize()` in the first place.
*(1 mark)*

### (c) 3 marks

$$128 \times 256 + 128 = 32{,}768 + 128 = \mathbf{32{,}896 \text{ values}}$$

| Type     | Bytes each | Total footprint            |
| -------- | ---------- | -------------------------- |
| `double` | 8          | 263,168 B (exactly 257 KB) |
| `float`  | 4          | 131,584 B (128.5 KB)       |
| `int8`   | 1          | 32,896 B (32.125 KB)       |

*(2 marks)*

The budget is $256 \times 1024 = 262{,}144$ bytes.

$$263{,}168 > 262{,}144 \Rightarrow \textbf{double does not fit}, \text{ by } 1024 \text{ bytes}$$
$$131{,}584 < 262{,}144 \Rightarrow \textbf{float fits}$$
$$32{,}896 < 262{,}144 \Rightarrow \textbf{int8 fits}$$

The comparison has to be made in bytes. "About 257 KB against a 256 KB budget" reads like a rounding
question and is in fact a miss; it is a miss of exactly one kilobyte, which is the kind of
margin that turns into a link failure rather than a warning.
*(1 mark)*

---
