# Paper B, Solutions

Marks are shown per part. Method carries them: a correct rule with an arithmetic slip in it is worth
more than a correct number with no working, and later parts consume earlier answers, so an error
should be followed through rather than penalized twice.

Code answers are marked on the algorithm, not on the syntax. A candidate who writes correct
pseudocode loses nothing.

---

## Question 1, The model, and the rate that destroys it (10 marks)

### (a) 2 marks

**Traditional programming** is given the **input** and the **rules** and produces the **output**. The
rules are written by hand.

**Machine learning** is given the **input** and the **output** and produces the **rules**. It is
handed pairs of examples and finds the relationship itself.
*(1 mark)*

**Why the traditional arrangement fails on images.** The input varies without bound. Two photographs
of the same object share almost no pixel values, so there is no finite set of conditions on pixel
values that recognizes the object; a rule that covers one image fails on the next. Symbolic AI could
handle chess, where every position can be described exactly, and could not handle vision, speech or
translation, where the input cannot be enumerated.
*(1 mark)*

### (b) 3 marks

$y_p = kx + m$, $\delta = y_{ref} - y_p$, $\Delta e = \delta \cdot LR$, then $m \mathrel{+}= \Delta e$
and $k \mathrel{+}= \Delta e \cdot x$, with $LR = 0.5$.

**Step 1** ($x = 3$, $y_{ref} = 7$):

$$y_p = 0, \quad \delta = 7, \quad \Delta e = 3.5, \quad m = 3.5, \quad k = 0 + 3.5(3) = 10.5$$

**Step 2** ($x = 4$, $y_{ref} = 9$):

$$y_p = 10.5(4) + 3.5 = 45.5, \quad \delta = 9 - 45.5 = -36.5, \quad \Delta e = -18.25$$
$$m = 3.5 - 18.25 = -14.75, \quad k = 10.5 - 18.25(4) = -62.5$$

**Step 3** ($x = 3$, $y_{ref} = 7$):

$$y_p = -62.5(3) - 14.75 = -202.25, \quad \delta = 7 + 202.25 = 209.25, \quad \Delta e = 104.625$$
$$m = -14.75 + 104.625 = \mathbf{89.875}, \quad k = -62.5 + 104.625(3) = \mathbf{251.375}$$

*(1 mark per step)*

$\lvert\delta\rvert$ runs $7 \rightarrow 36.5 \rightarrow 209.25$: it grows by roughly a factor of
5 to 6 every step, and alternates in sign. This is not slow convergence, it is divergence. The
parameters are further from $(2, 1)$ after three steps than they were at $(0, 0)$.

### (c) 3 marks

Let $p = kx + m$ and $\delta = y_{ref} - p$. One optimization step gives

$$m' = m + \delta\,LR, \qquad k' = k + \delta\,LR\,x$$

so the new prediction for the **same** input is

$$p' = k'x + m' = (k + \delta\,LR\,x)x + m + \delta\,LR = p + \delta\,LR\,(1 + x^2)$$

and therefore

$$\delta' = y_{ref} - p' = \delta - \delta\,LR\,(1 + x^2) = \boxed{\;\delta\left[1 - LR(1 + x^2)\right]\;}$$

*(2 marks)*

The deviation shrinks when $\lvert 1 - LR(1+x^2)\rvert < 1$, that is when
$0 < LR(1+x^2) < 2$:

$$LR < \frac{2}{1 + x^2}$$

At $x = 4$: $LR < 2/17 = \mathbf{0.1176}$.
*(1 mark)*

Note what the formula says: the bias contributes the 1 and the weight contributes the $x^2$, because
the weight is adjusted by $\Delta e \cdot x$ and that adjustment is then multiplied by $x$ again on
the next prediction. The loop gain grows with the *square* of the input.

### (d) 2 marks

**The reasons the course states.** A learning rate of $0.0$ or less can never improve the model:
zero makes every adjustment zero, and a negative rate moves every parameter away from the reference
value. A learning rate of $1.0$ or more corrects by at least the full error on every step, which
makes training oscillate or diverge rather than converge.
*(1 mark)*

**Why passing the check guarantees nothing.** The $LR < 1.0$ bound is the right bound only for
$x = 0$, where the weight does not move at all and (c) reduces to $LR < 2$. As soon as
$\lvert x \rvert > 1$ the real bound $2/(1+x^2)$ is much tighter, and it tightens quadratically. At
$x = 4$ it is $0.1176$; the $LR = 0.5$ used in (b) sails through `train()`'s check and diverges
anyway, which is exactly what (b) shows happening.

The check is necessary, not sufficient. What actually bounds the safe learning rate is the scale of
the input data, which `train()` never looks at.
*(1 mark)*

---

## Question 2, Order, precision, and knowing when to stop (10 marks)

### (a) 2 marks

Trained in a fixed order every epoch, the model risks fitting the **ordering** rather than the data:
it sees the same sequence of corrections over and over and can settle into a pattern that reflects
how the training sets happen to be stored, rather than the relationship between $x$ and $y$. The
course also notes it raises the risk of getting stuck in a local minimum.
*(1 mark)*

What the shuffle protects is the model's generalization: reshuffling every epoch exposes the model
to the same data in varying order, so nothing in the storage order can be learned. Shuffling once at
construction would not do it; one fixed order is still one fixed order.
*(1 mark)*

### (b) 3 marks

The `static` local (`initialized`) keeps its value between calls, so the body that seeds the
generator runs on the first call and never again.
*(1 mark)*

Without it, every call would re-seed. `std::srand(std::time(nullptr))` has one-second resolution, so
several calls in the same second would reset the generator to the identical state and
`std::rand()` would return the identical sequence each time; the shuffle would stop shuffling.
*(1 mark)*

**What breaks in the L02 arrangement.** An anonymous namespace gives every translation unit its own
private copy of the function *and its own private `initialized` flag*. With two `.cpp` files that
each need the generator, the seeding happens once per file rather than once per program: the second
file's call re-seeds a generator the first file had already set up, and the "exactly once"
guarantee is gone. Seeding has to happen once per **program**, which is what a single definition in
`utils.cpp`, shared through `utils.hpp`, gives.
*(1 mark)*

### (c) 3 marks

$$\lvert 2 - 1.85\rvert = 0.15, \quad \lvert 4 - 3.9\rvert = 0.1, \quad \lvert 6 - 6.3\rvert = 0.3, \quad \lvert 8 - 8.1\rvert = 0.1$$

$$MAE = \frac{0.15 + 0.1 + 0.3 + 0.1}{4} = \frac{0.65}{4} = 0.1625$$

$$\text{precision} = 1.0 - 0.1625 = \mathbf{0.8375}$$

*(1 mark)*

**Why the threshold must be strictly inside $(0.0, 1.0)$.**

* **At $1.0$ or above:** unreachable. Precision is $1 - MAE$ and the mean *absolute* error cannot be
  negative, so precision can never exceed $1.0$ and reaches it only on an exactly perfect model.
  Training would always run the full epoch count and the early stop would never fire.
* **At $0.0$ or below:** it accepts a model whose mean absolute error is $1.0$ or worse, which
  defeats the point of checking.

*(1 mark)*

**A negative precision.** Predictions $3, 7, 1, 15$ against $2, 4, 6, 8$ give absolute errors
$1, 3, 5, 7$, so $MAE = 4$ and precision $= 1 - 4 = \mathbf{-3}$.

What this reveals: **precision here is not a percentage and not a bounded score.** It is
$1 - MAE$ measured in the *units of the output data*, so it depends entirely on the scale of $y$. On
data whose outputs run into the thousands, even a very good model reports a large negative
"precision". It is a stopping heuristic tuned to this course's small integer training data, not a
general accuracy metric, and the name flatters it.
*(1 mark)*

### (d) 2 marks

**What it saves.** Computing the precision scans every training set and performs a prediction for
each. Over thousands of epochs that is a full extra forward pass over the whole data set per epoch,
which can be a large fraction of the total work when the training loop itself is cheap.
*(1 mark)*

**The two prices.**

1. Up to nine epochs of unnecessary training after the threshold has actually been reached, since
   the check only fires on every tenth.
2. The reported epoch count is quantized to multiples of ten and so is not the true one. This is
   why L02's example prints "reached after 10 epochs" for data that in fact converged somewhere in
   epochs 1 to 10.

*(1 mark)*

---

## Question 3, Two output nodes (14 marks)

### (a) 3 marks

$x_1 = 1$, $x_2 = 0$.

$$s_1 = 0.2 + 1(0.3) + 0(0.6) = 0.5 \Rightarrow y_1 = \mathbf{0.5}$$
$$s_2 = 0.5 + 1(-0.4) + 0(0.2) = 0.1 \Rightarrow y_2 = \mathbf{0.1}$$
$$s_3 = -0.1 + 0.5(0.7) + 0.1(0.5) = -0.1 + 0.35 + 0.05 = 0.3 \Rightarrow y_3 = \mathbf{0.3}$$
$$s_4 = 0.4 + 0.5(-0.2) + 0.1(0.9) = 0.4 - 0.1 + 0.09 = 0.39 \Rightarrow y_4 = \mathbf{0.39}$$

Every sum is positive, so ReLU passes all four through unchanged.
*(3 marks)*

### (b) 4 marks

**Output layer:**

$$\delta_3 = Y_1 - y_3 = 0 - 0.3 = -0.3, \qquad e_3 = -0.3 \cdot 1 = \mathbf{-0.3}$$
$$\delta_4 = Y_2 - y_4 = 1 - 0.39 = 0.61, \qquad e_4 = 0.61 \cdot 1 = \mathbf{0.61}$$

*(2 marks)*

**Hidden layer:** each hidden node feeds **both** output nodes, so its deviation is the sum of two
terms, one per output node, each weighted by the connection from that hidden node to it:

$$\delta_1 = e_3 w_5 + e_4 w_7 = -0.3(0.7) + 0.61(-0.2) = -0.21 - 0.122 = -0.332 \Rightarrow e_1 = \mathbf{-0.332}$$
$$\delta_2 = e_3 w_6 + e_4 w_8 = -0.3(0.5) + 0.61(0.9) = -0.15 + 0.549 = 0.399 \Rightarrow e_2 = \mathbf{0.399}$$

*(2 marks)*

**How many terms, and why.** Two, one per node in the next layer. A hidden node's output was sent
to every output node, so it is partly responsible for every output node's error, and its share along
each path is that path's weight. In general $\delta_i = \sum_j e_j w_{ji}$ over all $j$ nodes in the
next layer. A candidate who used only one term has effectively disconnected one of the two output
nodes.

Note $\delta_2$: the two contributions have **opposite signs**, and the larger one wins. Node 2 is
being pulled down by output node 3 and up harder by output node 4.

### (c) 4 marks

**Output layer.** $\Delta c_3 = -0.3(0.1) = -0.03$, $\Delta c_4 = 0.61(0.1) = 0.061$. The weights are
scaled by the output layer's input, which is $y_1 = 0.5$ and $y_2 = 0.1$:

$$b_3 = -0.1 - 0.03 = \mathbf{-0.13}, \quad w_5 = 0.7 - 0.03(0.5) = \mathbf{0.685}, \quad w_6 = 0.5 - 0.03(0.1) = \mathbf{0.497}$$
$$b_4 = 0.4 + 0.061 = \mathbf{0.461}, \quad w_7 = -0.2 + 0.061(0.5) = \mathbf{-0.1695}, \quad w_8 = 0.9 + 0.061(0.1) = \mathbf{0.9061}$$

*(2 marks)*

**Hidden layer.** $\Delta c_1 = -0.332(0.1) = -0.0332$, $\Delta c_2 = 0.399(0.1) = 0.0399$. The
weights are scaled by the network's input, $x_1 = 1$ and $x_2 = 0$:

$$b_1 = 0.2 - 0.0332 = \mathbf{0.1668}, \quad w_1 = 0.3 - 0.0332(1) = \mathbf{0.2668}, \quad w_2 = 0.6 - 0.0332(0) = \mathbf{0.6}$$
$$b_2 = 0.5 + 0.0399 = \mathbf{0.5399}, \quad w_3 = -0.4 + 0.0399(1) = \mathbf{-0.3601}, \quad w_4 = 0.2 + 0.0399(0) = \mathbf{0.2}$$

*(2 marks)*

### (d) 2 marks

$$s_1 = 0.1668 + 0.2668 = 0.4336 \Rightarrow y_1 = 0.4336, \qquad s_2 = 0.5399 - 0.3601 = 0.1798 \Rightarrow y_2 = 0.1798$$
$$s_3 = -0.13 + 0.4336(0.685) + 0.1798(0.497) = -0.13 + 0.297016 + 0.089361 = \mathbf{0.256377}$$
$$s_4 = 0.461 + 0.4336(-0.1695) + 0.1798(0.9061) = 0.461 - 0.073495 + 0.162917 = \mathbf{0.550422}$$

$$\delta_3 = 0 - 0.256377 = \mathbf{-0.2564} \quad (\text{was } -0.3)$$
$$\delta_4 = 1 - 0.550422 = \mathbf{0.4496} \quad (\text{was } 0.61)$$

Both deviations shrank in magnitude, so the step helped, and it helped node 4 substantially more
than node 3, which is what you would expect from the much larger error node 4 started with.
*(2 marks)*

### (e) 1 mark

$w_2$ and $w_4$ are unchanged, at $0.6$ and $0.2$. Both are the hidden layer's weights on input
$x_2$, and $x_2 = 0$ for this training set. The weight update is
$w \mathrel{+}= \Delta c \cdot \text{input}$, so a weight whose input is zero receives an adjustment
of exactly zero however large its node's error was.

This is not the dying ReLU problem. Nothing is broken: a weight that had no influence on this
prediction gets no share of the blame for it, which is correct. It will move on the next training
set that has $x_2 = 1$.
*(1 mark)*

---

## Question 4, The network class, and what a stub can prove (12 marks)

### (a) 3 marks

**Two consequences of storing the layers by reference:**

1. **`Shallow` does not own its layers.** The caller constructs them and must keep them alive for at
   least as long as the network; a network outliving its layers holds dangling references. It also
   means the class has nothing to destroy, which is why its destructor is `default`.
2. **A layer can be replaced without touching `Shallow`.** This is precisely what L05 does: `Stub`
   is swapped for `Dense` and not a line of the network class changes. It also lets a *test* reach
   the layer after the network was built and manipulate it, which is what `setOutput()` exists for.

Also creditable: reference members make the class non-assignable, which is consistent with the
course deleting the copy and move operators anyway.
*(2 marks for any two)*

**`predict()` returns `myOutputLayer.output()`**, a reference to the output layer's own output
vector. No member variable is needed because the output layer already stores it and the network
holds that layer by reference: the value is always live. Keeping a copy would introduce the
possibility of it going stale, which is the exact failure `setOutput()` is designed to expose.
*(1 mark)*

### (b) 3 marks

The three steps, per training set: **feedforward**, then **backpropagation**, then **optimization**.
*(1 mark)*

**Constraint 1: between the steps.** Backpropagation computes each node's error from the outputs
feedforward just produced, so it cannot run first. Optimization adjusts parameters using the errors
backpropagation just computed, so it cannot run before that. The three are a strict chain.
*(1 mark)*

**Constraint 2: inside backpropagation.** The output layer must go first. The hidden layer's
deviation is $\sum_j e_j w_{ji}$ over the output layer's *computed errors*, which do not exist until
the output layer has compared its output against the reference value.
*(1 mark)*

Worth noting what is *not* constrained: the two `optimize()` calls can be made in either order.
Optimizing the hidden layer does not change `myHiddenLayer.output()`; only `feedforward()` writes
that, so the output layer still receives the values it actually consumed.

### (c) 4 marks

**`setOutput()`** sets every element of the stub's output to a chosen value. It makes it testable
that `predict()` reads its output layer **live**: build a network, change the output layer's output,
and the network's prediction must change with it. Nothing on `Interface` can establish this, because
a network that cached a copy at construction would satisfy every interface method identically; it
would return a plausible vector of the right size, and only an *external* change to the layer can
tell the two apart.
*(1 mark)*

**`feedforwardCount()`** tallies calls to `feedforward()`. It makes the **shape of the training
loop** testable: `train()` must perform exactly one feedforward per training set per epoch, so the
count after training must be $sets \times epochs$. Nothing else can establish it. A loop that runs a
single pass instead of every epoch lines up dimensionally, touches no invalid memory, and returns
`true` exactly as a correct loop does; without a tally the two are indistinguishable.
*(1 mark)*

Note that these two work because the stub *computes nothing*. A layer whose output is a fixed
constant is what makes the network's wiring visible: any change in what the network predicts must
have come from the wiring, because it cannot have come from the arithmetic.
*(1 mark)*

**Why the counter is incremented before the size check.** The tally then records how often the layer
was **asked** to feed forward, not how often it agreed to. That is the quantity a test of the
training loop actually wants: the loop's job is to make the right number of calls, and whether each
call was accepted is a separate question, tested separately by the return value. Incrementing after
the check would conflate the two, and a loop that made the right calls with wrongly sized data would
under-report rather than fail.
*(1 mark)*

### (d) 2 marks

**The rule:** `std::terminate()` is reserved for the constructor. Everything else reports failure
through its return value.
*(1 mark)*

**The reason:** a constructor has no return value, so it has no way to tell the caller that
construction failed. A `Shallow` built with no training data, or with an output layer whose weight
count does not match the hidden layer's node count, can never do anything useful, and letting it
exist only defers the failure to somewhere less informative. `train()` does have a return value, so
it can hand the decision back to the caller, who may reasonably want to retry with a different
learning rate rather than have the program killed.
*(1 mark)*

---

## Question 5, The dense layer, and the mistake only Tanh can catch (12 marks)

### (a) 3 marks

| Member                  | Size                          | L03 quantity                                  |
| ----------------------- | ----------------------------- | --------------------------------------------- |
| `myOutput`              | `nodeCount()`                 | $y$, each node's output, after the activation |
| `myPreActivationOutput` | `nodeCount()`                 | $s$, each node's weighted sum, before it      |
| `myError`               | `nodeCount()`                 | $\Delta e$, each node's computed error        |
| `myBias`                | `nodeCount()`                 | $b$, each node's bias                         |
| `myWeights`             | `nodeCount() x weightCount()` | $w$, each node's weights                      |
| `myActFunc`             | none                          | $\sigma$, the layer's activation function     |

*(3 marks; deduct for a missing `myPreActivationOutput`, which is the one candidates forget and the
one part (c) is about)*

### (b) 5 marks

**Feedforward.**

$$s = b + w_0 x_0 + w_1 x_1 = 0.3 + 0.6(1.0) + 0.6(1.0) = \mathbf{1.5}$$
$$y = \tanh(1.5) = \mathbf{0.905148}$$

*(1 mark)*

**Backpropagation.**

$$\delta = y_{ref} - y = 0.5 - 0.905148 = \mathbf{-0.405148}$$
$$\sigma'(s) = 1 - \tanh^2(1.5) = 1 - 0.905148^2 = 1 - 0.819293 = \mathbf{0.180707}$$
$$\Delta e = \delta \cdot \sigma'(s) = -0.405148 \times 0.180707 = \mathbf{-0.073213}$$

*(2 marks)*

**Optimization.** $\Delta e \cdot LR = -0.073213 \times 0.2 = -0.014643$.

$$b = 0.3 - 0.014643 = \mathbf{0.285357}$$
$$w_0 = 0.6 - 0.014643(1.0) = \mathbf{0.585357}, \qquad w_1 = 0.6 - 0.014643(1.0) = \mathbf{0.585357}$$

Both weights move by the same amount because both inputs are 1.0.
*(1 mark)*

**Verification.**

$$s = 0.285357 + 0.585357 + 0.585357 = 1.456072 \Rightarrow y = \tanh(1.456072) = 0.896887$$
$$\delta = 0.5 - 0.896887 = \mathbf{-0.396887} \quad (\text{was } -0.405148)$$

*(1 mark)*

The deviation moved by only $0.008$ on an error of $0.41$, about 2%. This is tanh saturation doing
exactly what it is supposed to: at $s = 1.5$ the derivative is already down to $0.18$, so a large
error produces a small step. A candidate who observes that this node will take many epochs to move,
and that this is a feature rather than a fault, has understood the question.

### (c) 4 marks

Passing the **output** instead of the weighted sum:

$$\sigma'(y) = 1 - \tanh^2(0.905148) = 1 - 0.718796^2 = 1 - 0.516668 = \mathbf{0.483332}$$
$$\Delta e_{wrong} = -0.405148 \times 0.483332 = \mathbf{-0.195821}$$

$$\text{ratio} = \frac{-0.195821}{-0.073213} = \mathbf{2.675}$$

*(2 marks)*

The step is 2.7x too large. And the error is worst where it matters most: as $\lvert s \rvert$ grows
the true derivative collapses toward zero while the mistaken one flattens out at a much larger
value, so the bug removes precisely the braking that tanh saturation is there to provide.

**Why ReLU hides it.** ReLU's derivative tests only the predicate "is this value positive", and
$\max(0, s)$ is positive exactly when $s$ is. So `actFuncDelta(Relu, s)` and `actFuncDelta(Relu, y)`
return the same number for every $s$, not merely for the values that happen to be tested. There is
no input that distinguishes them.
*(1 mark)*

**The consequence for testing.** A suite that exercises only `Relu` cannot detect the bug at all -
every assertion passes and the layer is wrong. The test has to be written against `Tanh`
specifically, which is what `BackpropagateUsesPreActivationDerivative` in the L05 suite does. The
general point: a test has to include a case that *distinguishes* the correct implementation from the
plausible wrong one, not merely a case both of them survive.
*(1 mark)*

---

## Question 6, Padding, sharing, and the kernel's gradient (14 marks)

### (a) 3 marks

$\text{pad} = 3 / 2 = 1$, so the padded input is 6 x 6:

```text
0 0 0 0 0 0
0 0 1 1 0 0
0 1 1 1 1 0
0 1 1 1 1 0
0 0 1 1 0 0
0 0 0 0 0 0
```

*(1 mark)*

With `ActFunc::None` and a bias of 0, the output is the raw weighted sum. The kernel's middle column
is all zeros, so each cell is (sum of the window's left column) minus (sum of its right column):

$$s(0,0) = (0 + 0 + 0) - (0 + 1 + 1) = -2, \qquad s(0,3) = (0 + 1 + 1) - (0 + 0 + 0) = 2$$

**Output:**

```text
-2  -1   1   2
-3  -1   1   3
-3  -1   1   3
-2  -1   1   2
```

*(1 mark)*

**What the kernel responds to.** It is a **vertical edge detector**. It compares the three pixels down
the left of the window against the three down the right and ignores the middle column entirely, so
it reacts to a left-right change in brightness and not at all to a top-bottom one.

**What the sign means.** Positive means brighter on the left of the window than on the right;
negative means the reverse. Zero means the neighbourhood is left-right symmetric, which is why the
output above is antisymmetric across the image: this input is a symmetric blob, so the left half
gives negative responses and the right half gives the mirrored positive ones. The magnitude is the
strength of the edge.
*(1 mark)*

### (b) 3 marks

Both cases pad to **6 x 6**: $\text{pad} = 3/2 = 1$ and $\text{pad} = 2/2 = 1$ are the same number,
so the padded input is $4 + 2(1) = 6$ either way.
*(1 mark)*

**With the 3 x 3 kernel**, output cell $(i, j)$ reads padded rows $i \ldots i+2$. The largest output
index is 3, so the largest padded index read is $3 + 2 = 5$, the last row and column. Everything is
used.

**With the 2 x 2 kernel**, output cell $(i, j)$ reads padded rows $i \ldots i+1$. The largest padded
index read is $3 + 1 = 4$. **Padded row 5 and padded column 5 are never touched by `feedforward()`
at all**, and correspondingly never receive anything in `backpropagate()`.
*(1 mark)*

**Why, and what it implies.** A same-size output needs $kernelSize - 1$ extra rows in total. For an
odd kernel that is an even number and splits evenly, one on each side. For a 2 x 2 kernel it is 1,
but `pad = kernelSize / 2` adds 1 on *each* side, which is one too many; the loop starts at $(0,0)$
so the surplus lands at the bottom and the right and is dead storage.

The alignment consequence is the real point. The padding is effectively one row and column *before*
the image and none after, so output cell $(i,j)$ is computed from the window covering input cells
$(i-1, j-1)$ to $(i, j)$. There is no centre cell for an even kernel to sit on, so the output is not
centred on the input: a feature at input position $(r, c)$ appears in the output shifted down and
right by half a kernel. An odd kernel has a true centre and does not suffer this.
*(1 mark)*

### (c) 4 marks

With `ActFunc::None` the derivative is 1 everywhere, so `outGrad` equals the incoming gradient at
every position and no gradient is lost.

$$\text{biasGradient} = 2 + 4 + 6 + 8 = \mathbf{20}$$

*(1 mark)*

$$\text{kernelGradient}[a][b] = \sum_{i,j} P[i+a][j+b] \cdot g[i][j]$$

Only four gradients are non-zero: $g(0,1) = 2$, $g(1,3) = 4$, $g(2,0) = 6$, $g(3,2) = 8$. Each
kernel entry therefore has four terms. The centre entry, as a worked example:

$$k_{11} = P[1][2](2) + P[2][4](4) + P[3][1](6) + P[4][3](8) = 1(2) + 1(4) + 1(6) + 1(8) = 20$$

**Kernel gradients:**

```text
12   14   14
12   20    8
 6    6    8
```

*(2 marks)*

**Updated parameters** at $LR = 0.01$:

$$\text{bias} = 0.0 + 20(0.01) = \mathbf{0.2}$$

```text
1.12   0.14  -0.86
1.12   0.20  -0.92
1.06   0.06  -0.92
```

*(1 mark)*

### (d) 4 marks

**Conv layer:** $3 \times 3 + 1 = \mathbf{10}$ trainable parameters.

**Dense layer** producing 16 outputs from 16 inputs: $16 \times 16 + 16 = \mathbf{272}$.
*(1 mark)*

**How many positions contributed to the centre kernel gradient.** **Four**, the four output
positions with a non-zero gradient, $(0,1)$, $(1,3)$, $(2,0)$ and $(3,2)$. In principle all
**sixteen** output positions contribute; twelve of them contributed a term of zero here only because
their gradients were zero.
*(1 mark)*

**Why one update nevertheless.** The kernel is a *single* set of nine numbers used at every one of
the sixteen output positions. Each position therefore gives a partial derivative with respect to the
same nine parameters, and the derivative of the total error with respect to one parameter is the
**sum** of those partials. Accumulating first and applying once is not an optimization; it *is* the
gradient.
*(1 mark)*

Applying an update per position instead would be a different algorithm and a wrong one: each of the
sixteen updates after the first would be computed against a kernel that the previous updates had
already moved, so the sum would no longer be the gradient at the point where it was evaluated, and
the effective step size would scale with the image size. This is the one place where a conv layer
genuinely differs from a dense layer in its backward pass, and it is a direct consequence of weight
sharing: a parameter used in many places accumulates blame from all of them.
*(1 mark)*

---

## Question 7, A pooling layer that runs and returns true (12 marks)

### (a) 4 marks

**Fault 1: `>=` where `feedforward()` uses `>`.** `feedforward()` keeps a value only when it is
strictly greater than the best so far, so on a tie it records the **first** occurrence.
`backpropagate()` accepts equality, so it walks past the first occurrence and records the **last**.
On any block with a tied maximum the two halves disagree.

What the conv layer behind receives: a gradient at a position that did **not** produce the pooled
output, and nothing at the position that did. The gradient is real and lands on the wrong pixel, so
the conv layer updates its kernel from a neighbourhood the prediction never used.
*(2 marks)*

**Fault 2: `myInputGradients` is never reset to zero.** Each call writes exactly one cell per block
and leaves every other cell untouched, so those cells retain whatever the *previous* call left in
them.

What the conv layer behind receives: this example's gradients mixed with stale gradients from
earlier training examples, at up to $poolSize^2 - 1$ positions per block. Since the course trains
with a batch size of 1, that is one earlier example's blame added to every example's. The layer
never sees a clean gradient matrix after the first training set.
*(2 marks)*

### (b) 3 marks

| Block        | Values (position)                              | First max | Last max |
| ------------ | ---------------------------------------------- | --------- | -------- |
| top left     | 0.5 (0,0), 0.5 (0,1), 0.2 (1,0), 0.3 (1,1)     | (0, 0)    | (0, 1)   |
| top right    | -0.1 (0,2), -0.4 (0,3), -0.6 (1,2), -0.9 (1,3) | (0, 2)    | (0, 2)   |
| bottom left  | 1.0 (2,0), 0.4 (2,1), 0.1 (3,0), 0.9 (3,1)     | (2, 0)    | (2, 0)   |
| bottom right | 0.7 (2,2), 0.2 (2,3), 0.3 (3,2), 0.8 (3,3)     | (3, 3)    | (3, 3)   |

**What the code produces** (assuming `myInputGradients` happened to be all zeros beforehand):

```text
 0   10   20    0
 0    0    0    0
30    0    0    0
 0    0    0   40
```

**What it should have produced:**

```text
10    0   20    0
 0    0    0    0
30    0    0    0
 0    0    0   40
```

*(2 marks)*

**They differ in the top-left block**, the only one with a tied maximum: the gradient 10 lands on
$(0,1)$ instead of $(0,0)$. Note how small the visible difference is, one value one column across,
and that the layer returns `true` either way.
*(1 mark)*

### (c) 3 marks

**What is wrong.** `MaxPool` has **no trainable parameters**, and `myInput` is not one. It is the
last feedforward input, stored only so `backpropagate()` can re-locate which cell held each block's
maximum, and it is overwritten wholesale by the next `feedforward()`. This method therefore adjusts
something that is never learned and never read again: it burns $O(n^2)$ work per training example to
corrupt the layer's own record of what it was last shown.
*(1 mark)*

**The correct body:**

```cpp
bool MaxPool::optimize(const double learningRate) noexcept
{
    (void)(learningRate);
    return true;
}
```

**The general rule:** a layer with no trainable parameters has an `optimize()` that is a **genuine
no-op**, not a placeholder awaiting an implementation. There is nothing to update, and inventing
something to update is worse than doing nothing.
*(1 mark)*

**How `flatten_layer::Interface` differs, and which is stronger.** It declares **no `optimize()` at
all**, not an empty one, not a defaulted one. The interface simply does not ask for it.

That is the stronger design, because it moves "this layer has nothing to optimize" from a convention
into the type system: a `Flatten` that tried to optimize something would not compile, whereas
`MaxPool` relies on whoever writes it remembering that the method must do nothing, exactly the
memory lapse this listing shows. The course keeps `optimize()` on `conv_layer::Interface` only
because `Conv` and `MaxPool` share that interface and `Conv` genuinely needs it; the no-op is the
price of the shared interface, not a preference.
*(1 mark)*

### (d) 2 marks

**The symptom.** Nothing at all, on the surface. The network compiles, runs, predicts, and every
call returns `true`. What the user sees is a network that simply does not learn well: predictions
that never converge on the training targets, precision that plateaus at roughly whatever the random
initialization gave, and no error message anywhere. Training more epochs does not help, which is the
tell.
*(1 mark)*

**The test.** `LearnsToRecognizeAllFourDigits` in the L10 suite, the component test that trains on
the digits 0 to 3 and checks the right output node wins for each. Every *unit* test can pass while
this one fails, which is the L10 appendix's stated signal that the layers are individually right and
the wiring or the gradient flow is not. The next place to look is
`BackpropagationReachesTheConvLayerThroughEveryLayer`, which chains the four layers by hand and
checks the gradients survive the trip back; it would catch fault (a) directly.
*(1 mark)*

---

## Question 8, Every layer, side by side (8 marks)

### (a) 4 marks

| Layer     | Trainable parameters                    | `feedforward()`                                                      | `backpropagate()`                                                                       | `optimize()`                                    |
| --------- | --------------------------------------- | -------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | ----------------------------------------------- |
| `Fixed`   | weight $k$, bias $m$                     | $y = kx + m$                                                          | $\delta = y_{ref} - y_p$                                                                  | $k \mathrel{+}= \delta \cdot LR \cdot x$, $m \mathrel{+}= \delta \cdot LR$ |
| `Dense`   | one bias and `weightCount()` weights per node | $y = \sigma\!\left(b + \sum w_i x_i\right)$ per node                | $\Delta e = \delta \cdot \sigma'(s)$ per node, $\delta$ from the reference values or from the next layer | $b \mathrel{+}= \Delta e \cdot LR$, $w \mathrel{+}= \Delta e \cdot LR \cdot \text{input}$ |
| `Conv`    | one shared kernel and one bias           | as `Dense`, but local (one small window per output) and shared (the same kernel everywhere) | gradients accumulated across every position the kernel visited, plus input gradients for the layer behind | $\text{kernel} \mathrel{+}= \text{grad} \cdot LR$, $\text{bias} \mathrel{+}= \text{grad} \cdot LR$ |
| `MaxPool` | none                                     | passes the largest value of each block forward                        | routes the incoming gradient to the position that held that maximum; every other position gets 0 | no-op: nothing to update                        |
| `Flatten` | none                                     | reshapes 2D to 1D, row-major                                          | reshapes 1D back to 2D, the exact inverse                                                 | does not exist in the interface                 |

*(4 marks; roughly one per row beyond the first, with the `Fixed` row treated as free)*

### (b) 2 marks

**Locality.** Each output looks only at a small window of the input rather than at all of it.

*Buys:* the parameter count stops scaling with the input size, and the spatial structure of the
image is preserved rather than flattened away.
*Costs:* a single layer cannot relate two distant parts of the image at all; reach has to be built
up by stacking layers, or by pooling between them to shrink the image under a fixed window.
*(1 mark)*

**Weight sharing.** The same kernel is reused at every position instead of each output owning its
own weights.

*Buys:* the parameter count becomes independent of the image size entirely, and a pattern is
recognized wherever in the image it appears, which is what makes the layer generalize from far less
training data.
*Costs:* the layer *cannot* learn anything position-specific; it is forced to treat every location
identically. That is right for images and wrong for data where position itself carries meaning.
*(1 mark)*

### (c) 2 marks

**Feedforward**, left to right, each layer given the previous layer's `output()`:

```text
Conv::feedforward(input)
MaxPool::feedforward(conv.output())
Flatten::feedforward(maxPool.output())
Dense::feedforward(flatten.output())
```

**Backpropagation**, right to left. The target vector goes straight into the dense layer; every
earlier layer is given the next layer's `inputGradients()`:

```text
Dense::backpropagate(target)
Flatten::backpropagate(dense.inputGradients())
MaxPool::backpropagate(flatten.inputGradients())
Conv::backpropagate(maxPool.inputGradients())
```

*(1 mark for both, including the target going into `Dense` and `inputGradients()` carrying the rest)*

**Optimization**, using gradients `backpropagate()` has already computed:

```text
Conv::optimize(learningRate)
MaxPool::optimize(learningRate)      // no-op
Dense::optimize(flatten.output(), learningRate)
Flatten                              // no optimize() exists
```

The dense layer needs the previous layer's `output()` as well as the learning rate, because its
weight update is scaled by the value that actually flowed along each weight. The order of the
`optimize()` calls does not matter: all the gradients already exist, and no layer's optimization
changes any other layer's stored output.
*(1 mark)*

---

## Question 9, Off the laptop (8 marks)

### (a) 3 marks

Shapes: conv preserves 32 x 32 (it pads); pooling halves it to 16 x 16; flatten gives
$16 \times 16 = 256$ values; the dense layer has 64 nodes over those 256 inputs.

| Layer       | Trainable parameters                    |
| ----------- | --------------------------------------- |
| conv        | $3 \times 3 + 1 = 10$                   |
| max pooling | 0                                       |
| flatten     | 0                                       |
| dense       | $64 \times 256 + 64 = 16{,}448$         |
| **total**   | $\mathbf{16{,}458}$                     |

*(1 mark)*

| Type     | Bytes each | Total footprint            |
| -------- | ---------- | -------------------------- |
| `double` | 8          | 131,664 B (128.58 KB)      |
| `float`  | 4          | 65,832 B (64.29 KB)        |
| `int8`   | 1          | 16,458 B (16.07 KB)        |

*(1 mark)*

The budget is $64 \times 1024 = 65{,}536$ bytes.

$$131{,}664 > 65{,}536 \Rightarrow \textbf{double does not fit}$$
$$65{,}832 > 65{,}536 \Rightarrow \textbf{float does not fit either, by 296 bytes}$$
$$16{,}458 < 65{,}536 \Rightarrow \textbf{only int8 fits}$$

The `float` case is the one that has to be done in bytes. "About 64 KB against a 64 KB budget" reads
like a fit and is a miss, by 296 bytes; that is before a single byte of program code, stack, or
input buffer. This is the argument for post-training quantization in one line: train at full
precision on a workstation, deploy at `int8`.
*(1 mark)*

### (b) 3 marks

Q16.16 scales by $2^{16} = 65536$.

$$\texttt{toFixed(0.75)} = 0.75 \times 65536 = \mathbf{49{,}152}$$
$$\texttt{toFixed(-2.5)} = -2.5 \times 65536 = \mathbf{-163{,}840}$$

*(1 mark)*

$$49{,}152 \times (-163{,}840) = \mathbf{-8{,}053{,}063{,}680}$$

A 32-bit signed integer holds $-2{,}147{,}483{,}648$ to $2{,}147{,}483{,}647$. The product exceeds
that in magnitude by nearly a factor of four, so it must be formed in `std::int64_t` before being
scaled back down, which is precisely what the cast in `fixedMultiply()` is for. Two values each
scaled by $2^{16}$ multiply to a result scaled by $2^{32}$, so the operands can each be small and
the intermediate still overflow.

$$-8{,}053{,}063{,}680 \gg 16 = -122{,}880 \qquad \texttt{toDouble}(-122{,}880) = \frac{-122{,}880}{65{,}536} = \mathbf{-1.875}$$

and $0.75 \times (-2.5) = -1.875$, exactly.
*(1 mark)*

$$\text{resolution} = \frac{1}{65536} \approx \mathbf{0.0000153}$$

with a representable range of roughly $\pm 32{,}768$. That is the trade: fixed dynamic range and
fixed absolute precision, in exchange for arithmetic that is a single integer instruction on a core
with no FPU rather than a software-emulated floating-point call.
*(1 mark)*

### (c) 2 marks

**Two other constraints** (any two):

* **Dynamic memory allocation.** Every layer in this course is built on `std::vector`, which
  allocates on the heap. Long-running embedded systems risk heap fragmentation, allocation is not
  constant-time, and some bare-metal targets have no heap at all. The usual answer is `std::array`
  or plain arrays sized at compile time, trading runtime-sized layers for fully static memory use.
* **Real-time deadlines.** Inference inside a control loop has a hard budget; the course's example
  is a 1 kHz motor controller, giving sensor read, inference and actuation 1 ms in total.
* Also creditable: the need to quantize before deployment, and the absence of a filesystem or of
  `printf()` for diagnostics.

*(1 mark)*

**Multiply-accumulate count.** A conv layer performs roughly
$inputSize^2 \times kernelSize^2$ multiply-adds per feedforward pass:

$$32 \times 32 \times 3 \times 3 = \mathbf{9{,}216 \text{ MACs}}$$

**Why worst case, not average.** A hard deadline is missed by a single late iteration; it is not an
average that has to be met but every instance. A network that is usually fast and occasionally
spikes still fails, and it fails in the way casual testing does not reveal, because the average
looks fine and the spike is rare. Scheduling has to be budgeted from the worst case, so the worst
case is the number that has to be measured.
*(1 mark)*

---
