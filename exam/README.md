# Written Examinations

Two complete four-hour papers for **Machine Learning**, with worked solutions.

```text
paper_a.md              Paper A, questions only. Hand this out.
paper_a_solutions.md    Paper A, model answers with marks.
paper_b.md              Paper B, questions only.
paper_b_solutions.md    Paper B, model answers with marks.
```

---

## What these are for

The course as written has **no exam and nothing is marked**. What you get instead is a shipped test
suite per lecture (L01 to L05 and L08 to L10), the L06/L07 demos checked against the numbers you
worked out by hand in [L06's appendix B](../lectures/L06/appendix/b_exercises.md), and the
self-assessment questions in [L10's appendix C](../lectures/L10/appendix/c_exercises.md). Every one
of those checks a component you built or a number you computed beside it.

These papers do not replace that. **They are here to test your own skills and knowledge, nothing
more.** They gate nothing, they are not a qualification, and no part of the course requires them.
Nothing in the repository depends on them and `make test` does not know they exist. They are useful
where a written result is wanted anyway, for a certifying employer or a formal course credit, and
useful on their own for finding out what you can reconstruct with nothing in front of you.

**Take one after the course is over.** Every paper draws on all ten lectures, so sitting one partway
through examines material nobody has taught you yet, and the result says more about how far you have
read than about what you have understood. The intended point is after L10, once `cnn_work` trains
end to end and the appendices have settled.

---

## The two papers

Both cover the whole course, L01 to L10, at the same weighting. They share no question. Either can
be used alone; use both as a main sitting and a resit, or in alternate years.

### Paper A

| Question | Topic                                                 | Lectures      | Marks   |
| -------- | ----------------------------------------------------- | ------------- | ------- |
| 1        | Learning the rules instead of writing them            | L01, L02      | 12      |
| 2        | The training loop that trains nothing                 | L02, L04      | 12      |
| 3        | A network trained by hand                             | L03           | 14      |
| 4        | The activation function, and the argument it is given | L03, L05      | 10      |
| 5        | The dense layer, written out                          | L05           | 12      |
| 6        | Why an image is not just a long vector                | L06           | 10      |
| 7        | A convolutional layer, traced by hand                 | L06, L07, L08 | 14      |
| 8        | Max pooling, and the value it forgets                 | L07, L09      | 8       |
| 9        | The finished network, and the target it runs on       | L10           | 8       |
|          |                                                       |               | **100** |

### Paper B

| Question | Topic                                                | Lectures      | Marks   |
| -------- | ---------------------------------------------------- | ------------- | ------- |
| 1        | The model, and the rate that destroys it             | L01           | 10      |
| 2        | Order, precision, and knowing when to stop           | L02, L04      | 10      |
| 3        | Two output nodes                                     | L03           | 14      |
| 4        | The network class, and what a stub can prove         | L03, L04      | 12      |
| 5        | The dense layer, and the mistake only Tanh can catch | L05           | 12      |
| 6        | Padding, sharing, and the kernel's gradient          | L06, L07, L08 | 14      |
| 7        | A pooling layer that runs and returns true           | L07, L09      | 12      |
| 8        | Every layer, side by side                            | L10           | 8       |
| 9        | Off the laptop                                       | L10           | 8       |
|          |                                                      |               | **100** |

Paper A leans towards **tracing an algorithm forwards**: one epoch of linear regression by hand, a
full feedforward()/backpropagate()/optimize() step through a small network, a 4 x 4 image pushed
through conv, pooling and flatten, and the gradients routed back. Paper B leans towards **the
conditions under which the algorithm stops working**: the learning rate that diverges inside the
range `train()` permits, the derivative that is right for ReLU and wrong for Tanh, the even kernel
whose padding does not line up, and a pooling layer whose faults never produce an error message.

---

## Code on a written paper

Unlike the papers for some other QAcademy courses, these two **do ask a candidate to write C++**.
That is deliberate. This course teaches machine learning by building it: the specification of every
layer is prose in an appendix and a test suite that tells you whether what you wrote satisfies it,
and a paper that avoided code entirely would be examining a different course.

The code questions come in two shapes, and neither of them is a typing exercise:

* **Write one method.** `Dense::feedforward()`, `backpropagate()` and `optimize()` are asked for in
  Paper A; the answers are ten to twenty lines each. What carries the marks is the algorithm,
  namely the validation, the weighted sum starting at the bias, the derivative taken at the right
  argument; not the syntax.
* **Find what is wrong with the algorithm.** Both papers hand over a listing that compiles, runs,
  and returns `true`, and ask what it does instead of what it should. The faults are the ones the
  course's own test-suite READMEs single out: a shuffle nobody reads, backpropagation in the wrong
  order, a max search that starts at `0.0`, a gradient matrix that is never cleared.

Both papers state in their rubric that answers may be written in C++17 **or in unambiguous
pseudocode**, and that nothing is marked on syntax. A missing semicolon costs nothing. A missing
bounds check costs everything.

---

## Conventions the papers assume

Both papers state these in their own rubric, so a candidate never has to have read this file.

* **The course's sign convention.** The deviation is $\delta = y_{ref} - y_p$ and every parameter is
  *increased* by its adjustment, so every update reads `parameter += gradient * learningRate` with
  no minus sign. A candidate who flips the sign consistently and says so is not penalised twice.
* **The activation derivative is evaluated at the weighted sum $s$**, never at the output $y$. Both
  papers have a question about what happens when it is not.
* **Conv layers zero-pad** with `pad = kernelSize / 2`, so the output is the same size as the input.
* **Max pooling breaks ties toward the first occurrence** in row-major order, and **flatten is
  row-major**, matching the L06 hand-training exercise and the shipped implementations.
* **Only the values nobody could be expected to carry are supplied**: a handful of `tanh()` values,
  $2^{16}$, and that a KB is 1024 bytes. Parameter counts, output shapes and padding sizes are
  derived, because deriving them is the examinable skill.

---

## Marking

Every solution is written to be marked by somebody who has read the appendices and does not
otherwise write machine learning code, so each carries the reasoning rather than the answer alone.
Marks are shown per part in the solutions.

Three conventions worth agreeing before a paper is marked:

* **Method carries the marks.** A correct update rule with an arithmetic slip in it is worth more
  than a correct number with no working, and both papers are built so that later parts consume
  earlier answers. Follow through an error rather than penalising it twice.
* **The named traps are worth full marks on their own.** Several questions exist entirely to see
  whether a candidate avoids a specific mistake: the activation derivative taken from $y$ instead of
  $s$, a gradient that ReLU has already annihilated, `optimize()` given the network's input instead
  of the previous layer's output, a max search starting at `0.0`, an `optimize()` invented for a
  layer that has nothing to optimize. Where a solution flags one of these, a candidate who walks
  into it loses those marks and no others.
* **The discussion parts are not decoration.** "State why" is where the course's actual content is,
  and a paper marked only on the arithmetic would pass a candidate who has understood none of it.

---
