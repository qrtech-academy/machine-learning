# L02 - Linear Regression (Part II)

## Agenda
* Further implementation of regression models in software, including randomizing the training
  order and computing precision.
* A second regression model, which adapts its own learning rate while training.
* Concepts and implementation techniques that will prepare you for neural networks in the next
  lecture.

---

## Objectives
* Be able to randomize the order of the training sets before each new epoch.
* Be able to compute the precision of a trained regression model.
* Be able to determine when the model is sufficiently trained based on a given threshold value.
* Be able to adjust the learning rate during training, based on how much the model is improving.

---

## Instructions

### Preparation
* Read through [appendix A](./appendix/a_theory.md), covering randomizing the training order and precision calculation for linear regression models.

### During the Lecture
* Participate in the lecture (live coding session).

### After the lecture
* Complete the exercises in [appendix B](./appendix/b_exercises.md).
* Verify your implementation with the [test suite](./exercises/test/README.md).
* Solutions are available [here](./exercises/README.md).

---

## Evaluation
* Why is it beneficial to randomize the training order every epoch?
* Why is a static local variable used in `initRandGen()` to ensure the random number generator is only initialized once? Why does the function belong in `ml/utils.cpp` rather than in an anonymous namespace in each model's source file?
* How is the precision of a linear regression model computed?
* What does it mean for the precision to exceed a given threshold, and how is this used to stop training early?
* Why is precision calculation important when training machine learning models?
* How does `ml::lin_reg::Adaptive` decide whether to raise or lower its learning rate, and why is the rate clamped at both ends?

---

## Next Lecture
* Introduction to neural networks: feedforward, backpropagation, gradient descent, activation functions.
* Training neural networks by hand.
* Implementation of a dense layer interface and stub.

---
