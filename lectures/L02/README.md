# L02 - Linear Regression (Part II)

## Agenda
* Further implementation of regression models in software: randomizing the training order.
* Further implementation of regression models in software: computing precision.

---

## Objectives
* Be able to randomize the order of the training sets before each new epoch.
* Be able to compute the precision of a trained regression model.
* Be able to determine when the model is sufficiently trained based on a given threshold value.

---

## Instructions

### Preparation
* Read through [appendix A](./appendix/a_theory.md), covering randomizing the training order and precision calculation for linear regression models.

### During the Lecture
* Participate in the lecture (live coding session).

### After the lecture
* Complete the exercises in [appendix B](./appendix/b_exercises.md).
* Solutions will be available a few days after the lecture.

---

## Evaluation
* Why is it beneficial to randomize the training order every epoch?
* Why is a static local variable used in `initRandom()` to ensure the random number generator is only initialized once?
* How is the precision of a linear regression model computed?
* What does it mean for the precision to exceed a given threshold, and how is this used to stop training early?
* Why is precision calculation important when training machine learning models?

---

## Next Lecture
* Introduction to neural networks: feedforward, backpropagation, gradient descent, activation functions.
* Training neural networks by hand.
* Implementation of a dense layer interface and stub.

---
