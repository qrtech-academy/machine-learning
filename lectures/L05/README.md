# L05 - Dense Layer

## Agenda
* Implementing a dense layer in software (using the code from the exercise in L04):
    * Creating a concrete subclass `Dense`, replacing the existing stub class.
    * Adding a handful of activation functions.
* Completing the implementation - feedforward, backpropagation, and optimization.
* Discussion: overfitting and why training error alone doesn't tell you whether a network has learned to generalize.
* Discussion: extending the optimizer with momentum, and recognizing/debugging a numerically unstable network.

---

## Objectives
* Have created a concrete subclass that inherits the existing dense layer interface.
* Have implemented all methods, including `feedforward()`, `backpropagate()`, and `optimize()`.
* Have created a fully functioning dense layer implementation.
* Have tested the implementation with a complete neural network (the one created in the exercise in L04).
* Be able to recognize overfitting from a train/test error gap, and explain why held-out data is needed to measure generalization.
* Be able to implement a momentum-based optimizer as an extension of the existing `optimize()` method.
* Be able to recognize and debug a numerically unstable or "stalled" network in a from-scratch implementation.

---

## Instructions

### Preparation
* Use your code from the exercise in L04.
* Read through [appendix A](./appendix/a_theory.md), covering the dense layer's architecture and how the math maps onto the code.

### During the Lecture
* Participate in the lecture (live coding session).

### After the lecture
* Complete the exercises in [appendix B](./appendix/b_exercises.md): declaring a concrete implementation that inherits `ml::dense_layer::Interface`, then implementing `feedforward()`, `backpropagate()`, and `optimize()`.
* Test the implementation by training the neural network on existing training data, adjusting the number of epochs and/or the learning rate as needed.
* Read [appendix A](./appendix/a_theory.md)'s sections on overfitting, momentum, and debugging numerical instability, and complete the matching exercises in [appendix B](./appendix/b_exercises.md).
* Solutions will be available a few days after the lecture.

---

## Evaluation
* What's the difference between `ml::dense_layer::Interface` and your concrete class, and why is this split beneficial?
* Which activation functions does your implementation support, and when is each one suitable?
* Can you connect the mathematical formulas for backpropagation to your implementation; what does each computation step in the code represent?
* How do you test that the implementation is correct, and how do you know when the network is sufficiently trained?
* Why can a model with very low training error still perform poorly on new data?
* What does it mean if training error keeps falling but test error stops improving or starts rising?
* Why does momentum tend to converge faster than plain SGD on noisy, per-sample training?
* Why does an unstable network in this course tend to go silent (flat precision) rather than print `nan` or `inf`?

---

## Next Lecture
* Introduction to convolutional neural networks (CNNs) for image classification.

---
