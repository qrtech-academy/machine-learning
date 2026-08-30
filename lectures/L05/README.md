# L05 - Dense Layer

## Agenda
* Implementing a dense layer in software (using the code from the exercise in L04):
    * Creating a concrete subclass `Dense`, taking over from the existing stub class.
    * Adding a handful of activation functions.
* Completing the implementation: feedforward, backpropagation, and optimization.

---

## Objectives
* Have created a concrete subclass that inherits the existing dense layer interface.
* Have implemented all methods, including `initParams()`, `feedforward()`, `backpropagate()`, and `optimize()`.
* Have created a fully functioning dense layer implementation.
* Have tested the implementation with a complete neural network (the one created in the exercise in L04).

---

## Instructions

### Preparation
* Use your code from the exercise in L04.
* Read through [appendix A](./appendix/a_theory.md), covering the dense layer's architecture and how the math maps onto the code.

### During the Lecture
* Participate in the lecture (live coding session).

### After the lecture
* Complete the exercises in [appendix B](./appendix/b_exercises.md): declaring a concrete implementation that inherits `ml::dense_layer::Interface`, then implementing `feedforward()`, `backpropagate()`, and `optimize()`.
* Test the implementation by training the neural network on existing training data, adjusting the number of epochs as needed. There's no learning rate to adjust: the network sets its own, and stops as soon as it reaches its precision threshold.
* Verify your implementation with the [test suite](./appendix/exercises/test/README.md).
* Solutions will be available a few days after the lecture.

---

## Evaluation
* What's the difference between `ml::dense_layer::Interface` and your concrete class, and why is this split beneficial?
* Which activation functions does your implementation support, and when is each one suitable?
* Can you connect the mathematical formulas for backpropagation to your implementation; what does each computation step in the code represent?
* How do you test that the implementation is correct, and how do you know when the network is sufficiently trained?

---

## Next Lecture
* Introduction to convolutional neural networks (CNNs) for image classification.

---
