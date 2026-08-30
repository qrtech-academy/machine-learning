# L04 - Neural Networks (Part II)

## Agenda
* Continuing the exercise: creating a neural network interface and a concrete `Shallow` class that uses the `dense_layer::Interface`/`Stub` pair from last lecture.
* Implementing the network's training method, including the precision it measures itself by and the adaptive learning rate it derives from that measurement.

---

## Objectives
* Have created an interface for a neural network.
* Be able to connect components via interfaces.
* Have implemented a concrete class for a simple neural network, including a fully working training method.
* Be able to explain how a model can set its own learning rate, and why a training method that resets its parameters first is worth having.

---

## Instructions

### Preparation
* Read through [appendix A](./appendix/a_theory.md), covering the neural network class you'll be building on top of last lecture's `dense_layer::Interface`/`Stub`, and how the training loop maps onto L03's theory.

### During the Lecture
* Participate in the lecture (live coding session).

### After the lecture
* Complete the exercises in [appendix B](./appendix/b_exercises.md): declaring the neural network class and implementing its training method, with the adaptive learning rate from **L02** carried over to it.
* Verify your implementation with the [test suite](./appendix/exercises/test/README.md).
* Solutions will be available a few days after the lecture.

---

## Evaluation
* Why is it beneficial to define an interface for the neural network rather than implementing a class directly?
* Without looking at the code, can you describe how the components in the network are connected via interfaces?
* Can you explain the flow from input to prediction in your implementation, step by step?
* Why doesn't `feedforward()` return the layer's output, and how does the rest of the network access it instead?
* Why is the precision evaluated every hundredth epoch here, rather than every tenth as in L02, and what does the network do with the number?
* Why does `train()` reset both layers before its first epoch, and what would calling it twice do without that reset?

---

## Next Lecture
* Implementing a real dense layer in software, replacing the stub used since L03.

---
