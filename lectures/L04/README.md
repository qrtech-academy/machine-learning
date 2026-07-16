# L04 - Neural Networks (Part II)

## Agenda
* Continuing the exercise: creating a neural network interface and a concrete `Shallow` class that uses the `dense_layer::Interface`/`Stub` pair from last lecture.
* Implementing the network's training method.

---

## Objectives
* Have created an interface for a neural network.
* Be able to connect components via interfaces.
* Have implemented a concrete class for a simple neural network, including a fully working training method.

---

## Instructions

### Preparation
* Read through [appendix A](./appendix/a_theory.md), covering the neural network class you'll be building on top of last lecture's `dense_layer::Interface`/`Stub`, and how the training loop maps onto L03's theory.

### During the Lecture
* Participate in the lecture (live coding session).

### After the lecture
* Complete the exercises in [appendix B](./appendix/b_exercises.md): declaring the neural network class and implementing its training method.
* Solutions will be available a few days after the lecture.

---

## Evaluation
* Why is it beneficial to define an interface for the neural network rather than implementing a class directly?
* Without looking at the code, can you describe how the components in the network are connected via interfaces?
* Can you explain the flow from input to prediction in your implementation, step by step?
* Why does `feedforward()` return nothing (`void`), and how does the rest of the network access a layer's computed output?

---

## Next Lecture
* Implementing a real dense layer in software, replacing the stub used since L03.

---
