# L10 - Flatten Layer & Course Review

## Agenda
* Implementing a real, working flatten layer, `ml::flatten_layer::Flatten`, in the polymorphic
`cnn_work` pipeline: replacing the `flatten_layer::Stub`.
* Wiring the complete CNN together: switching `factory.cpp` from stubs to the real classes, and
seeing every layer type work together for the first time.
* Review: how linear regression, neural networks, and CNNs are all the same underlying pattern
  (feedforward, backpropagation, optimization), applied to increasingly structured layers.
* Closing topic: what changes when a network like this has to run on a resource-constrained target
  instead of a laptop.

---

## Objectives
* Have created a concrete subclass `Flatten` that inherits `ml::flatten_layer::Interface`.
* Have implemented `feedforward()` and `backpropagate()` for `Flatten` (note: no `optimize()`; the
interface has none, since a flatten layer has no trainable parameters).
* Have a fully working, end-to-end CNN: `input → Conv → MaxPool → Flatten → Dense → prediction`.
* Recognize the shared feedforward/backpropagate/optimize pattern common to every layer type built in this course.
* Be able to compare the layer types built in this course (`Fixed`, `Dense`, `Conv`, `MaxPool`, `Flatten`) in terms of trainable parameters, computation, and role in the network.
* Be able to reason about memory footprint, numeric precision, and real-time constraints when deploying a trained network outside of a general-purpose computer.

---

## Instructions
* Use your code from L08/L09 ([cnn_work](../L08/cnn_work)): keep working in the same project.
* Open [flatten.hpp](../L08/cnn_work/include/ml/flatten_layer/flatten.hpp) and
[flatten.cpp](../L08/cnn_work/source/ml/flatten_layer/flatten.cpp) (currently placeholders) and
implement the class `ml::flatten_layer::Flatten`. See [appendix A](./appendix/a_flatten_layer.md)
for the full spec, then follow its "Wiring It In" section to add `flatten.cpp` and switch
`factory.cpp` over to the real classes for every layer. Run `make`: compare the (now meaningful)
predictions against the placeholder output you first saw from L08's `cnn_work`.
* Read [appendix B](./appendix/b_review.md) for a summary of every layer type built in this course,
and how they connect in the full `cnn_demo` pipeline (the finished reference copy alongside this
lecture's material).
* Work through [appendix C](./appendix/c_exercises.md) for self-assessment: recognizing layer types
from their algorithms, implementation/shape/parameter-count questions, and a set of exam-style
questions covering the whole course. Solutions are in [appendix E](./appendix/e_solutions.md).
* To wrap up the course, read [appendix D](./appendix/d_embedded_constraints.md) for a discussion of
what changes when a network like the one you've built during this course has to run on a
resource-constrained target, and complete the memory-footprint and fixed-point exercises there.

---

## Evaluation
* Can you trace a single training example through your finished `cnn_work` pipeline, from feedforward through backpropagation to weight updates, describing what happens at each layer?
* What do all the layer types built in this course have in common in their interface (feedforward/backpropagate/optimize), and where do they genuinely differ?
* Which layer types have trainable parameters, and which don't? Why?
* If you were designing a new CNN for a different image classification task, how would you decide how many conv/pool layers to use, and why?
* Why might a network that runs comfortably on a laptop fail to fit, or fail to meet a deadline, on a microcontroller?
* What's the trade-off between fixed-point and floating-point arithmetic, and why does it matter more on some targets than others?

---

## Next Lecture
* No more lectures; thank you for taking this course!

---
