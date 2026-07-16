# L10 - Course Review

## Agenda
* Review: how linear regression, neural networks, and CNNs are all the same underlying pattern —
  feedforward, backpropagation, optimization — applied to increasingly structured layers.
* Live walkthrough of the complete `cnn_demo` pipeline (conv → max pooling → flatten → dense) as
  the course's capstone example, tying every layer type together.
* Closing topic: what changes when a network like the one built in this course has to run on a
  resource-constrained target instead of a laptop.

---

## Objectives
* Be able to explain, end-to-end, how a trained CNN goes from raw pixel input to a classification prediction.
* Recognize the shared feedforward/backpropagate/optimize pattern common to every layer type built in this course.
* Be able to compare the layer types built in this course (`Fixed`, `Dense`, `Conv`, `MaxPool`, `Flatten`) in terms of trainable parameters, computation, and role in the network.
* Be able to reason about memory footprint, numeric precision, and real-time constraints when deploying a trained network outside of a general-purpose computer.

---

## Instructions
* Read [appendix A](./appendix/a_review.md) for a summary of every layer type built in this course,
and how they connect in the full `cnn_demo` pipeline.
* Revisit [cnn_demo](../L06/cnn_demo) (from L06) — walk through `Cnn::feedforward()`,
`backpropagate()`, and `optimize()` (`source/ml/cnn/cnn.cpp`) end-to-end as a class, tracing one
training example through every layer.
* Work through [appendix B](./appendix/b_exercises.md) for self-assessment: recognizing layer types
from their algorithms, implementation/shape/parameter-count questions, and a set of exam-style
questions covering the whole course. Solutions are in [appendix D](./appendix/d_solutions.md).
* To wrap up the course, read [appendix C](./appendix/c_embedded_constraints.md) for a discussion of
what changes when a network like the one you've built during this course has to run on a
resource-constrained target, and complete the memory-footprint and fixed-point exercises there.

---

## Evaluation
* Can you trace a single training example through the entire `cnn_demo` pipeline, from feedforward through backpropagation to weight updates, describing what happens at each layer?
* What do all the layer types built in this course have in common in their interface (feedforward/backpropagate/optimize), and where do they genuinely differ?
* Which layer types have trainable parameters, and which don't? Why?
* If you were designing a new CNN for a different image classification task, how would you decide how many conv/pool layers to use, and why?
* Why might a network that runs comfortably on a laptop fail to fit, or fail to meet a deadline, on a microcontroller?
* What's the trade-off between fixed-point and floating-point arithmetic, and why does it matter more on some targets than others?

---

## Next Lecture
* No more lectures — thank you for taking this course!

---
