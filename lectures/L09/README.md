# L09 - Convolutional Neural Networks (Part IV)

## Agenda
* Implementing a real, working max pooling layer, `ml::conv_layer::MaxPool`, in the polymorphic
`cnn_work` pipeline: replacing the `MaxPoolStub`.

---

## Objectives
* Have created a concrete subclass `MaxPool` that inherits `ml::conv_layer::Interface`.
* Have implemented `feedforward()`, `backpropagate()`, and `optimize()` for `MaxPool`.

---

## Instructions
* Use your code from L08 ([cnn_work](../L08/cnn_work)): keep working in the same project.
* Open [max_pool.hpp](../L08/cnn_work/include/ml/conv_layer/max_pool.hpp) and
[max_pool.cpp](../L08/cnn_work/source/ml/conv_layer/max_pool.cpp) (currently placeholders) and
implement the class `ml::conv_layer::MaxPool`, satisfying the same
[`conv_layer::Interface`](../L08/cnn_work/include/ml/conv_layer/interface.hpp) as `Conv`. See
[appendix A](./appendix/a_max_pool_layer.md) for the full spec, including how to wire it into
`factory.cpp` in place of `MaxPoolStub`.

---

## Evaluation
* How does your `ml::conv_layer::MaxPool` implementation know which positions the gradients should be propagated back to during backpropagation?
* What differences exist between a max pooling layer and a conv or dense layer, in terms of trainable parameters?
* Why can `Conv` and `MaxPool` share a single interface (`conv_layer::Interface`) despite one having trainable parameters and the other not?

---

## Next Lecture
* Implementing the flatten layer, and wiring the complete CNN together as the course's capstone.

---
