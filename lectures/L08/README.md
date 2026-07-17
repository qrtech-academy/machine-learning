# L08 - Convolutional Neural Networks (Part III)

## Agenda
* Implementing a real, working convolutional layer, `ml::conv_layer::Conv`, in the polymorphic
`cnn_work` pipeline: replacing the `ConvStub` currently used for it.

---

## Objectives
* Have created a concrete subclass `Conv` that inherits `ml::conv_layer::Interface`.
* Have implemented `feedforward()`, `backpropagate()`, and `optimize()` for `Conv`.

---

## Instructions
* [cnn_work](./cnn_work) is the polymorphic CNN pipeline, wired together but currently running on
stub layers (see `factory.cpp`); build and run it with `make` first to see the meaningless stub
output, then start replacing layers with real ones. This is your own copy from here on; you'll
keep building it up through L09 and L10.
* Open [conv.hpp](./cnn_work/include/ml/conv_layer/conv.hpp) and
[conv.cpp](./cnn_work/source/ml/conv_layer/conv.cpp) (currently placeholders) and implement
the class `ml::conv_layer::Conv`, satisfying
[`conv_layer::Interface`](./cnn_work/include/ml/conv_layer/interface.hpp). See
[appendix A](./appendix/a_conv_layer.md) for the full spec, including how to wire it into
`factory.cpp` in place of `ConvStub`.
* Note: the rest of `cnn_work` (max pooling, flatten) is still stubbed after this lecture, so
running the program won't yet produce meaningful predictions: that happens once every layer is
real, at the end of L10. In the meantime,
[dense.hpp](./cnn_work/include/ml/dense_layer/dense.hpp)/[dense.cpp](./cnn_work/source/ml/dense_layer/dense.cpp)
are left in the project, already fully implemented, as a working example of what a finished layer
class looks like.

---

## Evaluation
* What's the difference between `ml::conv_layer::Interface` and your concrete class `Conv`, and why is this split beneficial?
* Can you connect the mathematical formulas from L06's appendix B/C to each computation step in your `feedforward()`/`backpropagate()`?
* How does your `ml::conv_layer::Conv` compare to the `ml::ConvLayer` struct you wrote in L06/L07: same math, what's different about the design?
* Why do `Conv` and `MaxPool` (next lecture) share a single interface, `conv_layer::Interface`?

---

## Next Lecture
* Implementing the max pooling layer in the CNN pipeline.

---
