# L07 - Convolutional Neural Networks (Part II)

## Agenda
* Finish implementing the conv layer struct (`ml::ConvLayer`), started in L06.
* Implement a max pooling layer struct (`ml::MaxPoolLayer`) in software.
* Implement a flatten layer struct (`ml::FlattenLayer`) in software.

---

## Objectives
* Be able to implement a simple conv, max pooling, and flatten layer in software.

---

## Instructions
* Use your code from L06: open [conv_demo.cpp](../L06/conv_layer/cpp/conv_demo.cpp) and finish the
`ml::ConvLayer` struct you started. See [appendix A](./appendix/a_conv_layer.md) for the wrap-up
steps.
* Open [max_pool_demo.cpp](./max_pool_layer/cpp/max_pool_demo.cpp). A struct named
`ml::MaxPoolLayer` should be added to this file to implement a simple max pooling layer. \
See [appendix B](./appendix/b_max_pool_layer.md) for more information.
* Open [flatten_demo.cpp](./flatten_layer/cpp/flatten_demo.cpp). A struct named
`ml::FlattenLayer` should be added to this file to implement a simple flatten layer. \
See [appendix C](./appendix/c_flatten_layer.md) for more information.
* As a reference for both, see the corresponding implementations in Python
([max_pool_demo.py](./max_pool_layer/python/max_pool_demo.py),
[flatten_demo.py](./flatten_layer/python/flatten_demo.py)). \
Test-run them with the command `python3 <name>_demo.py` in the terminal, in the corresponding directory.
* If you're interested, more developed implementations exist in C for both
([max_pool_layer/c](./max_pool_layer/c/source/max_pool_demo.c),
[flatten_layer/c](./flatten_layer/c/source/flatten_demo.c)), demonstrating encapsulation via an
opaque struct with accessor functions only exposed through the corresponding header file.

---

## Evaluation
* Which parts of your `ml::ConvLayer` implementation correspond to feedforward and backpropagation, respectively?
* Why do we use the same kernel (the same weights) across the entire image instead of unique weights per position?
* How does your `ml::MaxPoolLayer` implementation know which positions the gradients should be propagated back to during backpropagation?
* Why is a flatten layer needed between the convolutional/pooling layers and the dense layer?

---

## Next Lecture
* Implementing the conv layer in the real, polymorphic CNN pipeline (`cnn_work`), replacing the stub.

---
