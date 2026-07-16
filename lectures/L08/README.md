# L08 - Convolutional Neural Networks (Part III)

## Agenda
* Implementing a max pooling layer in software.

---

## Objectives
* Be able to implement a simple max pooling layer in software.

---

## Instructions
* Open the file [max_pool_demo.cpp](./max_pool_layer/cpp/max_pool_demo.cpp). 
A struct named `ml::MaxPoolLayer` is to be added to this file to implement a simple max pooling layer. \
See [appendix A](./appendix/a_max_pool_layer.md) for more information.
* As a reference, see the corresponding implementation in Python [here](./max_pool_layer/python/max_pool_demo.py). \
Test-run the program with the command `python3 max_pool_demo.py` in the terminal, in the corresponding directory.
* If you're interested, a somewhat more developed implementation exists in C [here](./max_pool_layer/c/source/max_pool_demo.c).
    * This implementation demonstrates encapsulation, since the struct `max_pool_layer_t` is only implemented in the source file 
    [max_pool_layer.c](./max_pool_layer/c/source/ml/max_pool_layer.c). 
    * The corresponding header file [max_pool_layer.h](./max_pool_layer/c/include/ml/max_pool_layer.h) only contains functions 
    for using the max pooling layer.
    * The struct's member variables are therefore kept private, in the same way the `private` keyword is used in C++. 
    * This also demonstrates how matrices can be implemented in C via the struct [matrix_t](./max_pool_layer/c/include/ml/matrix.h).

---

## Evaluation
* How does your `ml::MaxPoolLayer` implementation know which positions the gradients should be propagated back to during backpropagation?
* What differences exist between a max pooling layer and a dense or conv layer, e.g. in terms of parameters and computations?

---

## Next Lecture
* Implementing a flatten layer in software.

---
