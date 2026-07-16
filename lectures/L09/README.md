# L09 - Convolutional Neural Networks (Part IV)

## Agenda
* Implementing a flatten layer in software.

---

## Objectives
* Be able to implement a simple flatten layer in software.

---

## Instructions
* Open the file [flatten_demo.cpp](./flatten_layer/cpp/flatten_demo.cpp). 
A struct named `ml::FlattenLayer` is to be added to this file to implement a simple flatten layer. \
See [appendix A](./appendix/a_flatten_layer.md) for more information.
* As a reference, see the corresponding implementation in Python [here](./flatten_layer/python/flatten_demo.py). \
Test-run the program with the command `python3 flatten_demo.py` in the terminal, in the corresponding directory.
* If you're interested, a somewhat more developed implementation exists in C [here](./flatten_layer/c/source/flatten_demo.c).
    * This implementation demonstrates encapsulation, since the struct `flatten_layer_t` is only implemented in the source file 
    [flatten_layer.c](./flatten_layer/c/source/ml/flatten_layer.c). 
    * The corresponding header file [flatten_layer.h](./flatten_layer/c/include/ml/flatten_layer.h) only contains functions 
    for using the flatten layer.
    * The struct's member variables are therefore kept private, in the same way the `private` keyword is used in C++. 
    * This also demonstrates how matrices can be implemented in C via the struct [matrix_t](./flatten_layer/c/include/ml/matrix.h).

---

## Evaluation
* Why is a flatten layer needed between the convolutional/pooling layers and the dense layer?
* What does the flatten layer's `backpropagate()` do, and how does that computation differ from `feedforward()`?

---

## Next Lecture
* Course review: putting linear regression, neural networks, and CNNs together.

---
