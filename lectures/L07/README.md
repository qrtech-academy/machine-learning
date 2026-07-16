# L07 - Convolutional Neural Networks (Part II)

## Agenda
* Implementing a conv layer in software.

---

## Objectives
* Be able to implement a simple conv layer in software.

---

## Instructions
* Open the file [conv_demo.cpp](./conv_layer/cpp/conv_demo.cpp). A struct named `ml::ConvLayer` is to be be added to this
file to implement a simple conv layer. \
See [appendix A](./appendix/a_conv_layer.md) for more information.
* As a reference, see the corresponding implementation in Python [here](./conv_layer/python/conv_demo.py). \
Test-run the program with the command `python3 conv_demo.py` in the terminal, in the corresponding directory.
* If you're interested, a somewhat more developed implementation exists in C [here](./conv_layer/c/source/conv_demo.c).
    * This implementation demonstrates encapsulation, since the struct `conv_layer_t` is only implemented in the source file 
    [conv_layer.c](./conv_layer/c/source/ml/conv_layer.c). 
    * The corresponding header file [conv_layer.h](./conv_layer/c/include/ml/conv_layer.h) only contains functions 
    for using the conv layer.
    * The struct's member variables are therefore kept private, in the same way the `private` keyword is used in C++. 
    * This also demonstrates how matrices can be implemented in C via the struct [matrix_t](./conv_layer/c/include/ml/matrix.h).

---

## Evaluation
* Which parts of your `ml::ConvLayer` implementation correspond to feedforward and backpropagation, respectively?
* Why do we use the same kernel (the same weights) across the entire image instead of unique weights per position?

---

## Next Lecture
* Implementing a max pooling layer in software.

---
