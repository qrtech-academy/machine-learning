# L06 - Convolutional Neural Networks (Part I)

## Agenda
* Introduction to convolutional neural networks (CNNs) for image classification.
* Hand-training a CNN: feedforward, worked through together during the lecture.
* Begin implementing a real convolutional layer in software: the `ml::ConvLayer` struct.

---

## Objectives
* Know why regular neural networks aren't well suited to image classification.
* Know at a high level how convolutional layers are structured.
* Be able to extract features from images via kernels.
* Be able to feed an input through a small CNN by hand (feedforward).
* Have started implementing a conv layer in software (`ml::ConvLayer`).

---

## Instructions
* Read [appendix A](./appendix/a_theory.md) for information about why regular neural networks
aren't well suited to image classification, and how convolutional neural networks are structured.
* Work through the hand-training example in [appendix B](./appendix/b_exercises.md): the feedforward
pass together during the lecture, then backpropagation and optimization for the same example as
self-study.
* Start [appendix C](./appendix/c_conv_layer.md): open
[conv_demo.cpp](./conv_layer/cpp/conv_demo.cpp) and begin implementing the struct `ml::ConvLayer`.
You don't need to finish it this lecture; you'll finish it in L07.

---

## Evaluation
* Why are regular neural networks (dense layers only) less suited to image classification than a CNN?
* Can you, in your own words, describe the purpose of each layer type in a CNN: convolutional, pooling, flatten, and dense?
* Without looking at the appendix, can you describe how a kernel is applied across an image during feedforward in a conv layer?
* Which parts of your `ml::ConvLayer` so far correspond to feedforward?

---

## Next Lecture
* Finishing the conv layer, then implementing max pooling and flatten layers in software.

---
