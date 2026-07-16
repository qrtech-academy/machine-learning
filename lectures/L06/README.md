# L06 - Convolutional Neural Networks (Part I)

## Agenda
* Introduction to convolutional neural networks (CNNs) for image classification.
* Hand-training a convolutional neural network.
* Live demo: a complete CNN classifying digits, built from conv, pooling, flatten, and dense layers.

---

## Objectives
* Know why regular neural networks aren't well suited to image classification.
* Know at a high level how convolutional layers are structured.
* Be able to extract features from images via kernels.
* Be able to train simple convolutional neural networks by hand.
* Have seen a complete, working CNN in action, tying the conv, pooling, flatten, and dense layers together.

---

## Instructions
* Read [appendix A](./appendix/a_theory.md) for information about why regular neural networks
aren't well suited to image classification, and how convolutional neural networks are structured.
* Work through the hand-training example in [appendix B](./appendix/b_exercises.md) together during
the lecture.
* See the [cnn_demo](./cnn_demo) directory for a complete, working CNN (conv → max pooling → flatten
→ dense) that classifies 4×4 pixel-art digits (0-3). Demonstrated live during the lecture; build and
run it with `make` in that directory.

---

## Evaluation
* Why are regular neural networks (dense layers only) less suited to image classification than a CNN?
* Can you, in your own words, describe the purpose of each layer type in a CNN: convolutional, pooling, flatten, and dense?
* Without looking at the appendix, can you describe how a kernel is applied across an image during feedforward in a conv layer?
* Why, during backpropagation, are gradients through a max pooling layer only propagated to the positions that held the maximum value during feedforward?
* In the demo CNN, why does the dense layer's output use one-hot encoding (one node per digit) instead of a single output node?

---

## Next Lecture
* Implementing a conv layer in software.

---
