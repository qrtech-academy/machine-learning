# Course Info

## Teacher
Erik Pihl ([erik.axel.pihl@gmail.com](mailto:erik.axel.pihl@gmail.com))

---

## Course Plan – Machine Learning

### Part I – Linear Regression & Neural Networks
| Week | Lecture | Topic |
|------|---------|-------|
| 1  | L01 | Linear regression (part I) |
| 2  | L02 | Linear regression (part II) |
| 3  | L03 | Neural networks (part I) |
| 4  | L04 | Neural networks (part II) |
| 5  | L05 | Dense layer |

### Part II – Convolutional Neural Networks
| Week | Lecture | Topic |
|------|---------|-------|
| 6  | L06 | Convolutional neural networks (part I) |
| 7  | L07 | Convolutional neural networks (part II) |
| 8  | L08 | Convolutional neural networks (part III) |
| 9  | L09 | Convolutional neural networks (part IV) |
| 10 | L10 | Course review |

---

## Lecture Content

### L01 – Linear Regression (Part I)
Introduction to machine learning and the first regression model.

Topics include:
* What is machine learning? (input/rules/output vs. traditional programming)
* Brief introduction to `std::vector`
* Linear regression: prediction, parameters (weight/bias), error, epochs, learning rate
* Training a linear regression model by hand
* Implementing and training the model in software
* Interface-based model design

### L02 – Linear Regression (Part II)
Improving the regression training loop.

Topics include:
* Randomizing the training order each epoch
* One-time RNG initialization (`initRandom()`, static local variable)
* Computing model precision
* Threshold-based early stopping

### L03 – Neural Networks (Part I)
Feedforward, backpropagation, and the dense-layer interface.

Topics include:
* Neural network concepts: layers, nodes, weights
* Feedforward and backpropagation
* Gradient descent and activation functions
* Training neural networks by hand
* Dense layer interface and a stub implementation

### L04 – Neural Networks (Part II)
Assembling a network from interfaces.

Topics include:
* Neural network interface
* Connecting components via interfaces
* Concrete `Shallow` network class (built on the `dense_layer::Interface`/`Stub`)
* Implementing the training method

### L05 – Dense Layer
A real, from-scratch dense layer.

Topics include:
* Concrete `Dense` subclass replacing the stub
* Activation functions
* `feedforward()`, `backpropagate()`, `optimize()`
* Overfitting and generalization (train/test error gap)
* Momentum-based optimizer
* Recognizing and debugging numerical instability

### L06 – Convolutional Neural Networks (Part I)
Why CNNs, and the convolutional layer.

Topics include:
* Why dense-only networks aren't suited to image classification
* CNN structure: convolutional, pooling, flatten, dense layers
* Feature extraction via kernels
* Feedforward through a small CNN by hand
* Starting the `ml::ConvLayer` struct

### L07 – Convolutional Neural Networks (Part II)
Conv, pooling, and flatten layers in software.

Topics include:
* Finishing the conv layer (`ml::ConvLayer`)
* Max pooling layer (`ml::MaxPoolLayer`)
* Flatten layer (`ml::FlattenLayer`)
* Weight sharing (same kernel across the image)
* Python/C reference implementations; encapsulation via opaque structs

### L08 – Convolutional Neural Networks (Part III)
A real convolutional layer in the CNN pipeline.

Topics include:
* The polymorphic `cnn_work` pipeline
* Concrete `Conv` class implementing `conv_layer::Interface`
* `feedforward()`, `backpropagate()`, `optimize()`
* Replacing `ConvStub` in `factory.cpp`
* Interface vs. concrete class

### L09 – Convolutional Neural Networks (Part IV)
A real max pooling layer.

Topics include:
* Concrete `MaxPool` class on the shared `conv_layer::Interface`
* `feedforward()`, `backpropagate()`, `optimize()`
* Gradient routing in max pooling
* Layers with vs. without trainable parameters
* Replacing `MaxPoolStub`

### L10 – Flatten Layer & Course Review
Completing and reviewing the CNN.

Topics include:
* Concrete `Flatten` class (`feedforward()`/`backpropagate()`, no `optimize()`)
* Wiring the full CNN: `input → Conv → MaxPool → Flatten → Dense → prediction`
* Switching `factory.cpp` from stubs to real classes
* Review: the shared feedforward/backpropagate/optimize pattern
* Comparing layer types (`Fixed`, `Dense`, `Conv`, `MaxPool`, `Flatten`)
* Deploying on resource-constrained targets (memory, numeric precision, real-time)

---

## Course Material

### Literature
* Consists of text documents attached in connection with the lectures where they're covered.

### Software
* [Visual Studio Code](https://code.visualstudio.com/download):
    * Primary editor.
* [Online C compiler](https://www.onlinegdb.com/online_c_compiler):
    * For compiling and running smaller programs in a terminal environment.
    * To compile and run C++ code, switch the language to **C++17** in the top-right corner.

---
