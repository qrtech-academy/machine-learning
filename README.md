# Machine Learning

## About the Course
The course covers the fundamentals of machine learning, focusing on:
* Linear regression and gradient-based optimization.
* Neural networks: feedforward, backpropagation, and activation functions.
* Convolutional neural networks (CNNs) for image classification: kernels, pooling, and flatten layers.
* Implementing ML algorithms from scratch in C++, without external ML libraries.

The course is split into two parts, with a natural break point in between:
* **Part I** (L01-L05): Linear regression and neural network fundamentals, culminating in a from-scratch dense-layer implementation.
* **Part II** (L06-L09): Convolutional neural networks — convolutional, pooling, and flatten layers,
  followed by a full-course review (L10) that ties linear regression, neural networks, and CNNs
  together.

See [info/README.md](./info/README.md) for the full week-by-week schedule.

The course includes hands-on exercises covering the implementation of:
* An adaptive linear regression model.
* A simple neural network (dense layer) from scratch in C++.
* Convolutional, pooling, and flatten layers for convolutional neural networks.

After completing the course, participants should be able to:
* Implement basic ML algorithms from scratch in C++.
* Train and evaluate neural networks, both by hand and in software.
* Reason about design choices and limitations.

---

## Structure

```text
ci/          Scripts for code formatting and compilation
info/        Course information and schedule
lectures/    Lecture plans, literature, and exercises
```

---

## Code Formatting
`ci/format.sh` formats C/C++ code with `clang-format` and Python code with `black`:

```bash
ci/format.sh          # Format all files.
ci/format.sh --check  # Check formatting without modifying files.
```

`clang-format` and `black` need to be installed and available on `PATH` before running the
script.

`clang-format` is installed via `apt` on WSL:

```bash
sudo apt -y update
sudo apt -y install clang-format
```

`black` is installed via `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## Compilation
`ci/build.sh` builds every lecture demo that has a `Makefile`:

```bash
ci/build.sh
```

A C++17-capable compiler (e.g. `g++`) needs to be installed and available on `PATH` before running
the script.

---
