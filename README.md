# Machine Learning

## About the Course
The course covers the fundamentals of machine learning, focusing on:
* Linear regression and gradient-based optimization.
* Neural networks: feedforward, backpropagation, and activation functions.
* Convolutional neural networks (CNNs) for image classification: kernels, pooling, and flatten layers.
* Implementing ML algorithms from scratch in C++, without external ML libraries.

The course is split into two parts, with a natural break point in between:
* **Part I** (L01-L05): Linear regression and neural network fundamentals, culminating in a from-scratch dense-layer implementation.
* **Part II** (L06-L10): Convolutional neural networks: convolutional, pooling, and flatten layers,
  closing with a full-course review in L10 that ties linear regression, neural networks, and CNNs
  together.

See [info/README.md](./info/README.md) for the full week-by-week schedule.

The course includes hands-on exercises covering the implementation of:
* A linear regression model trained with a fixed learning rate.
* A simple neural network (dense layer) from scratch in C++.
* Convolutional, pooling, and flatten layers for convolutional neural networks.

After completing the course, participants should be able to:
* Implement basic ML algorithms from scratch in C++.
* Train and evaluate neural networks, both by hand and in software.
* Reason about design choices and limitations.

---

## Written Examinations
Nothing in this course is marked. Assessment is the exercises in each lecture's appendix, the test
suite most of them ship with, and the self-assessment questions in
[L10's appendix C](./lectures/L10/appendix/c_exercises.md), each of which checks a component you
built or a number you computed beside it.

[`exam/`](./exam/README.md) holds two four-hour papers with worked solutions, and they check
something else: **your theoretical knowledge, on paper, with nothing in front of you.** Nine
questions each, mixing theory with code, and every derivation the appendices make asked for without
the appendix in the room.

**They gate nothing, they are not a qualification, and no part of the course requires them.**
Nothing in this repository depends on them and `make test` does not know they exist. They are here
purely so students can test their own skills and knowledge after the course, and for the case where
a written result is wanted anyway.

**Take one after the course is over**, after L10. Both papers draw on all ten lectures, so sitting
one partway through examines material nobody has taught you yet, and the result says more about how
far you have read than about what you have understood.

---

## Structure

```text
ci/          Scripts for code formatting, compilation, and testing
exam/        Two written papers and their solutions. Optional, and marked by nobody here.
info/        Course information and schedule
lectures/    Lecture plans, literature, and exercises
libs/        The QAcademy Test framework, checked out as a git submodule
```

---

## Code Formatting
The root `Makefile` formats C/C++ code with `clang-format` and Python code with `black`:

```bash
make format        # Format all files.
make format-check  # Check formatting without modifying files.
```

`clang-format` and `black` need to be installed and available on `PATH` before running these
targets.

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
The root `Makefile` builds every lecture demo that has a `Makefile`:

```bash
make build
```

A C++17-capable compiler (e.g. `g++`) needs to be installed and available on `PATH` before running
this target.

---

## Tests
Most lectures ship a test suite for the exercises in their appendix. The root `Makefile` builds and
runs every suite that has a solution to run against:

```bash
make test
```

The suites are built against the [QAcademy Test](https://github.com/qrtech-academy/test-framework)
framework, which lives in `libs/test` as a git submodule. Check it out once before running any
suite:

```bash
git submodule update --init
```

Solutions are published after each lecture, so `make test` skips the suites whose solution isn't in
place yet and reports how many it ran and how many it skipped. Each suite can also be run on its own
from its own directory with `make`; see the `README.md` in any lecture's `exercises/test/`
directory for the details.

---
