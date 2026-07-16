# Appendix C - Constraints on Resource-Constrained Targets

## Overview
Everything built during this course has run on a general-purpose computer, where memory is
effectively unlimited and every CPU core has a hardware floating-point unit (FPU). Deploying the
same kind of network to a microcontroller or other resource-constrained target introduces
constraints that never show up on a laptop. This appendix surveys the main ones, then walks
through two of them hands-on: memory footprint and fixed-point arithmetic.

---

## 1. Memory Footprint
Every weight and bias your network learned during training has to be stored somewhere on the
target device, typically in flash (read-only after deployment) or RAM (if the network keeps
learning on-device). The `std::vector<double>` used throughout this course stores each value as
an 8-byte `double`.

Consider the small hidden/output layer pair many of the exercises built (3 nodes/2 weights in the
hidden layer, 1 node/3 weights in the output layer):

```
Hidden layer: 3 * 2 weights + 3 biases  =  9 values
Output layer: 1 * 3 weights + 1 bias    =  4 values
Total: 13 doubles * 8 bytes             =  104 bytes
```

That's negligible on any target. But course-sized examples are intentionally tiny. A single hidden
layer sized for 28×28 pixel images (784 inputs) with 128 nodes — a very ordinary size for a first
hidden layer — tells a different story:

```
Weights: 128 * 784 = 100,352
Biases:  128
Total:   100,480 values

As double (8 bytes each): 803,840 bytes  ≈ 785 KB
As float  (4 bytes each): 401,920 bytes  ≈ 392.5 KB
As int8   (1 byte each):  100,480 bytes  ≈ 98.125 KB
```

That's **one layer**. Many microcontrollers have on the order of tens of KB of RAM; mid-range parts
often have a few hundred KB; only higher-end parts reach into the multiple-MB range. The `double`
version of this single layer already exceeds what a lot of embedded targets have available for
their *entire* program — and a real network has several layers. This is why embedded ML almost
always trains at full precision on a workstation and then **quantizes** the trained weights down to
a smaller type before deployment (see below), rather than training or storing at `double` precision
on the target itself.

---

## 2. Precision: Floating-Point vs. Fixed-Point
Many microcontrollers — especially smaller Cortex-M0/M0+ parts or 8/16-bit MCUs — have no hardware
FPU. Floating-point operations on those targets are emulated in software, which can be an order of
magnitude or more slower than an integer add or multiply.

**Fixed-point** arithmetic sidesteps this by representing a real number as a scaled integer instead
of a `float`/`double`. A common convention is Q*m*.*n* notation: *m* integer bits, *n* fractional
bits. In **Q16.16** format, a 32-bit signed integer represents a real number scaled by 2^16
(65536):

```cpp
using Fixed16_16 = std::int32_t;

constexpr int kFractionalBits{16};
constexpr Fixed16_16 kOne{1 << kFractionalBits};  // 65536

// Convert a floating-point value to Q16.16 fixed-point.
Fixed16_16 toFixed(const double value) noexcept
{
    return static_cast<Fixed16_16>(value * kOne);
}

// Convert a Q16.16 fixed-point value back to floating-point.
double toDouble(const Fixed16_16 value) noexcept
{
    return static_cast<double>(value) / kOne;
}
```

Addition of two fixed-point values is just integer addition — no special handling needed, since
both operands are scaled by the same factor. Multiplication needs more care: multiplying two
Q16.16 values naively overflows a 32-bit integer almost immediately (two values scaled by 2^16
multiply to a result scaled by 2^32), so the multiply has to happen in a wider type before scaling
back down:

```cpp
// Multiply two Q16.16 fixed-point values.
Fixed16_16 fixedMultiply(const Fixed16_16 a, const Fixed16_16 b) noexcept
{
    const auto product = static_cast<std::int64_t>(a) * static_cast<std::int64_t>(b);
    return static_cast<Fixed16_16>(product >> kFractionalBits);
}
```

The trade-off: fixed-point trades dynamic range and some precision (Q16.16 can represent roughly
±32,767 with a resolution of about 0.0000153) for arithmetic that's fast and deterministic on
targets without an FPU — no software floating-point emulation involved.

---

## 3. Quantization
Even on targets *with* an FPU, storing weights as 8-bit integers instead of 32-bit floats cuts
memory 4x, and many embedded cores can execute int8 operations faster than float32 ones. The
common approach — **post-training quantization** — trains the network at full precision as done
throughout this course, then converts the trained weights to a smaller integer type for
deployment, typically alongside a per-layer scale factor (conceptually similar to the fixed-point
conversion above, but chosen per-tensor rather than a single fixed format for the whole network). A
full quantization pipeline is out of scope for this course, but it's worth recognizing the idea:
*train* at high precision, *deploy* at low precision.

---

## 4. Real-Time and Latency Constraints
A conv layer performs on the order of `inputSize² × kernelSize²` multiply-adds per feedforward
pass. On a laptop that's negligible; inside a control loop with a hard deadline — e.g. a motor
controller running at 1 kHz, giving the entire loop (sensor read, inference, actuation) a 1 ms
budget — it can dominate the loop's execution time. For embedded systems, the *worst-case*
execution time matters far more than the average case: a network that's usually fast but
occasionally spikes in latency can violate a real-time deadline in exactly the way that doesn't
show up during casual testing.

---

## 5. Dynamic Memory Allocation
The `Dense`, `ConvLayer`, `MaxPoolLayer`, and `FlattenLayer` implementations built during this
course all use `std::vector`, which allocates on the heap. On many embedded targets, heap
allocation is avoided or tightly controlled: long-running systems risk heap fragmentation over
time, allocation latency isn't constant-time, and some bare-metal targets have no heap at all. A
common alternative is sizing everything at compile time with `std::array` (or plain C arrays)
instead of `std::vector`, trading the convenience of runtime-sized layers for fully static, more
predictable memory use.

---

## Exercise

### Part 1 — Memory Footprint
Using the layer sizes from the network you built in **L04–L05** (or any other layer configuration
of your choice):
1. Compute the total number of trainable values (weights + biases) across all layers.
2. Compute the memory footprint at `double` (8 bytes), `float` (4 bytes), and `int8` (1 byte) per
   value.
3. Compare your numbers against the MNIST-sized example above (785 KB / 392.5 KB / 98.125 KB). Which
   of the three precisions would plausibly fit into a microcontroller with 64 KB of RAM? With 512 KB?

### Part 2 — Fixed-Point Conversion
Take the `predict()` method from **L01**'s `ml::lin_reg::Fixed` class (`y = k * x + m`):
1. Add `toFixed()`, `toDouble()`, and `fixedMultiply()` as shown above (an addition helper isn't
   needed — plain integer `+` already works for Q16.16 values).
2. Implement a free function `Fixed16_16 predictFixed(Fixed16_16 weight, Fixed16_16 bias, Fixed16_16 input)`
   that computes the same prediction as `predict()`, but entirely in Q16.16 fixed-point using
   `fixedMultiply()` for the multiplication.
3. In `main()`, pick a few `(weight, bias, input)` triples, convert them to fixed-point, run both
   `predict()` (floating-point) and `predictFixed()` (fixed-point), convert the fixed-point result
   back with `toDouble()`, and print both side by side.
4. Confirm the two results agree to within roughly 0.0001 for each test case. If they don't,
   double-check the `>>` shift in `fixedMultiply()` — it's the step most likely to be off by a
   factor of `kOne`.

---
