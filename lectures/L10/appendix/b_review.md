# Appendix B - Course Review

## Overview
Every layer built in this course (from L01's linear regression up through L06-L10's CNN layers)
follows the same three-step pattern: **feedforward** (compute a prediction from an input),
**backpropagate** (compute how wrong that prediction was, and route the blame backward), and
**optimize** (adjust trainable parameters to reduce that error next time). What changes from layer
to layer is *what* the parameters are, *how* the feedforward sum is computed, and *whether* there
are trainable parameters at all.

---

## 1. Every Layer Type, Side by Side

| Layer | From | Trainable parameters | Feedforward | Backpropagate | Optimize |
|---|---|---|---|---|---|
| `Fixed` (linear regression) | L01-L02 | weight `k`, bias `m` | `y = k*x + m` | `δ = y_ref - y_p` | `k,m += δ * LR * (x or 1)` |
| `Dense` | L03-L05 | weights + bias per node | `y = σ(b + Σ w*x)` | `Δe = δ * σ'(s)` per node | `w,b += Δe * LR * input` |
| `Conv` | L06-L08 | one shared kernel + bias | same as `Dense`, but local (small window) and shared (same kernel at every position) | gradients accumulated across every position the kernel visited, then applied once | `kernel,bias += gradient * LR` |
| `MaxPool` | L06-L07, L09 | none | passes the largest value in each block forward | routes the incoming gradient only to the position that held the max value; everywhere else gets `0` | no-op: nothing to update |
| `Flatten` | L06-L07, L10 | none | reshapes 2D → 1D | reshapes 1D → 2D (the exact inverse) | doesn't exist in the interface: there's nothing to optimize |

A few things worth noticing across the row:
* **Only `Conv` and `Dense` have trainable parameters.** `MaxPool` and `Flatten` are purely
  structural: they reshape or downsample data, but never learn anything themselves.
* **`Conv` is `Dense` with two constraints added**: locality (each output only looks at a small
  window of the input) and weight sharing (the same kernel is reused everywhere instead of a
  unique weight per input/output pair). That's the entire conceptual jump from L05 to L06-L07: see
  L06's appendix A if you want the full "why" behind those two constraints.
* **`Fixed` is a `Dense` layer with one node, no activation function, and no hidden layer around
  it.** The same `error → adjustment → update` shape from L01 reappears, unchanged in spirit,
  all the way through the CNN layers.

---

## 2. The Full Pipeline (`cnn_demo`, from L10)
`ml::cnn::Cnn` (see [`cnn_demo`](../cnn_demo), the finished reference implementation) wires four of
these layer types together into one working classifier. This is the same pipeline you built up
yourself as `cnn_work` across L08-L10; worth comparing the two if anything in your own version
doesn't behave the way you expect:

```
input → Conv → MaxPool → Flatten → Dense → prediction
```

**Feedforward** runs left to right, each layer consuming the previous layer's `output()`.

**Backpropagation** runs right to left: the target vector goes into `Dense`'s `backpropagate()`
first, and each earlier layer is then fed the next layer's `inputGradients()`: `Dense` → `Flatten`
→ `MaxPool` → `Conv`.

**Optimization** updates every layer's parameters (`Conv` and `Dense` only) using the gradients
already computed by `backpropagate()`.

See `cnn_demo/README.md` for the full method-by-method breakdown, including the exact formulas
used in the C++ implementation.

---
