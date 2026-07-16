# Appendix A - Creating a Simple Conv Layer in C++

### Task description
A struct named `ml::ConvLayer` should be added to [conv_demo.cpp](../conv_layer/cpp/conv_demo.cpp) to implement a simple conv layer. To keep things as simple as possible, we implement a struct and skip get/set methods, deletion of copy and move constructors, and so on.

![](./images/conv_layer.png)

Note the "Dimensions" panel above: `ConvLayer` zero-pads its input (`pad = kernelSize / 2`) so the
output is the **same size as the input**, not smaller. This is why `outputGradients` in the example
below is a 4×4 matrix, matching the 4×4 `input` — your `feedforward()` and `backpropagate()` need to
pad/unpad internally (see the private methods sketched at the bottom of the struct) to make that
size match up.

Study the code in the `main()` function. Your implementation should be written so this code works:

```cpp
// Create a convolutional layer: 4x4 input, 2x2 kernel.
constexpr std::size_t inputSize{4U};
constexpr std::size_t kernelSize{2U};
ml::ConvLayer convLayer{inputSize, kernelSize};

// Example 4x4 input matrix (could represent an image or feature map).
const Matrix2d input{{1, 1, 1, 1},
                     {1, 0, 0, 1},
                     {1, 0, 0, 1},
                     {1, 1, 1, 1}};

// Perform feedforward (convolution).
convLayer.feedforward(input);

// Example output gradients (target output for demonstration).
const Matrix2d outputGradients{{1, 1, 1, 1},
                               {1, 1, 1, 1},
                               {1, 1, 1, 1},
                               {1, 1, 1, 1}};

// Perform backpropagation.
convLayer.backpropagate(outputGradients);
```

### Compiling and running the program
As usual, the program can be run by typing the command `make` in the terminal:

```bash
make
```

Once the implementation is complete, uncomment the compiler flag `CXX_FLAGS` in
the [Makefile](../conv_layer/cpp/Makefile). That is, change the following:

```bash
# TODO: Uncomment this line once the implementation is finished.
#CXX_FLAGS := -std=c++17 -Wall -Werror -DCONV_LAYER_IMPLEMENTED
```

to

```bash
CXX_FLAGS := -std=c++17 -Wall -Werror -DCONV_LAYER_IMPLEMENTED
```

You can then also remove the header guard `CONV_LAYER_IMPLEMENTED` from
[conv_demo.cpp](../conv_layer/cpp/conv_demo.cpp) if you like. In that case, change the following:

```cpp
/**
 * @brief Create and demonstrate a simple convolutional layer.
 *
 * @return 0 on success, -1 on failure.
 */
int main()
{
//! @todo Remove this header guard (and/or uncomment the compiler flags in the makefile) once the
//        implementation is finished.
#ifdef CONV_LAYER_IMPLEMENTED

    // Function content.
    return 0;
    
//! @todo Remove this header guard (and/or uncomment the compiler flags in the makefile) once the
//        implementation is finished.
#endif /** CONV_LAYER_IMPLEMENTED */
}
```

to

```cpp
/**
 * @brief Create and demonstrate a simple convolutional layer.
 *
 * @return 0 on success, -1 on failure.
 */
int main()
{
    // Function content.
    return 0;
}
```

---
