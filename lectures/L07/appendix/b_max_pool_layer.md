# Appendix B - Creating a Simple Max Pooling Layer in C++

### Task description
A struct named `ml::MaxPoolLayer` should be added to
[max_pool_demo.cpp](../max_pool_layer/cpp/max_pool_demo.cpp) to implement a simple
max pooling layer. To keep things as simple as possible, we implement a struct and skip
get/set methods, deletion of copy and move constructors, and so on.

![](./images/maxpool_layer.png)

Study the code in the `main()` function. Your implementation should be written so this code works to
create and use a max pooling layer named `poolLayer`:

```cpp
// Create a max pooling layer: 4x4 input, 2x2 pooling regions, produces 2x2 output.
constexpr std::size_t inputSize{4U};
constexpr std::size_t poolSize{2U};
ml::MaxPoolLayer poolLayer{inputSize, poolSize};

// Example 4x4 input matrix (could represent an image or feature map).
const Matrix2d input{{2, 1, 6, 1},
                     {3, 0, 4, 6},
                     {1, 2, 4, 5},
                     {3, 4, 7, 7}};
        
// Perform feedforward (pooling).
poolLayer.feedforward(input);

// Example output gradients (same shape as pooling output, used for backpropagation demo).
const Matrix2d outputGradients{{1, 2},
                               {3, 4}};
// Perform backpropagation.
poolLayer.backpropagate(outputGradients);
```

### Compiling and running the program
As usual, the program can be run by typing the command `make` in the terminal:

```bash
make
```

Once the implementation is complete, uncomment the compiler flag `CXX_FLAGS` in
the [Makefile](../max_pool_layer/cpp/Makefile). That is, change the following:

```bash
# TODO: Uncomment this line once the implementation is finished.
#CXX_FLAGS := -std=c++17 -Wall -Werror -DMAX_POOL_LAYER_IMPLEMENTED
```

to

```bash
CXX_FLAGS := -std=c++17 -Wall -Werror -DMAX_POOL_LAYER_IMPLEMENTED
```

You can then also remove the header guard `MAX_POOL_LAYER_IMPLEMENTED` from
[max_pool_demo.cpp](../max_pool_layer/cpp/max_pool_demo.cpp) if you like. In that case, change
the following:

```cpp
/**
 * @brief Create and demonstrate a simple max pooling layer.
 * 
 * @return 0 on success, -1 on failure.
 */
int main()
{
//! @todo Remove this header guard (and/or uncomment the compiler flags in the makefile) once the
//        implementation is finished.
#ifdef MAX_POOL_LAYER_IMPLEMENTED
    
    // Function content.
    return 0;

//! @todo Remove this header guard (and/or uncomment the compiler flags in the makefile) once the
//        implementation is finished.
#endif /** MAX_POOL_LAYER_IMPLEMENTED */
}
```

to

```cpp
/**
 * @brief Create and demonstrate a simple max pooling layer.
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
