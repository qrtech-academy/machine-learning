# Appendix A - Finishing the Conv Layer
You started implementing `ml::ConvLayer` in [L06](../../L06/appendix/d_conv_layer.md): finish it
here. The full task description, the diagram, and the target `main()` usage are all in
[L06's appendix D](../../L06/appendix/d_conv_layer.md); this appendix only covers the wrap-up steps
for compiling and running it for the first time, in
[conv_demo.cpp](../../L06/conv_layer/cpp/conv_demo.cpp).

---

## Compiling and running the program
As usual, the program can be run by typing the command `make` in the terminal:

```bash
make
```

Once the implementation is complete, uncomment the compiler flag `CXX_FLAGS` in
the [Makefile](../../L06/conv_layer/cpp/Makefile). That is, change the following:

```bash
# TODO: Uncomment this line once the implementation is finished.
#CXX_FLAGS := -std=c++17 -Wall -Werror -DCONV_LAYER_IMPLEMENTED
```

to

```bash
CXX_FLAGS := -std=c++17 -Wall -Werror -DCONV_LAYER_IMPLEMENTED
```

You can then also remove the header guard `CONV_LAYER_IMPLEMENTED` from
[conv_demo.cpp](../../L06/conv_layer/cpp/conv_demo.cpp) if you like. In that case, change the following:

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
