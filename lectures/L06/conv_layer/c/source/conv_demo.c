/**
 * @brief Simple convolutional layer demo.
 *
 *        The layer is driven with the same values as the hand-training example in L06's
 *        appendix B, so every result printed below can be compared against the numbers worked
 *        out by hand there. The expected values are shown next to the computed ones.
 */
#include <stdio.h>

#include "ml/conv_layer.h"
#include "ml/matrix.h"

/** Convolutional layer input/output) size. */
#define CONV_SIZE 4U

/** Convolutional layer kernel size. */
#define KERNEL_SIZE 2U

/** Learning rate used for the optimization step. */
#define LEARNING_RATE 0.001

/**
 * @brief Print a computed matrix next to the expected one from appendix B.
 *
 * @param[in] label Text describing what's being printed.
 * @param[in] computed Matrix holding the computed values.
 * @param[in] expected Matrix holding the expected values.
 * @param[in] size The size of the matrices (assumed square).
 */
static void print_comparison(const char* label, const matrix_t* computed, const matrix_t* expected,
                             const size_t size)
{
    const double* computed_data = matrix_data_const(computed);
    const double* expected_data = matrix_data_const(expected);

    printf("%s:\n", label);
    printf("\t%-*s  expected (appendix B):\n", (int)(size * 8U), "computed:");

    // Print the two matrices side by side, row by row.
    for (size_t i = 0U; i < size; ++i)
    {
        printf("\t");
        for (size_t j = 0U; j < size; ++j)
        {
            printf("%7.2f ", computed_data[i * size + j]);
        }
        printf("  ");
        for (size_t j = 0U; j < size; ++j)
        {
            printf("%7.2f ", expected_data[i * size + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(void)
{
    // Declared up front and initialized to null, so the cleanup at the end of the function is
    // safe to jump to from any of the checks below, however early one of them fails.
    conv_layer_t* conv_layer = NULL;
    int ret                  = 0;

    // Input image from appendix B, resembling the digit 0 made up of ones.
    const double input_data[CONV_SIZE][CONV_SIZE] = {
        {1, 1, 1, 1},
        {1, 0, 0, 1},
        {1, 0, 0, 1},
        {1, 1, 1, 1},
    };
    matrix_t* input = matrix_copy(input_data, sizeof(input_data));

    // Kernel and bias from appendix B, replacing the randomized ones.
    const double kernel_data[KERNEL_SIZE][KERNEL_SIZE] = {
        {0.2, 0.4},
        {0.6, 0.8},
    };
    matrix_t* kernel        = matrix_copy(kernel_data, sizeof(kernel_data));
    const double bias_value = 0.5;

    // Gradients the max pooling layer sends back in appendix B.
    const double gradients[CONV_SIZE][CONV_SIZE] = {
        {0, 10, 20, 0},
        {0, 0, 0, 0},
        {0, 0, 0, 0},
        {0, 30, 0, 40},
    };
    matrix_t* output_gradients = matrix_copy(gradients, sizeof(gradients));

    // The results appendix B works out by hand, used to check the computed ones.
    const double expected_output[CONV_SIZE][CONV_SIZE] = {
        {1.3, 1.9, 1.9, 1.9},
        {1.7, 1.7, 1.1, 1.9},
        {1.7, 1.3, 0.5, 1.7},
        {1.7, 2.1, 1.9, 2.3},
    };
    const double expected_input_gradients[CONV_SIZE][CONV_SIZE] = {
        {6, 20, 16, 0},
        {0, 0, 0, 0},
        {6, 12, 8, 16},
        {18, 24, 24, 32},
    };
    const double expected_kernel[KERNEL_SIZE][KERNEL_SIZE] = {
        {0.23, 0.44},
        {0.70, 0.90},
    };
    const double expected_bias = 0.6;

    matrix_t* expected_output_matrix = matrix_copy(expected_output, sizeof(expected_output));
    matrix_t* expected_input_gradients_matrix =
        matrix_copy(expected_input_gradients, sizeof(expected_input_gradients));
    matrix_t* expected_kernel_matrix = matrix_copy(expected_kernel, sizeof(expected_kernel));

    // Every matrix above is heap allocated, so check that they all came back before using them.
    if ((NULL == input) || (NULL == kernel) || (NULL == output_gradients) ||
        (NULL == expected_output_matrix) || (NULL == expected_input_gradients_matrix) ||
        (NULL == expected_kernel_matrix))
    {
        printf("Failed to allocate the demo matrices, aborting program!\n");
        ret = -1;
        goto end;
    }

    // Create a convolutional layer: 4x4 input, 2x2 kernel, return -1 on failure.
    conv_layer = conv_layer_new(CONV_SIZE, KERNEL_SIZE);

    // Terminate the program with error code -1 if the layer couldn't be created.
    if (NULL == conv_layer)
    {
        printf("Failed to create convolutional layer, aborting program!\n");
        ret = -1;
        goto end;
    }

    // Use appendix B's kernel and bias instead of the randomized ones, so the results below are
    // the same on every run and can be compared against the hand-worked numbers.
    if (!conv_layer_set_parameters(conv_layer, kernel, bias_value))
    {
        printf("Failed to set the kernel and bias, aborting program!\n");
        ret = -1;
        goto end;
    }

    // Show the input matrix.
    printf("Convolution input data (2D):\n");
    matrix_print(input, true);

    // Perform feedforward (convolution).
    if (!conv_layer_feedforward(conv_layer, input))
    {
        printf("Feedforward failed, aborting program!\n");
        ret = -1;
        goto end;
    }
    print_comparison("Convolution output (2D)", conv_layer_output(conv_layer),
                     expected_output_matrix, CONV_SIZE);

    // Show the output gradients.
    printf("Convolution output gradients (2D):\n");
    matrix_print(output_gradients, true);

    // Perform backpropagation.
    if (!conv_layer_backpropagate(conv_layer, output_gradients))
    {
        printf("Backpropagation failed, aborting program!\n");
        ret = -1;
        goto end;
    }
    print_comparison("Input gradients after backpropagation (2D)",
                     conv_layer_input_gradients(conv_layer), expected_input_gradients_matrix,
                     CONV_SIZE);

    // Perform optimization, which adjusts the kernel and bias using the computed gradients.
    if (!conv_layer_optimize(conv_layer, LEARNING_RATE))
    {
        printf("Optimization failed, aborting program!\n");
        ret = -1;
        goto end;
    }
    print_comparison("Kernel after optimization (2D)", conv_layer_kernel(conv_layer),
                     expected_kernel_matrix, KERNEL_SIZE);
    printf("Bias after optimization:\n");
    printf("\tcomputed: %.2f, expected (appendix B): %.2f\n\n", conv_layer_bias(conv_layer),
           expected_bias);

end:
    matrix_del(&input);
    matrix_del(&kernel);
    matrix_del(&output_gradients);
    matrix_del(&expected_output_matrix);
    matrix_del(&expected_input_gradients_matrix);
    matrix_del(&expected_kernel_matrix);
    conv_layer_del(&conv_layer);
    return ret;
}
