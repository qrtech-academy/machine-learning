/**
 * @file Simple convolutional layer demo.
 */
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <vector>

namespace
{
/** One-dimensional matrix. */
using Matrix1d = std::vector<double>;

/** Two-dimensional matrix. */
using Matrix2d = std::vector<Matrix1d>;

/**
 * @brief Initialize given matrix with zeros.
 *
 * @param[in] matrix The matrix to initialize.
 */
void initMatrix(Matrix2d& matrix) noexcept
{
    // Fill the matrix with zeros.
    for (auto& row : matrix)
    {
        for (auto& num : row)
        {
            num = 0.0;
        }
    }
}

/**
 * @brief Initialize matrix with zeros.
 *
 * @param[in] matrix The matrix to initialize.
 * @param[in] size The desired size of the matrix.
 */
void initMatrix(Matrix2d& matrix, const std::size_t size)
{
    // Resize the matrix if necessary, then fill the matrix with zeros.
    matrix.resize(size, Matrix1d(size));
    initMatrix(matrix);
}

/**
 * @brief Check whether given matrix is square.
 *
 * @param[in] matrix The matrix to check.
 *
 * @return True if the matrix is square, false otherwise.
 */
[[nodiscard]] bool isMatrixSquare(const Matrix2d& matrix) noexcept
{
    // Check the matrix row by row.
    for (const auto& row : matrix)
    {
        // Return false if the matrix isn't square.
        if (row.size() != matrix.size()) { return false; }
    }
    // Return true to indicate that the matrix is square.
    return true;
}

/**
 * @brief Print the contents of given matrix.
 *
 * @param[in] matrix The matrix to print.
 */
void printMatrix(const Matrix2d& matrix) noexcept
{
    // Print the matrix contents row by row.
    for (const auto& row : matrix)
    {
        const auto* lastNum{row.empty() ? nullptr : &row[row.size() - 1U]};
        std::printf("\t");

        // Separate each number in the row with a comma.
        for (const auto& num : row)
        {
            std::printf("%.1f", num);
            if (&num < lastNum) { std::printf(", "); }
        }
        std::printf("\n");
    }
    std::printf("\n");
}

/**
 * @brief Print a computed matrix next to the expected one from appendix B.
 *
 * @param[in] label Text describing what's being printed.
 * @param[in] computed The matrix holding the computed values.
 * @param[in] expected The matrix holding the expected values.
 */
void printComparison(const char* label, const Matrix2d& computed, const Matrix2d& expected) noexcept
{
    std::printf("%s:\n", label);
    std::printf("\t%-*s  expected (appendix B):\n", static_cast<int>(computed.size() * 8U),
                "computed:");

    // Print the two matrices side by side, row by row.
    for (std::size_t i{}; i < computed.size(); ++i)
    {
        std::printf("\t");
        for (const auto& num : computed[i])
        {
            std::printf("%7.2f ", num);
        }
        std::printf("  ");
        for (const auto& num : expected[i])
        {
            std::printf("%7.2f ", num);
        }
        std::printf("\n");
    }
    std::printf("\n");
}

/**
 * @brief Generate a random starting value between 0.0 and 1.0.
 *
 * @return Random floating-point value in the range [0.0, 1.0].
 */
[[nodiscard]] double randomStartVal() noexcept
{
    constexpr auto max = static_cast<double>(RAND_MAX);
    return std::rand() / max;
}

/**
 * @brief ReLU activation function (output).
 *
 * @param[in] input Input value.
 *
 * @return Output after ReLU activation.
 */
[[nodiscard]] constexpr double reluOutput(const double input) noexcept
{
    return 0.0 < input ? input : 0.0;
}

/**
 * @brief ReLU activation function (derivative).
 *
 * @param[in] input Input value.
 *
 * @return Derivative of ReLU at input.
 */
[[nodiscard]] constexpr double reluDelta(const double input) noexcept
{
    return 0.0 < input ? 1.0 : 0.0;
}

namespace ml
{
/**
 * @brief Convolutional layer structure.
 *
 * @todo Implement this struct!
 */
struct ConvLayer final
{
    /**
     * @brief Constructor.
     *
     * @param[in] inputSize Input size. Must be greater than 0.
     * @param[in] kernelSize Kernel size. Must be greater than 0 and smaller than the input size.
     */

    /**
     * @todo Check the input arguments in the constructor:
     *
     *       If the input size is 0, the kernel size is 0, or the input size is smaller than
     *       the kernel size, terminate the program with error message
     *       "Cannot create convolutional layer: invalid input arguments!".
     */

    /**
     * @brief Perform feedforward operation.
     *
     * @param[in] input Matrix holding input data.
     *
     * @return True on success, false on failure.
     */

    /**
     * @brief Perform backpropagation.
     *
     * @param[in] outputGradients Matrix holding gradients from the next layer.
     *
     * @return True on success, false on failure.
     */

    /**
     * @brief Perform optimization.
     *
     * @param[in] learningRate Learning rate to use. Must be in range (0.0, 1.0].
     *
     * @return True on success, false on failure.
     */

    /** @todo Define member variables here! Keep them public and skip the 'my' prefix.
     *
     * We need:
     *     - Padded input.
     *     - Input gradients, both padded and unpadded.
     *     - Output and preactivation output.
     *     - Kernel and kernel gradients.
     *     - Bias value and bias gradient.
     */

private:
    /**
     * @brief Pad input with zeros.
     *
     * @param[in] input Input data.
     */

    /**
     * @brief Extract input gradients.
     */
};
} // namespace ml
} // namespace

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

    // Input image from appendix B, resembling the digit 0 made up of ones.
    const Matrix2d input{{1, 1, 1, 1}, {1, 0, 0, 1}, {1, 0, 0, 1}, {1, 1, 1, 1}};

    // Gradients the max pooling layer sends back in appendix B.
    const Matrix2d outputGradients{{0, 10, 20, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 30, 0, 40}};

    // The results appendix B works out by hand, used to check the computed ones.
    const Matrix2d expectedOutput{
        {1.3, 1.9, 1.9, 1.9}, {1.7, 1.7, 1.1, 1.9}, {1.7, 1.3, 0.5, 1.7}, {1.7, 2.1, 1.9, 2.3}};
    const Matrix2d expectedInputGradients{
        {6, 20, 16, 0}, {0, 0, 0, 0}, {6, 12, 8, 16}, {18, 24, 24, 32}};
    const Matrix2d expectedKernel{{0.23, 0.44}, {0.70, 0.90}};
    constexpr double expectedBias{0.6};
    constexpr double learningRate{0.001};

    // Create a convolutional layer: 4x4 input, 2x2 kernel.
    constexpr std::size_t inputSize{4U};
    constexpr std::size_t kernelSize{2U};
    ml::ConvLayer convLayer{inputSize, kernelSize};

    // Use appendix B's kernel and bias instead of the randomized ones, so the results below are
    // the same on every run and can be compared against the hand-worked numbers.
    convLayer.kernel = Matrix2d{{0.2, 0.4}, {0.6, 0.8}};
    convLayer.bias   = 0.5;

    // Show the input matrix.
    std::printf("Convolution input data (2D):\n");
    printMatrix(input);

    // Perform feedforward (convolution).
    if (!convLayer.feedforward(input))
    {
        std::printf("Feedforward failed, aborting program!\n");
        return -1;
    }
    printComparison("Convolution output (2D)", convLayer.output, expectedOutput);

    // Show the output gradients.
    std::printf("Convolution output gradients (2D):\n");
    printMatrix(outputGradients);

    // Perform backpropagation.
    if (!convLayer.backpropagate(outputGradients))
    {
        std::printf("Backpropagation failed, aborting program!\n");
        return -1;
    }
    printComparison("Input gradients after backpropagation (2D)", convLayer.inputGradients,
                    expectedInputGradients);

    // Perform optimization, which adjusts the kernel and bias using the computed gradients.
    if (!convLayer.optimize(learningRate))
    {
        std::printf("Optimization failed, aborting program!\n");
        return -1;
    }
    printComparison("Kernel after optimization (2D)", convLayer.kernel, expectedKernel);
    std::printf("Bias after optimization:\n");
    std::printf("\tcomputed: %.2f, expected (appendix B): %.2f\n\n", convLayer.bias, expectedBias);
    return 0;

//! @todo Remove this header guard (and/or uncomment the compiler flags in the makefile) once the
//        implementation is finished.
#endif /** CONV_LAYER_IMPLEMENTED */
}
