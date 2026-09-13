/**
 * @file Dense layer demo.
 */
#include <cstddef>
#include <cstdio>

#include "ml/dense_layer/stub.hpp"
#include "ml/types.hpp"

namespace
{
// -----------------------------------------------------------------------------
void printMatrix(const ml::Matrix1d& matrix) noexcept
{
    if (matrix.empty()) { return; }
    const std::size_t last{matrix.size() - 1U};

    for (std::size_t i{}; i < last; ++i)
    {
        std::printf("%g, ", matrix[i]);
    }
    std::printf("%g\n", matrix[last]);
}

// -----------------------------------------------------------------------------
void printLayerInfo(const ml::dense_layer::Interface& denseLayer) noexcept
{
    std::printf("Node count:\t%zu\n", denseLayer.nodeCount());
    std::printf("Weight count:\t%zu\n", denseLayer.weightCount());
    std::printf("Output:\t\t");
    printMatrix(denseLayer.output());
    std::printf("\n");
}

// -----------------------------------------------------------------------------
[[nodiscard]] bool testOutputValue(const ml::dense_layer::Interface& denseLayer,
                                   const double expectedValue) noexcept
{
    bool result{true};

    // The stub stores the value it was given rather than computing it, so compare exactly.
    for (const auto value : denseLayer.output())
    {
        if (expectedValue != value)
        {
            std::fprintf(stderr, "Output value %g doesn't match the expected value %g!\n", value,
                         expectedValue);
            result = false;
        }
    }
    return result;
}

// -----------------------------------------------------------------------------
[[nodiscard]] bool testFeedforward(ml::dense_layer::Interface& denseLayer) noexcept
{
    bool result{true};
    const ml::Matrix1d validInput(denseLayer.weightCount());
    const ml::Matrix1d invalidInput(denseLayer.weightCount() + 1U);

    if (!denseLayer.feedforward(validInput))
    {
        std::fprintf(stderr, "Feedforward failed with valid input!\n");
        result = false;
    }
    if (denseLayer.feedforward(invalidInput))
    {
        std::fprintf(stderr, "Feedforward succeeded with invalid input!\n");
        result = false;
    }
    return result;
}
} // namespace

/**
 * @brief Application entry point.
 *
 * @return 0 on success, -1 on failure.
 */
int main()
{
    constexpr std::size_t inputCount{2U};
    constexpr std::size_t hiddenCount{3U};
    constexpr std::size_t outputCount{1U};
    constexpr double defaultOutputValue{0.5};
    constexpr double explicitOutputValue{0.75};

    // Create dense layers, the hidden layer with the default output value and the output layer
    // with an explicit one, and print layer info.
    ml::dense_layer::Stub hiddenLayer{hiddenCount, inputCount};
    ml::dense_layer::Stub outputLayer{outputCount, hiddenCount, explicitOutputValue};
    printLayerInfo(hiddenLayer);
    printLayerInfo(outputLayer);

    // Test the output values and the feedforward operations.
    const auto outputResult = testOutputValue(hiddenLayer, defaultOutputValue) &&
                              testOutputValue(outputLayer, explicitOutputValue);
    const auto feedforwardResult = testFeedforward(hiddenLayer) && testFeedforward(outputLayer);

    if (outputResult) { std::printf("Output values work as expected!\n"); }
    if (feedforwardResult) { std::printf("Feedforward operations work as expected!\n"); }

    // Return 0 on success, -1 on failure.
    return (outputResult && feedforwardResult) ? 0 : -1;
}
