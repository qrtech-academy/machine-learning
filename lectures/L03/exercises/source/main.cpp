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

    const ml::Matrix1d input1{1.0, 1.0};
    const ml::Matrix1d input2{1.0};

    // Create dense layers and print layer info.
    ml::dense_layer::Stub hiddenLayer{hiddenCount, inputCount};
    ml::dense_layer::Stub outputLayer{outputCount, hiddenCount};
    printLayerInfo(hiddenLayer);
    printLayerInfo(outputLayer);

    // Test the feedforward operations, return 0 on success, -1 on failure.
    const auto result = testFeedforward(hiddenLayer) && testFeedforward(outputLayer);
    if (result) { std::printf("Feedforward operations work as expected!\n"); }
    return result ? 0 : -1;
}
