/**
 * @file Unit tests for ml::conv_layer::Conv.
 */
#include <cstddef>

#include "ml/act_func/type.hpp"
#include "ml/conv_layer/conv.hpp"
#include "ml/conv_layer/interface.hpp"
#include "ml/types.hpp"
#include "ml/utils.hpp"
#include "qacademy/test/test.hpp"

using ActFunc   = ml::act_func::Type;
using ConvLayer = ml::conv_layer::Conv;
using Matrix1d  = ml::Matrix1d;
using Matrix2d  = ml::Matrix2d;

namespace
{
/**
 * @brief Test parameters.
 */
struct Test
{
    /** Value tolerance for hand-computed results when using EXPECT_NEAR(). */
    static constexpr double ExactTolerance{1e-9};

    /** Input size used by most tests. */
    static constexpr std::size_t InputSize{4U};

    /** Kernel size used by most tests. Deliberately unequal to InputSize. */
    static constexpr std::size_t KernelSize{2U};

    /** A learning rate inside the valid range. */
    static constexpr double LearningRate{0.1};

    /** Learning rates outside the valid range (0.0, 1.0), which is open at both ends. */
    static constexpr double ZeroLearningRate{0.0};
    static constexpr double NegativeLearningRate{-0.1};
    static constexpr double UpperBoundLearningRate{1.0};
    static constexpr double AboveUpperLearningRate{1.5};

    /** Value filling a matrix that carries no signal. */
    static constexpr double ZeroValue{0.0};

    /** Value filling a matrix whose contents don't matter, only its size. */
    static constexpr double FillerValue{1.0};

    /** How far a deliberately wrong-sized matrix differs from the size the layer expects. */
    static constexpr std::size_t SizeDelta{1U};

    /** Index of the first row/column of a matrix. */
    static constexpr std::size_t FirstIndex{0U};
};

// -----------------------------------------------------------------------------
[[nodiscard]] Matrix2d testInput() noexcept
{
    // The digit 0 made up of ones, the same input used throughout L06 - L10.
    return Matrix2d{
        {1.0, 1.0, 1.0, 1.0}, {1.0, 0.0, 0.0, 1.0}, {1.0, 0.0, 0.0, 1.0}, {1.0, 1.0, 1.0, 1.0}};
}

// -----------------------------------------------------------------------------
[[nodiscard]] Matrix2d testGradients() noexcept
{
    // Deliberately asymmetric, so a transposed index shows up as a failure.
    return Matrix2d{
        {0.0, 10.0, 20.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 30.0, 0.0, 40.0}};
}

// -----------------------------------------------------------------------------
[[nodiscard]] constexpr std::size_t paddingOffset(const std::size_t kernelSize) noexcept
{
    // Pad by half the kernel size per side, which keeps the output the same size as the input.
    return kernelSize / 2U;
}

// -----------------------------------------------------------------------------
[[nodiscard]] constexpr std::size_t paddedSize(const std::size_t inputSize,
                                               const std::size_t kernelSize) noexcept
{
    return inputSize + (2U * paddingOffset(kernelSize));
}

// -----------------------------------------------------------------------------
[[nodiscard]] Matrix2d padded(const Matrix2d& input, const std::size_t kernelSize) noexcept
{
    const std::size_t offset{paddingOffset(kernelSize)};
    Matrix2d result{};
    ml::initMatrix(result, paddedSize(input.size(), kernelSize));

    for (std::size_t i{}; i < input.size(); ++i)
    {
        for (std::size_t j{}; j < input.size(); ++j)
        {
            result[i + offset][j + offset] = input[i][j];
        }
    }
    return result;
}

// -----------------------------------------------------------------------------
[[nodiscard]] double windowSum(const Matrix2d& inputPadded, const Matrix2d& kernel,
                               const std::size_t row, const std::size_t col) noexcept
{
    double sum{};

    for (std::size_t ki{}; ki < kernel.size(); ++ki)
    {
        for (std::size_t kj{}; kj < kernel.size(); ++kj)
        {
            sum += inputPadded[row + ki][col + kj] * kernel[ki][kj];
        }
    }
    return sum;
}

// -----------------------------------------------------------------------------
[[nodiscard]] Matrix2d expectedOutput(const ConvLayer& convLayer, const Matrix2d& input) noexcept
{
    const auto inputPadded = padded(input, convLayer.kernel().size());
    Matrix2d result{};
    ml::initMatrix(result, input.size());

    for (std::size_t i{}; i < input.size(); ++i)
    {
        for (std::size_t j{}; j < input.size(); ++j)
        {
            result[i][j] = convLayer.bias() + windowSum(inputPadded, convLayer.kernel(), i, j);
        }
    }
    return result;
}

/**
 * @brief Verify that the layer reports its constructed sizes, and that output size equals input.
 */
TEST(ConvLayerConv, ConstructedDimensions)
{
    const ConvLayer convLayer{Test::InputSize, Test::KernelSize};
    EXPECT_EQ(convLayer.inputSize(), Test::InputSize);
    EXPECT_EQ(convLayer.outputSize(), Test::InputSize);
}

/**
 * @brief Verify that the output and input gradient matrices are built square and input-sized.
 */
TEST(ConvLayerConv, ConstructedMatrixShapes)
{
    const ConvLayer convLayer{Test::InputSize, Test::KernelSize};
    EXPECT_EQ(convLayer.output().size(), Test::InputSize);
    EXPECT_EQ(convLayer.inputGradients().size(), Test::InputSize);

    for (std::size_t i{}; i < Test::InputSize; ++i)
    {
        EXPECT_EQ(convLayer.output()[i].size(), Test::InputSize);
        EXPECT_EQ(convLayer.inputGradients()[i].size(), Test::InputSize);
    }
}

/**
 * @brief Verify that the kernel is built square and sized as the constructor was told.
 */
TEST(ConvLayerConv, ConstructedKernelShape)
{
    const ConvLayer convLayer{Test::InputSize, Test::KernelSize};
    EXPECT_EQ(convLayer.kernel().size(), Test::KernelSize);

    for (const auto& row : convLayer.kernel())
    {
        EXPECT_EQ(row.size(), Test::KernelSize);
    }
}

/**
 * @brief Verify that the bias and kernel weights start randomized inside the range [0.0, 1.0].
 */
TEST(ConvLayerConv, ConstructedParametersAreRandomized)
{
    constexpr double randMin{0.0};
    constexpr double randMax{1.0};

    bool kernelVaries{false};
    const ConvLayer convLayer{Test::InputSize, Test::KernelSize};
    EXPECT_TRUE(randMin <= convLayer.bias());
    EXPECT_TRUE(randMax >= convLayer.bias());

    for (const auto& row : convLayer.kernel())
    {
        for (const auto value : row)
        {
            EXPECT_TRUE(randMin <= value);
            EXPECT_TRUE(randMax >= value);

            // Expect at least one weight to differ from the first one.
            if (value != convLayer.kernel()[Test::FirstIndex][Test::FirstIndex])
            {
                kernelVaries = true;
            }
        }
    }
    EXPECT_TRUE(kernelVaries);
}

/**
 * @brief Verify that a larger, odd-sized kernel still yields an output the size of the input.
 */
TEST(ConvLayerConv, LargerKernelKeepsOutputSize)
{
    constexpr std::size_t inputSize{6U};
    constexpr std::size_t kernelSize{3U};

    const ConvLayer convLayer{inputSize, kernelSize};
    EXPECT_EQ(convLayer.inputSize(), inputSize);
    EXPECT_EQ(convLayer.outputSize(), inputSize);
    EXPECT_EQ(convLayer.kernel().size(), kernelSize);
}

/**
 * @brief Verify that feedforward computes the bias plus the kernel's dot product with each window.
 */
TEST(ConvLayerConv, FeedforwardMatchesHandComputedConvolution)
{
    ConvLayer convLayer{Test::InputSize, Test::KernelSize, ActFunc::None};
    const auto input    = testInput();
    const auto expected = expectedOutput(convLayer, input);

    EXPECT_TRUE(convLayer.feedforward(input));

    for (std::size_t i{}; i < Test::InputSize; ++i)
    {
        for (std::size_t j{}; j < Test::InputSize; ++j)
        {
            EXPECT_NEAR(convLayer.output()[i][j], expected[i][j], Test::ExactTolerance);
        }
    }
}

/**
 * @brief Verify that an all-zero input yields the bias in every output position.
 */
TEST(ConvLayerConv, ZeroInputYieldsBiasEverywhere)
{
    ConvLayer convLayer{Test::InputSize, Test::KernelSize, ActFunc::None};
    const Matrix2d zeroInput(Test::InputSize, Matrix1d(Test::InputSize, Test::ZeroValue));

    EXPECT_TRUE(convLayer.feedforward(zeroInput));

    for (const auto& row : convLayer.output())
    {
        for (const auto value : row)
        {
            EXPECT_NEAR(value, convLayer.bias(), Test::ExactTolerance);
        }
    }
}

/**
 * @brief Verify that a 1x1 kernel adds no padding and scales each pixel by the single weight.
 */
TEST(ConvLayerConv, UnitKernelIsAffinePerPixel)
{
    constexpr std::size_t kernelSize{1U};
    ConvLayer convLayer{Test::InputSize, kernelSize, ActFunc::None};
    const auto input = testInput();

    EXPECT_TRUE(convLayer.feedforward(input));

    // A 1x1 kernel holds exactly one weight.
    const auto onlyWeight = convLayer.kernel()[Test::FirstIndex][Test::FirstIndex];

    for (std::size_t i{}; i < Test::InputSize; ++i)
    {
        for (std::size_t j{}; j < Test::InputSize; ++j)
        {
            const auto expected = convLayer.bias() + (input[i][j] * onlyWeight);
            EXPECT_NEAR(convLayer.output()[i][j], expected, Test::ExactTolerance);
        }
    }
}

/**
 * @brief Verify that the zero-padding only affects the border.
 */
TEST(ConvLayerConv, PaddingOnlyAffectsTheBorder)
{
    constexpr double constantValue{1.0};

    // Windows lying wholly inside the image run from offset to offset + inputSize - kernelSize.
    constexpr std::size_t offset{paddingOffset(Test::KernelSize)};
    constexpr std::size_t firstInterior{offset};
    constexpr std::size_t lastInterior{offset + Test::InputSize - Test::KernelSize};

    ConvLayer convLayer{Test::InputSize, Test::KernelSize, ActFunc::None};
    const Matrix2d input(Test::InputSize, Matrix1d(Test::InputSize, constantValue));
    EXPECT_TRUE(convLayer.feedforward(input));

    // Expect every interior window to produce the same value.
    const auto interior = convLayer.output()[firstInterior][firstInterior];

    for (std::size_t i{firstInterior}; i <= lastInterior; ++i)
    {
        for (std::size_t j{firstInterior}; j <= lastInterior; ++j)
        {
            EXPECT_NEAR(convLayer.output()[i][j], interior, Test::ExactTolerance);
        }
    }

    // Expect the corner to overlap the image only at the kernel's (offset, offset).
    const auto corner = convLayer.bias() + (constantValue * convLayer.kernel()[offset][offset]);
    EXPECT_NEAR(convLayer.output()[Test::FirstIndex][Test::FirstIndex], corner,
                Test::ExactTolerance);
}

/**
 * @brief Verify that ReLU is applied to the weighted sum, clamping negative sums to zero.
 */
TEST(ConvLayerConv, AppliesReluActivation)
{
    constexpr double negativeValue{-100.0};
    ConvLayer convLayer{Test::InputSize, Test::KernelSize, ActFunc::Relu};
    const Matrix2d input(Test::InputSize, Matrix1d(Test::InputSize, negativeValue));
    const auto expected = expectedOutput(convLayer, input);

    EXPECT_TRUE(convLayer.feedforward(input));

    for (std::size_t i{}; i < Test::InputSize; ++i)
    {
        for (std::size_t j{}; j < Test::InputSize; ++j)
        {
            // Expect the sum when positive, otherwise zero.
            const auto relu = Test::ZeroValue < expected[i][j] ? expected[i][j] : Test::ZeroValue;
            EXPECT_NEAR(convLayer.output()[i][j], relu, Test::ExactTolerance);
            EXPECT_TRUE(Test::ZeroValue <= convLayer.output()[i][j]);
        }
    }
}

/**
 * @brief Verify that feedforward rejects an input whose size doesn't match the layer.
 */
TEST(ConvLayerConv, FeedforwardRejectsWrongSizedInput)
{
    constexpr std::size_t smallerSize{Test::InputSize - Test::SizeDelta};
    constexpr std::size_t largerSize{Test::InputSize + Test::SizeDelta};

    ConvLayer convLayer{Test::InputSize, Test::KernelSize};
    const Matrix2d tooSmall(smallerSize, Matrix1d(smallerSize, Test::FillerValue));
    const Matrix2d tooLarge(largerSize, Matrix1d(largerSize, Test::FillerValue));

    EXPECT_FALSE(convLayer.feedforward(tooSmall));
    EXPECT_FALSE(convLayer.feedforward(tooLarge));
}

/**
 * @brief Verify that feedforward rejects a ragged input, i.e. one with a row of the wrong length.
 */
TEST(ConvLayerConv, FeedforwardRejectsRaggedInput)
{
    // Any row will do.
    constexpr std::size_t shortenedRow{2U};

    ConvLayer convLayer{Test::InputSize, Test::KernelSize};
    auto ragged = testInput();
    ragged[shortenedRow].pop_back();

    EXPECT_FALSE(convLayer.feedforward(ragged));
}

/**
 * @brief Verify that backpropagation rejects gradients whose size doesn't match the output.
 */
TEST(ConvLayerConv, BackpropagateRejectsWrongSizedGradients)
{
    constexpr std::size_t smallerSize{Test::InputSize - Test::SizeDelta};

    ConvLayer convLayer{Test::InputSize, Test::KernelSize};
    EXPECT_TRUE(convLayer.feedforward(testInput()));

    const Matrix2d tooSmall(smallerSize, Matrix1d(smallerSize, Test::FillerValue));
    EXPECT_FALSE(convLayer.backpropagate(tooSmall));
}

/**
 * @brief Verify that backpropagation rejects ragged gradients.
 */
TEST(ConvLayerConv, BackpropagateRejectsRaggedGradients)
{
    // Any row will do.
    constexpr std::size_t shortenedRow{1U};

    ConvLayer convLayer{Test::InputSize, Test::KernelSize};
    EXPECT_TRUE(convLayer.feedforward(testInput()));

    auto ragged = testGradients();
    ragged[shortenedRow].pop_back();

    EXPECT_FALSE(convLayer.backpropagate(ragged));
}

/**
 * @brief Verify that zero gradients from the next layer propagate no error backwards.
 */
TEST(ConvLayerConv, ZeroGradientsYieldZeroInputGradients)
{
    ConvLayer convLayer{Test::InputSize, Test::KernelSize, ActFunc::None};
    const Matrix2d zeroGradients(Test::InputSize, Matrix1d(Test::InputSize, Test::ZeroValue));

    EXPECT_TRUE(convLayer.feedforward(testInput()));
    EXPECT_TRUE(convLayer.backpropagate(zeroGradients));

    for (const auto& row : convLayer.inputGradients())
    {
        for (const auto value : row)
        {
            EXPECT_NEAR(value, Test::ZeroValue, Test::ExactTolerance);
        }
    }
}

/**
 * @brief Verify that the input gradients are the kernel scattered back, then unpadded.
 */
TEST(ConvLayerConv, InputGradientsMatchHandComputedScatter)
{
    constexpr std::size_t offset{paddingOffset(Test::KernelSize)};

    ConvLayer convLayer{Test::InputSize, Test::KernelSize, ActFunc::None};
    const auto gradients = testGradients();

    EXPECT_TRUE(convLayer.feedforward(testInput()));
    EXPECT_TRUE(convLayer.backpropagate(gradients));

    // Scatter the kernel back into a padded matrix, exactly as backpropagate() should have.
    Matrix2d expectedPadded{};
    ml::initMatrix(expectedPadded, paddedSize(Test::InputSize, Test::KernelSize));

    for (std::size_t i{}; i < Test::InputSize; ++i)
    {
        for (std::size_t j{}; j < Test::InputSize; ++j)
        {
            for (std::size_t ki{}; ki < Test::KernelSize; ++ki)
            {
                for (std::size_t kj{}; kj < Test::KernelSize; ++kj)
                {
                    expectedPadded[i + ki][j + kj] += convLayer.kernel()[ki][kj] * gradients[i][j];
                }
            }
        }
    }

    // Expect the layer's unpadded gradients to match the interior of the scattered matrix.
    for (std::size_t i{}; i < Test::InputSize; ++i)
    {
        for (std::size_t j{}; j < Test::InputSize; ++j)
        {
            EXPECT_NEAR(convLayer.inputGradients()[i][j], expectedPadded[i + offset][j + offset],
                        Test::ExactTolerance);
        }
    }
}

/**
 * @brief Verify that optimization rejects a learning rate outside the range (0.0, 1.0).
 */
TEST(ConvLayerConv, OptimizeRejectsInvalidLearningRate)
{
    ConvLayer convLayer{Test::InputSize, Test::KernelSize};
    EXPECT_TRUE(convLayer.feedforward(testInput()));
    EXPECT_TRUE(convLayer.backpropagate(testGradients()));

    EXPECT_FALSE(convLayer.optimize(Test::ZeroLearningRate));
    EXPECT_FALSE(convLayer.optimize(Test::NegativeLearningRate));
    EXPECT_FALSE(convLayer.optimize(Test::UpperBoundLearningRate));
    EXPECT_FALSE(convLayer.optimize(Test::AboveUpperLearningRate));
}

/**
 * @brief Verify that optimization accepts a learning rate inside the range (0.0, 1.0).
 */
TEST(ConvLayerConv, OptimizeAcceptsValidLearningRate)
{
    ConvLayer convLayer{Test::InputSize, Test::KernelSize};
    EXPECT_TRUE(convLayer.feedforward(testInput()));
    EXPECT_TRUE(convLayer.backpropagate(testGradients()));

    EXPECT_TRUE(convLayer.optimize(Test::LearningRate));
}

/**
 * @brief Verify that optimization moves the bias by the bias gradient scaled by the learning rate.
 */
TEST(ConvLayerConv, OptimizeMovesBiasByGradientTimesLearningRate)
{
    ConvLayer convLayer{Test::InputSize, Test::KernelSize, ActFunc::None};
    const auto gradients  = testGradients();
    const auto biasBefore = convLayer.bias();

    double gradientSum{};

    for (const auto& row : gradients)
    {
        for (const auto value : row)
        {
            gradientSum += value;
        }
    }

    EXPECT_TRUE(convLayer.feedforward(testInput()));
    EXPECT_TRUE(convLayer.backpropagate(gradients));
    EXPECT_TRUE(convLayer.optimize(Test::LearningRate));

    const auto expected = biasBefore + (gradientSum * Test::LearningRate);
    EXPECT_NEAR(convLayer.bias(), expected, Test::ExactTolerance);
}

/**
 * @brief Verify that optimization moves every kernel weight by its gradient times the rate.
 */
TEST(ConvLayerConv, OptimizeMovesKernelByGradientTimesLearningRate)
{
    ConvLayer convLayer{Test::InputSize, Test::KernelSize, ActFunc::None};
    const auto input        = testInput();
    const auto gradients    = testGradients();
    const auto kernelBefore = convLayer.kernel();
    const auto inputPadded  = padded(input, Test::KernelSize);

    EXPECT_TRUE(convLayer.feedforward(input));
    EXPECT_TRUE(convLayer.backpropagate(gradients));
    EXPECT_TRUE(convLayer.optimize(Test::LearningRate));

    for (std::size_t ki{}; ki < Test::KernelSize; ++ki)
    {
        for (std::size_t kj{}; kj < Test::KernelSize; ++kj)
        {
            double kernelGradient{};

            for (std::size_t i{}; i < Test::InputSize; ++i)
            {
                for (std::size_t j{}; j < Test::InputSize; ++j)
                {
                    kernelGradient += inputPadded[i + ki][j + kj] * gradients[i][j];
                }
            }
            const auto expected = kernelBefore[ki][kj] + (kernelGradient * Test::LearningRate);
            EXPECT_NEAR(convLayer.kernel()[ki][kj], expected, Test::ExactTolerance);
        }
    }
}

/**
 * @brief Verify that optimizing with zero gradients leaves the bias and kernel unchanged.
 */
TEST(ConvLayerConv, OptimizeWithZeroGradientsLeavesParametersUnchanged)
{
    ConvLayer convLayer{Test::InputSize, Test::KernelSize, ActFunc::None};
    const auto biasBefore   = convLayer.bias();
    const auto kernelBefore = convLayer.kernel();
    const Matrix2d zeroGradients(Test::InputSize, Matrix1d(Test::InputSize, Test::ZeroValue));

    EXPECT_TRUE(convLayer.feedforward(testInput()));
    EXPECT_TRUE(convLayer.backpropagate(zeroGradients));
    EXPECT_TRUE(convLayer.optimize(Test::LearningRate));

    EXPECT_NEAR(convLayer.bias(), biasBefore, Test::ExactTolerance);

    for (std::size_t ki{}; ki < Test::KernelSize; ++ki)
    {
        for (std::size_t kj{}; kj < Test::KernelSize; ++kj)
        {
            EXPECT_NEAR(convLayer.kernel()[ki][kj], kernelBefore[ki][kj], Test::ExactTolerance);
        }
    }
}

/**
 * @brief Verify that the layer can be used through conv_layer::Interface.
 */
TEST(ConvLayerConv, UsableThroughInterface)
{
    ConvLayer convLayer{Test::InputSize, Test::KernelSize, ActFunc::None};
    ml::conv_layer::Interface& layer{convLayer};

    EXPECT_EQ(layer.inputSize(), Test::InputSize);
    EXPECT_EQ(layer.outputSize(), Test::InputSize);
    EXPECT_TRUE(layer.feedforward(testInput()));
    EXPECT_TRUE(layer.backpropagate(testGradients()));
    EXPECT_TRUE(layer.optimize(Test::LearningRate));
}
} // namespace
