/**
 * @file Unit tests for ml::flatten_layer::Flatten.
 */
#include <cstddef>

#include "ml/flatten_layer/flatten.hpp"
#include "ml/flatten_layer/interface.hpp"
#include "ml/types.hpp"
#include "qacademy/test/test.hpp"

using FlattenLayer = ml::flatten_layer::Flatten;
using Matrix1d     = ml::Matrix1d;
using Matrix2d     = ml::Matrix2d;

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
    static constexpr std::size_t InputSize{2U};

    /** Output size implied by the input size, i.e. InputSize squared. */
    static constexpr std::size_t OutputSize{InputSize * InputSize};

    /** Input size of a layer holding a single value, the smallest one there is. */
    static constexpr std::size_t SingleInputSize{1U};

    /** An arbitrary value, used where only the fact it survives the round trip matters. */
    static constexpr double SingleValue{7.0};

    /** Value filling a matrix whose contents don't matter, only its size. */
    static constexpr double FillerValue{1.0};

    /** How far a deliberately wrong-sized matrix differs from the size the layer expects. */
    static constexpr std::size_t SizeDelta{1U};

    /** Row shortened by the ragged-matrix test. Any row would do. */
    static constexpr std::size_t ShortenedRow{1U};
};

// -----------------------------------------------------------------------------
// Not noexcept: EXPECT_* reports failures by throwing, which would terminate a noexcept function.
void expectVectorNear(const Matrix1d& actual, const Matrix1d& expected)
{
    EXPECT_EQ(actual.size(), expected.size());

    for (std::size_t i{}; i < expected.size(); ++i)
    {
        EXPECT_NEAR(actual[i], expected[i], Test::ExactTolerance);
    }
}

// -----------------------------------------------------------------------------
// Not noexcept: EXPECT_* reports failures by throwing, which would terminate a noexcept function.
void expectMatrixNear(const Matrix2d& actual, const Matrix2d& expected)
{
    EXPECT_EQ(actual.size(), expected.size());

    for (std::size_t i{}; i < expected.size(); ++i)
    {
        EXPECT_EQ(actual[i].size(), expected[i].size());

        for (std::size_t j{}; j < expected[i].size(); ++j)
        {
            EXPECT_NEAR(actual[i][j], expected[i][j], Test::ExactTolerance);
        }
    }
}

/**
 * @brief Verify that the layer squares its input size: an n x n matrix flattens to n * n values.
 */
TEST(FlattenLayerFlatten, ConstructedDimensions)
{
    const FlattenLayer flattenLayer{Test::InputSize};
    EXPECT_EQ(flattenLayer.inputSize(), Test::InputSize);
    EXPECT_EQ(flattenLayer.outputSize(), Test::OutputSize);
}

/**
 * @brief Verify that the output is a flat vector and the input gradients are a square matrix.
 */
TEST(FlattenLayerFlatten, ConstructedMatrixShapes)
{
    const FlattenLayer flattenLayer{Test::InputSize};
    EXPECT_EQ(flattenLayer.output().size(), Test::OutputSize);
    EXPECT_EQ(flattenLayer.inputGradients().size(), Test::InputSize);

    for (const auto& row : flattenLayer.inputGradients())
    {
        EXPECT_EQ(row.size(), Test::InputSize);
    }
}

/**
 * @brief Verify the size relationship holds for a larger layer too.
 */
TEST(FlattenLayerFlatten, LargerInputSquaresOutputSize)
{
    constexpr std::size_t inputSize{4U};
    const FlattenLayer flattenLayer{inputSize};
    EXPECT_EQ(flattenLayer.outputSize(), inputSize * inputSize);
}

/**
 * @brief Verify feedforward against the hand-training example in L06's appendix B.
 */
TEST(FlattenLayerFlatten, FeedforwardMatchesHandTrainedExample)
{
    FlattenLayer flattenLayer{Test::InputSize};
    EXPECT_TRUE(flattenLayer.feedforward(Matrix2d{{1.9, 1.9}, {2.1, 2.3}}));

    expectVectorNear(flattenLayer.output(), Matrix1d{1.9, 1.9, 2.1, 2.3});
}

/**
 * @brief Verify that flattening is row-major, so element (i, j) lands at index i * size + j.
 */
TEST(FlattenLayerFlatten, FeedforwardIsRowMajor)
{
    constexpr std::size_t inputSize{3U};
    FlattenLayer flattenLayer{inputSize};
    const Matrix2d input{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};

    EXPECT_TRUE(flattenLayer.feedforward(input));

    // Expect 1..9 in reading order; column-major would give 1, 4, 7, 2, 5, 8, 3, 6, 9.
    expectVectorNear(flattenLayer.output(), Matrix1d{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});
}

/**
 * @brief Verify that a 1x1 input flattens to a single value.
 */
TEST(FlattenLayerFlatten, FeedforwardHandlesSingleValue)
{
    FlattenLayer flattenLayer{Test::SingleInputSize};
    EXPECT_TRUE(flattenLayer.feedforward(Matrix2d{{Test::SingleValue}}));

    expectVectorNear(flattenLayer.output(), Matrix1d{Test::SingleValue});
}

/**
 * @brief Verify that flattening copies values through untouched, negatives included.
 */
TEST(FlattenLayerFlatten, FeedforwardPreservesValues)
{
    FlattenLayer flattenLayer{Test::InputSize};
    EXPECT_TRUE(flattenLayer.feedforward(Matrix2d{{-1.5, 0.0}, {2.25, -3.75}}));

    expectVectorNear(flattenLayer.output(), Matrix1d{-1.5, 0.0, 2.25, -3.75});
}

/**
 * @brief Verify that feeding the same input twice yields the same output.
 */
TEST(FlattenLayerFlatten, FeedforwardIsRepeatable)
{
    FlattenLayer flattenLayer{Test::InputSize};
    const Matrix2d input{{1.9, 1.9}, {2.1, 2.3}};

    EXPECT_TRUE(flattenLayer.feedforward(input));
    const auto first = flattenLayer.output();

    EXPECT_TRUE(flattenLayer.feedforward(input));
    expectVectorNear(flattenLayer.output(), first);
}

/**
 * @brief Verify that feedforward rejects an input whose size doesn't match the layer.
 */
TEST(FlattenLayerFlatten, FeedforwardRejectsWrongSizedInput)
{
    FlattenLayer flattenLayer{Test::InputSize};
    const Matrix2d tooSmall{{1.0}};
    const Matrix2d tooLarge{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};

    EXPECT_FALSE(flattenLayer.feedforward(tooSmall));
    EXPECT_FALSE(flattenLayer.feedforward(tooLarge));
}

/**
 * @brief Verify that feedforward rejects a ragged input.
 */
TEST(FlattenLayerFlatten, FeedforwardRejectsRaggedInput)
{
    FlattenLayer flattenLayer{Test::InputSize};
    Matrix2d ragged{{1.0, 2.0}, {3.0, 4.0}};
    ragged[Test::ShortenedRow].pop_back();

    EXPECT_FALSE(flattenLayer.feedforward(ragged));
}

/**
 * @brief Verify that backpropagation folds the flat gradients back into a matrix.
 */
TEST(FlattenLayerFlatten, BackpropagateReshapesGradientsToMatrix)
{
    FlattenLayer flattenLayer{Test::InputSize};
    EXPECT_TRUE(flattenLayer.backpropagate(Matrix1d{10.0, 20.0, 30.0, 40.0}));

    expectMatrixNear(flattenLayer.inputGradients(), Matrix2d{{10.0, 20.0}, {30.0, 40.0}});
}

/**
 * @brief Verify that backpropagation unfolds row by row, matching the order feedforward used.
 */
TEST(FlattenLayerFlatten, BackpropagateIsRowMajor)
{
    constexpr std::size_t inputSize{3U};
    FlattenLayer flattenLayer{inputSize};

    EXPECT_TRUE(flattenLayer.backpropagate(Matrix1d{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}));

    expectMatrixNear(flattenLayer.inputGradients(),
                     Matrix2d{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}});
}

/**
 * @brief Verify that flattening and then unflattening returns the original matrix.
 */
TEST(FlattenLayerFlatten, BackpropagateUndoesFeedforward)
{
    constexpr std::size_t inputSize{3U};
    FlattenLayer flattenLayer{inputSize};
    const Matrix2d input{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};

    EXPECT_TRUE(flattenLayer.feedforward(input));

    // Feed the flattened output back in as if it were the gradients.
    EXPECT_TRUE(flattenLayer.backpropagate(flattenLayer.output()));
    expectMatrixNear(flattenLayer.inputGradients(), input);
}

/**
 * @brief Verify that backpropagation rejects gradients of the wrong length.
 */
TEST(FlattenLayerFlatten, BackpropagateRejectsWrongSizedGradients)
{
    FlattenLayer flattenLayer{Test::InputSize};
    const Matrix1d tooShort(Test::InputSize, Test::FillerValue);
    const Matrix1d tooLong(Test::OutputSize + Test::SizeDelta, Test::FillerValue);

    EXPECT_FALSE(flattenLayer.backpropagate(tooShort));
    EXPECT_FALSE(flattenLayer.backpropagate(tooLong));
}

/**
 * @brief Verify that backpropagating twice doesn't accumulate.
 */
TEST(FlattenLayerFlatten, BackpropagateOverwritesPreviousGradients)
{
    FlattenLayer flattenLayer{Test::InputSize};
    const Matrix1d gradients{10.0, 20.0, 30.0, 40.0};

    EXPECT_TRUE(flattenLayer.backpropagate(gradients));
    EXPECT_TRUE(flattenLayer.backpropagate(gradients));

    expectMatrixNear(flattenLayer.inputGradients(), Matrix2d{{10.0, 20.0}, {30.0, 40.0}});
}

/**
 * @brief Verify that the layer can be used through flatten_layer::Interface.
 */
TEST(FlattenLayerFlatten, UsableThroughInterface)
{
    FlattenLayer flattenLayer{Test::InputSize};
    ml::flatten_layer::Interface& layer{flattenLayer};

    EXPECT_EQ(layer.inputSize(), Test::InputSize);
    EXPECT_EQ(layer.outputSize(), Test::OutputSize);
    EXPECT_TRUE(layer.feedforward(Matrix2d{{1.9, 1.9}, {2.1, 2.3}}));
    expectVectorNear(layer.output(), Matrix1d{1.9, 1.9, 2.1, 2.3});

    EXPECT_TRUE(layer.backpropagate(Matrix1d{10.0, 20.0, 30.0, 40.0}));
    expectMatrixNear(layer.inputGradients(), Matrix2d{{10.0, 20.0}, {30.0, 40.0}});
}
} // namespace
