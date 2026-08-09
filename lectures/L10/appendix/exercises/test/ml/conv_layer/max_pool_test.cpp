/**
 * @file Unit tests for ml::conv_layer::MaxPool.
 */
#include <cstddef>

#include "ml/conv_layer/interface.hpp"
#include "ml/conv_layer/max_pool.hpp"
#include "ml/types.hpp"
#include "qacademy/test/test.hpp"

using MaxPoolLayer = ml::conv_layer::MaxPool;
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
    static constexpr std::size_t InputSize{4U};

    /** Pool size used by most tests. */
    static constexpr std::size_t PoolSize{2U};

    /** Output size implied by the two above. */
    static constexpr std::size_t OutputSize{InputSize / PoolSize};

    /** A learning rate inside the valid range. */
    static constexpr double LearningRate{0.1};

    /** Learning rates the conv layer would reject, which this layer accepts regardless. */
    static constexpr double ZeroLearningRate{0.0};
    static constexpr double NegativeLearningRate{-1.0};
    static constexpr double UpperBoundLearningRate{1.0};

    /** Value filling a matrix whose contents don't matter, only its size. */
    static constexpr double FillerValue{1.0};

    /** How far a deliberately wrong-sized matrix differs from the size the layer expects. */
    static constexpr std::size_t SizeDelta{1U};

    /** Input and pool size used by the tests that exercise a pool that downsamples nothing. */
    static constexpr std::size_t UnitPoolInputSize{3U};
    static constexpr std::size_t UnitPoolSize{1U};

    /** Row of the gradient matrix shortened by the ragged-matrix test. Any row would do. */
    static constexpr std::size_t ShortenedRow{1U};

    /**
     * Row of the input shortened by the ragged-input test. It has to be a row beyond the output
     * size, since a validation loop bounded by the output size never reaches those rows.
     */
    static constexpr std::size_t ShortenedInputRow{InputSize - 1U};

    /** Output size when one pool covers the whole input. */
    static constexpr std::size_t SinglePoolOutputSize{1U};
};

// -----------------------------------------------------------------------------
[[nodiscard]] Matrix2d convOutput() noexcept
{
    // The conv layer's output from the hand-training example in L06's appendix B.
    return Matrix2d{
        {1.3, 1.9, 1.9, 1.9}, {1.7, 1.7, 1.1, 1.9}, {1.7, 1.3, 0.5, 1.7}, {1.7, 2.1, 1.9, 2.3}};
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
 * @brief Verify that the layer downsamples by the pool size.
 */
TEST(ConvLayerMaxPool, ConstructedDimensions)
{
    const MaxPoolLayer maxPool{Test::InputSize, Test::PoolSize};
    EXPECT_EQ(maxPool.inputSize(), Test::InputSize);
    EXPECT_EQ(maxPool.outputSize(), Test::OutputSize);
}

/**
 * @brief Verify that the output is pooled-sized and the input gradients are input-sized.
 */
TEST(ConvLayerMaxPool, ConstructedMatrixShapes)
{
    const MaxPoolLayer maxPool{Test::InputSize, Test::PoolSize};
    EXPECT_EQ(maxPool.output().size(), Test::OutputSize);
    EXPECT_EQ(maxPool.inputGradients().size(), Test::InputSize);

    for (const auto& row : maxPool.output())
    {
        EXPECT_EQ(row.size(), Test::OutputSize);
    }

    for (const auto& row : maxPool.inputGradients())
    {
        EXPECT_EQ(row.size(), Test::InputSize);
    }
}

/**
 * @brief Verify the degenerate 1x1 pool: nothing is downsampled, so the output is input-sized.
 */
TEST(ConvLayerMaxPool, PoolSizeOneKeepsInputSize)
{
    const MaxPoolLayer maxPool{Test::InputSize, Test::UnitPoolSize};
    EXPECT_EQ(maxPool.outputSize(), Test::InputSize);
}

/**
 * @brief Verify that pooling with the whole input as one block collapses it to a single value.
 */
TEST(ConvLayerMaxPool, PoolSizeEqualToInputYieldsSingleValue)
{
    const MaxPoolLayer maxPool{Test::InputSize, Test::InputSize};
    EXPECT_EQ(maxPool.outputSize(), Test::SinglePoolOutputSize);
}

/**
 * @brief Verify feedforward against the hand-training example in L06's appendix B.
 */
TEST(ConvLayerMaxPool, FeedforwardMatchesHandTrainedExample)
{
    MaxPoolLayer maxPool{Test::InputSize, Test::PoolSize};
    EXPECT_TRUE(maxPool.feedforward(convOutput()));

    expectMatrixNear(maxPool.output(), Matrix2d{{1.9, 1.9}, {2.1, 2.3}});
}

/**
 * @brief Verify that each output comes from its own block, using distinct values throughout.
 */
TEST(ConvLayerMaxPool, FeedforwardSelectsLargestInEachPool)
{
    MaxPoolLayer maxPool{Test::InputSize, Test::PoolSize};
    const Matrix2d input{{1.0, 2.0, 3.0, 4.0},
                         {5.0, 6.0, 7.0, 8.0},
                         {9.0, 10.0, 11.0, 12.0},
                         {13.0, 14.0, 15.0, 16.0}};

    EXPECT_TRUE(maxPool.feedforward(input));

    // Expect the bottom-right value of each 2x2 block, which is the largest here.
    expectMatrixNear(maxPool.output(), Matrix2d{{6.0, 8.0}, {14.0, 16.0}});
}

/**
 * @brief Verify that a 1x1 pool passes the input through unchanged.
 */
TEST(ConvLayerMaxPool, FeedforwardWithPoolSizeOneIsIdentity)
{
    MaxPoolLayer maxPool{Test::UnitPoolInputSize, Test::UnitPoolSize};
    const Matrix2d input{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};

    EXPECT_TRUE(maxPool.feedforward(input));
    expectMatrixNear(maxPool.output(), input);
}

/**
 * @brief Verify that pooling the whole input as one block yields the largest value in it.
 */
TEST(ConvLayerMaxPool, FeedforwardWithFullPoolYieldsGlobalMax)
{
    MaxPoolLayer maxPool{Test::InputSize, Test::InputSize};
    EXPECT_TRUE(maxPool.feedforward(convOutput()));

    expectMatrixNear(maxPool.output(), Matrix2d{{2.3}});
}

/**
 * @brief Verify that an all-negative input pools to its least negative values.
 */
TEST(ConvLayerMaxPool, FeedforwardHandlesNegativeValues)
{
    MaxPoolLayer maxPool{Test::InputSize, Test::PoolSize};
    const Matrix2d input{{-5.0, -3.0, -9.0, -7.0},
                         {-8.0, -1.0, -6.0, -4.0},
                         {-2.0, -6.0, -3.0, -1.0},
                         {-7.0, -9.0, -8.0, -5.0}};

    EXPECT_TRUE(maxPool.feedforward(input));
    expectMatrixNear(maxPool.output(), Matrix2d{{-1.0, -4.0}, {-2.0, -1.0}});
}

/**
 * @brief Verify that feeding the same input twice yields the same output.
 */
TEST(ConvLayerMaxPool, FeedforwardIsRepeatable)
{
    MaxPoolLayer maxPool{Test::InputSize, Test::PoolSize};
    EXPECT_TRUE(maxPool.feedforward(convOutput()));
    const auto first = maxPool.output();

    EXPECT_TRUE(maxPool.feedforward(convOutput()));
    expectMatrixNear(maxPool.output(), first);
}

/**
 * @brief Verify that feedforward rejects an input whose size doesn't match the layer.
 */
TEST(ConvLayerMaxPool, FeedforwardRejectsWrongSizedInput)
{
    constexpr std::size_t smallerSize{Test::InputSize - Test::SizeDelta};
    constexpr std::size_t largerSize{Test::InputSize + Test::SizeDelta};

    MaxPoolLayer maxPool{Test::InputSize, Test::PoolSize};
    const Matrix2d tooSmall(smallerSize, Matrix1d(smallerSize, Test::FillerValue));
    const Matrix2d tooLarge(largerSize, Matrix1d(largerSize, Test::FillerValue));

    EXPECT_FALSE(maxPool.feedforward(tooSmall));
    EXPECT_FALSE(maxPool.feedforward(tooLarge));
}

/**
 * @brief Verify that feedforward rejects a ragged input.
 */
TEST(ConvLayerMaxPool, FeedforwardRejectsRaggedInput)
{
    MaxPoolLayer maxPool{Test::InputSize, Test::PoolSize};
    auto ragged = convOutput();
    ragged[Test::ShortenedInputRow].pop_back();

    EXPECT_FALSE(maxPool.feedforward(ragged));
}

/**
 * @brief Verify backpropagation against the hand-training example in L06's appendix B.
 */
TEST(ConvLayerMaxPool, BackpropagateMatchesHandTrainedExample)
{
    MaxPoolLayer maxPool{Test::InputSize, Test::PoolSize};
    EXPECT_TRUE(maxPool.feedforward(convOutput()));
    EXPECT_TRUE(maxPool.backpropagate(Matrix2d{{10.0, 20.0}, {30.0, 40.0}}));

    expectMatrixNear(maxPool.inputGradients(), Matrix2d{{0.0, 10.0, 20.0, 0.0},
                                                        {0.0, 0.0, 0.0, 0.0},
                                                        {0.0, 0.0, 0.0, 0.0},
                                                        {0.0, 30.0, 0.0, 40.0}});
}

/**
 * @brief Verify that a tied maximum routes its gradient to the first occurrence.
 */
TEST(ConvLayerMaxPool, BackpropagateRoutesToFirstMaxOnTies)
{
    MaxPoolLayer maxPool{Test::InputSize, Test::PoolSize};
    const Matrix2d input{
        {5.0, 5.0, 7.0, 3.0}, {1.0, 1.0, 2.0, 7.0}, {9.0, 4.0, 6.0, 6.0}, {9.0, 2.0, 1.0, 6.0}};

    EXPECT_TRUE(maxPool.feedforward(input));
    expectMatrixNear(maxPool.output(), Matrix2d{{5.0, 7.0}, {9.0, 6.0}});

    EXPECT_TRUE(maxPool.backpropagate(Matrix2d{{1.0, 2.0}, {3.0, 4.0}}));

    // Expect each gradient at the first occurrence of its block's max.
    expectMatrixNear(maxPool.inputGradients(), Matrix2d{{1.0, 0.0, 2.0, 0.0},
                                                        {0.0, 0.0, 0.0, 0.0},
                                                        {3.0, 0.0, 4.0, 0.0},
                                                        {0.0, 0.0, 0.0, 0.0}});
}

/**
 * @brief Verify that a 1x1 pool routes every gradient straight through.
 */
TEST(ConvLayerMaxPool, BackpropagateWithPoolSizeOneRoutesEverything)
{
    MaxPoolLayer maxPool{Test::UnitPoolInputSize, Test::UnitPoolSize};
    const Matrix2d input{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};
    const Matrix2d gradients{{9.0, 8.0, 7.0}, {6.0, 5.0, 4.0}, {3.0, 2.0, 1.0}};

    EXPECT_TRUE(maxPool.feedforward(input));
    EXPECT_TRUE(maxPool.backpropagate(gradients));

    expectMatrixNear(maxPool.inputGradients(), gradients);
}

/**
 * @brief Verify that backpropagating twice doesn't accumulate.
 */
TEST(ConvLayerMaxPool, BackpropagateResetsGradientsBetweenCalls)
{
    MaxPoolLayer maxPool{Test::InputSize, Test::PoolSize};
    const Matrix2d gradients{{10.0, 20.0}, {30.0, 40.0}};

    EXPECT_TRUE(maxPool.feedforward(convOutput()));
    EXPECT_TRUE(maxPool.backpropagate(gradients));
    const auto first = maxPool.inputGradients();

    EXPECT_TRUE(maxPool.backpropagate(gradients));
    expectMatrixNear(maxPool.inputGradients(), first);
}

/**
 * @brief Verify that backpropagation routes according to the most recent input.
 */
TEST(ConvLayerMaxPool, BackpropagateUsesLatestInput)
{
    MaxPoolLayer maxPool{Test::InputSize, Test::PoolSize};
    const Matrix2d firstInput{
        {9.0, 0.0, 9.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {9.0, 0.0, 9.0, 0.0}, {0.0, 0.0, 0.0, 0.0}};
    const Matrix2d secondInput{
        {0.0, 0.0, 0.0, 0.0}, {0.0, 9.0, 0.0, 9.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 9.0, 0.0, 9.0}};

    EXPECT_TRUE(maxPool.feedforward(firstInput));
    EXPECT_TRUE(maxPool.feedforward(secondInput));
    EXPECT_TRUE(maxPool.backpropagate(Matrix2d{{1.0, 2.0}, {3.0, 4.0}}));

    // Expect the gradients at the second input's maxima.
    expectMatrixNear(maxPool.inputGradients(), Matrix2d{{0.0, 0.0, 0.0, 0.0},
                                                        {0.0, 1.0, 0.0, 2.0},
                                                        {0.0, 0.0, 0.0, 0.0},
                                                        {0.0, 3.0, 0.0, 4.0}});
}

/**
 * @brief Verify that backpropagation rejects gradients sized like the input, not the output.
 */
TEST(ConvLayerMaxPool, BackpropagateRejectsWrongSizedGradients)
{
    MaxPoolLayer maxPool{Test::InputSize, Test::PoolSize};
    EXPECT_TRUE(maxPool.feedforward(convOutput()));

    const Matrix2d inputSized(Test::InputSize, Matrix1d(Test::InputSize, Test::FillerValue));
    EXPECT_FALSE(maxPool.backpropagate(inputSized));
}

/**
 * @brief Verify that backpropagation rejects ragged gradients.
 */
TEST(ConvLayerMaxPool, BackpropagateRejectsRaggedGradients)
{
    MaxPoolLayer maxPool{Test::InputSize, Test::PoolSize};
    EXPECT_TRUE(maxPool.feedforward(convOutput()));

    Matrix2d ragged{{10.0, 20.0}, {30.0, 40.0}};
    ragged[Test::ShortenedRow].pop_back();

    EXPECT_FALSE(maxPool.backpropagate(ragged));
}

/**
 * @brief Verify that optimization accepts any learning rate, there being nothing to update.
 */
TEST(ConvLayerMaxPool, OptimizeAcceptsAnyLearningRate)
{
    MaxPoolLayer maxPool{Test::InputSize, Test::PoolSize};
    EXPECT_TRUE(maxPool.feedforward(convOutput()));
    EXPECT_TRUE(maxPool.backpropagate(Matrix2d{{10.0, 20.0}, {30.0, 40.0}}));

    EXPECT_TRUE(maxPool.optimize(Test::LearningRate));
    EXPECT_TRUE(maxPool.optimize(Test::ZeroLearningRate));
    EXPECT_TRUE(maxPool.optimize(Test::NegativeLearningRate));
    EXPECT_TRUE(maxPool.optimize(Test::UpperBoundLearningRate));
}

/**
 * @brief Verify that optimization is a genuine no-op.
 */
TEST(ConvLayerMaxPool, OptimizeChangesNothing)
{
    MaxPoolLayer maxPool{Test::InputSize, Test::PoolSize};
    EXPECT_TRUE(maxPool.feedforward(convOutput()));
    EXPECT_TRUE(maxPool.backpropagate(Matrix2d{{10.0, 20.0}, {30.0, 40.0}}));

    const auto output    = maxPool.output();
    const auto gradients = maxPool.inputGradients();

    EXPECT_TRUE(maxPool.optimize(Test::LearningRate));

    expectMatrixNear(maxPool.output(), output);
    expectMatrixNear(maxPool.inputGradients(), gradients);
}

/**
 * @brief Verify that the layer can be used through conv_layer::Interface.
 */
TEST(ConvLayerMaxPool, UsableThroughInterface)
{
    MaxPoolLayer maxPool{Test::InputSize, Test::PoolSize};
    ml::conv_layer::Interface& layer{maxPool};

    EXPECT_EQ(layer.inputSize(), Test::InputSize);
    EXPECT_EQ(layer.outputSize(), Test::OutputSize);
    EXPECT_TRUE(layer.feedforward(convOutput()));
    EXPECT_TRUE(layer.backpropagate(Matrix2d{{10.0, 20.0}, {30.0, 40.0}}));
    EXPECT_TRUE(layer.optimize(Test::LearningRate));

    expectMatrixNear(layer.output(), Matrix2d{{1.9, 1.9}, {2.1, 2.3}});
}
} // namespace
