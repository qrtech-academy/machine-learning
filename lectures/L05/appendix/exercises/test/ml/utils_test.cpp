/**
 * @file Unit tests for the helper functions in ml/utils.hpp.
 */
#include <cmath>
#include <cstddef>

#include "ml/types.hpp"
#include "ml/utils.hpp"
#include "qacademy/test/test.hpp"

using ActFunc = ml::ActFunc;

namespace
{
/**
 * @brief Test parameters.
 */
struct Test
{
    /** Value tolerance when using EXPECT_NEAR(). */
    static constexpr double Tolerance{1e-9};

    /** Number of draws taken by the randomStartVal() tests. */
    static constexpr std::size_t DrawCount{1000U};
};

/**
 * @brief Verify that every random start value is finite and inside the range [0.0, 1.0].
 */
TEST(Utils, RandomStartValRange)
{
    ml::initRandGen();

    for (std::size_t i{}; i < Test::DrawCount; ++i)
    {
        const auto value = ml::randomStartVal();
        EXPECT_TRUE(std::isfinite(value));
        EXPECT_TRUE((0.0 <= value) && (1.0 >= value));
    }
}

/**
 * @brief Verify that the random start values actually vary instead of repeating one number.
 */
TEST(Utils, RandomStartValVaries)
{
    ml::initRandGen();
    const auto first = ml::randomStartVal();
    bool varies{false};

    for (std::size_t i{}; (i < Test::DrawCount) && !varies; ++i)
    {
        varies = (ml::randomStartVal() != first);
    }

    // Test that at least one of the draws differs from the first.
    // Expect variation: a layer whose weights all start identical can never break their symmetry.
    EXPECT_TRUE(varies);
}

/**
 * @brief Verify that ReLU passes positive input through unchanged and clamps the rest to zero.
 */
TEST(Utils, ActFuncOutputRelu)
{
    constexpr double positives[]{1e-9, 0.5, 1.0, 42.0};
    constexpr double nonPositives[]{0.0, -1e-9, -0.5, -42.0};

    for (const auto value : positives)
    {
        EXPECT_NEAR(ml::actFuncOutput(ActFunc::Relu, value), value, Test::Tolerance);
    }
    for (const auto value : nonPositives)
    {
        EXPECT_NEAR(ml::actFuncOutput(ActFunc::Relu, value), 0.0, Test::Tolerance);
    }
}

/**
 * @brief Verify that Tanh matches std::tanh and stays inside the range [-1.0, 1.0].
 */
TEST(Utils, ActFuncOutputTanh)
{
    constexpr double values[]{-42.0, -1.0, -0.5, 0.0, 0.5, 1.0, 42.0};

    for (const auto value : values)
    {
        const auto output = ml::actFuncOutput(ActFunc::Tanh, value);
        EXPECT_NEAR(output, std::tanh(value), Test::Tolerance);
        EXPECT_TRUE((-1.0 <= output) && (1.0 >= output));
    }
}

/**
 * @brief Verify that None applies no activation at all, returning its input unchanged.
 */
TEST(Utils, ActFuncOutputNone)
{
    constexpr double values[]{-42.0, -1.0, -1e-9, 0.0, 1e-9, 0.5, 42.0};

    for (const auto value : values)
    {
        EXPECT_NEAR(ml::actFuncOutput(ActFunc::None, value), value, Test::Tolerance);
    }
}

/**
 * @brief Verify that ReLU's derivative is one above zero and zero at or below it.
 */
TEST(Utils, ActFuncDeltaRelu)
{
    constexpr double positives[]{1e-9, 0.5, 1.0, 42.0};
    constexpr double nonPositives[]{0.0, -1e-9, -0.5, -42.0};

    for (const auto value : positives)
    {
        EXPECT_NEAR(ml::actFuncDelta(ActFunc::Relu, value), 1.0, Test::Tolerance);
    }

    // Test the derivative at and below zero.
    // Expect zero, which is what makes a node that goes negative stop learning: the "dying ReLU".
    for (const auto value : nonPositives)
    {
        EXPECT_NEAR(ml::actFuncDelta(ActFunc::Relu, value), 0.0, Test::Tolerance);
    }
}

/**
 * @brief Verify that Tanh's derivative is 1 - tanh(x)^2, peaking at 1.0 for an input of zero.
 */
TEST(Utils, ActFuncDeltaTanh)
{
    constexpr double values[]{-42.0, -1.0, -0.5, 0.0, 0.5, 1.0, 42.0};

    for (const auto value : values)
    {
        const auto tanhOutput = std::tanh(value);
        EXPECT_NEAR(ml::actFuncDelta(ActFunc::Tanh, value), 1.0 - tanhOutput * tanhOutput,
                    Test::Tolerance);
    }
    EXPECT_NEAR(ml::actFuncDelta(ActFunc::Tanh, 0.0), 1.0, Test::Tolerance);

    // Test the derivative far out on either tail.
    // Expect exactly zero: std::tanh saturates to 1.0 in double well before an input of 42, so a
    // saturated Tanh node stops learning just as a dead ReLU node does.
    EXPECT_NEAR(ml::actFuncDelta(ActFunc::Tanh, 42.0), 0.0, Test::Tolerance);
    EXPECT_NEAR(ml::actFuncDelta(ActFunc::Tanh, -42.0), 0.0, Test::Tolerance);
}

/**
 * @brief Verify that None's derivative is one everywhere, as the identity function's must be.
 */
TEST(Utils, ActFuncDeltaNone)
{
    constexpr double values[]{-42.0, -1.0, 0.0, 0.5, 42.0};

    for (const auto value : values)
    {
        EXPECT_NEAR(ml::actFuncDelta(ActFunc::None, value), 1.0, Test::Tolerance);
    }
}
} // namespace
