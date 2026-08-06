/**
 * @file Unit and component tests for ml::lin_reg::Fixed.
 *
 * @note The component tests check each prediction with std::isfinite() first, because a diverged
 *       model yields NaN and every NaN comparison is false, including the one in EXPECT_NEAR().
 */
#include <algorithm>
#include <cmath>
#include <cstddef>

#include "ml/lin_reg/fixed.hpp"
#include "ml/lin_reg/interface.hpp"
#include "ml/types.hpp"
#include "qacademy/test/test.hpp"

using LinReg          = ml::lin_reg::Fixed;
using LinRegInterface = ml::lin_reg::Interface;
using Matrix          = ml::Matrix1d;

namespace
{
/**
 * @brief Test parameters.
 */
struct Test
{
    /** Value tolerance for hand-computed results when using EXPECT_NEAR(). */
    static constexpr double ExactTolerance{1e-9};

    /** Value tolerance for results produced by a full training run. */
    static constexpr double ConvergenceTolerance{1e-4};
};

// -----------------------------------------------------------------------------
void createTrainingData(Matrix& trainInput, Matrix& trainOutput, const double bias,
                        const double weight) noexcept
{
    const auto setCount = std::min(trainInput.size(), trainOutput.size());

    for (std::size_t i{}; i < setCount; ++i)
    {
        trainInput[i]  = static_cast<double>(i);
        trainOutput[i] = weight * static_cast<double>(i) + bias;
    }
}

/**
 * @brief Verify that an untrained model predicts zero for every input.
 */
TEST(LinRegFixed, PredictBeforeTraining)
{
    constexpr double zero{0.0};
    constexpr double inputs[]{0.0, 1.0, -2.5, 100.0};

    const Matrix trainInput{1.0, 2.0, 3.0};
    const Matrix trainOutput{2.0, 4.0, 6.0};
    const LinReg linReg{trainInput, trainOutput};

    for (const auto input : inputs)
    {
        EXPECT_NEAR(linReg.predict(input), zero, Test::ExactTolerance);
    }
}

/**
 * @brief Verify that train() returns false when the epoch count is zero, leaving the model
 *        parameters untouched.
 */
TEST(LinRegFixed, ZeroEpochCount)
{
    constexpr std::size_t epochCount{0U};
    constexpr double learningRate{0.1};
    constexpr double input{1.0};
    constexpr double zero{0.0};

    const Matrix trainInput{1.0};
    const Matrix trainOutput{5.0};
    LinReg linReg{trainInput, trainOutput};

    // Test training for zero epochs.
    // Expect training to fail.
    EXPECT_FALSE(linReg.train(epochCount, learningRate));

    // Test prediction after the rejected training.
    // Expect the model parameters to be untouched.
    EXPECT_NEAR(linReg.predict(input), zero, Test::ExactTolerance);
}

/**
 * @brief Verify that train() returns false for a learning rate of zero or less, leaving the
 *        model parameters untouched.
 */
TEST(LinRegFixed, NonPositiveLearningRate)
{
    constexpr std::size_t epochCount{10U};
    constexpr double learningRates[]{0.0, -0.1};
    constexpr double input{1.0};
    constexpr double zero{0.0};

    const Matrix trainInput{1.0};
    const Matrix trainOutput{5.0};
    LinReg linReg{trainInput, trainOutput};

    // Test invalid learning rates.
    // Expect training to fail.
    for (const auto learningRate : learningRates)
    {
        EXPECT_FALSE(linReg.train(epochCount, learningRate));
    }

    // Test prediction after the rejected training.
    // Expect the model parameters to be untouched.
    EXPECT_NEAR(linReg.predict(input), zero, Test::ExactTolerance);
}

/**
 * @brief Verify that train() returns false for a learning rate of 1.0 or more, leaving the
 *        model parameters untouched.
 */
TEST(LinRegFixed, LearningRateAtOrAboveOne)
{
    constexpr std::size_t epochCount{10U};
    constexpr double learningRates[]{1.0, 1.5};
    constexpr double input{1.0};
    constexpr double zero{0.0};

    const Matrix trainInput{1.0};
    const Matrix trainOutput{5.0};
    LinReg linReg{trainInput, trainOutput};

    // Test learning rates at and above the upper bound.
    // Expect training to fail.
    for (const auto learningRate : learningRates)
    {
        EXPECT_FALSE(linReg.train(epochCount, learningRate));
    }

    // Test prediction after the rejected training.
    // Expect the model parameters to be untouched.
    EXPECT_NEAR(linReg.predict(input), zero, Test::ExactTolerance);
}

/**
 * @brief Verify that train() returns true for a valid epoch count and learning rate.
 */
TEST(LinRegFixed, ValidArguments)
{
    constexpr std::size_t singleEpoch{1U};
    constexpr std::size_t manyEpochs{1000U};
    constexpr double smallLearningRate{0.1};
    constexpr double largeLearningRate{0.99};

    const Matrix trainInput{1.0};
    const Matrix trainOutput{5.0};
    LinReg linReg{trainInput, trainOutput};

    // Test the smallest valid epoch count, and a learning rate just inside the upper bound.
    // Expect training to succeed in both cases.
    EXPECT_TRUE(linReg.train(singleEpoch, smallLearningRate));
    EXPECT_TRUE(linReg.train(manyEpochs, largeLearningRate));
}

/**
 * @brief Verify that a single epoch on the training set (x = 1, y_ref = 5) with a learning rate
 *        of 0.1 adjusts both the bias and the weight to 0.5.
 */
TEST(LinRegFixed, SingleEpoch)
{
    constexpr std::size_t epochCount{1U};
    constexpr double learningRate{0.1};
    constexpr double biasInput{0.0};
    constexpr double input{1.0};
    constexpr double expectedBias{0.5};
    constexpr double expectedPrediction{1.0};

    const Matrix trainInput{1.0};
    const Matrix trainOutput{5.0};
    LinReg linReg{trainInput, trainOutput};

    EXPECT_TRUE(linReg.train(epochCount, learningRate));

    // Test prediction with input 0, which reads back the bias on its own, since y = k * 0 + m.
    // Expect the bias to have been adjusted to 0.5.
    EXPECT_NEAR(linReg.predict(biasInput), expectedBias, Test::ExactTolerance);

    // Test prediction with input 1.
    // Expect weight * 1 + bias = 0.5 + 0.5 = 1.0.
    EXPECT_NEAR(linReg.predict(input), expectedPrediction, Test::ExactTolerance);
}

/**
 * @brief Verify that a second epoch continues from the parameters left behind by the first,
 *        adjusting both the bias and the weight to 0.9.
 */
TEST(LinRegFixed, TwoEpochs)
{
    constexpr std::size_t epochCount{2U};
    constexpr double learningRate{0.1};
    constexpr double biasInput{0.0};
    constexpr double input{1.0};
    constexpr double expectedBias{0.9};
    constexpr double expectedPrediction{1.8};

    const Matrix trainInput{1.0};
    const Matrix trainOutput{5.0};
    LinReg linReg{trainInput, trainOutput};

    EXPECT_TRUE(linReg.train(epochCount, learningRate));

    // Test prediction with input 0, which reads back the bias on its own.
    // Expect the bias to have been adjusted to 0.9.
    EXPECT_NEAR(linReg.predict(biasInput), expectedBias, Test::ExactTolerance);

    // Test prediction with input 1.
    // Expect weight * 1 + bias = 0.9 + 0.9 = 1.8.
    EXPECT_NEAR(linReg.predict(input), expectedPrediction, Test::ExactTolerance);
}

/**
 * @brief Verify that training one epoch twice gives the same result as training two epochs in a
 *        single call, i.e. that train() never starts over.
 */
TEST(LinRegFixed, RepeatedTrainCalls)
{
    constexpr std::size_t singleEpoch{1U};
    constexpr std::size_t twoEpochs{2U};
    constexpr double learningRate{0.1};
    constexpr double inputs[]{0.0, 2.0};

    const Matrix trainInput{1.0};
    const Matrix trainOutput{5.0};
    LinReg twoCalls{trainInput, trainOutput};
    LinReg oneCall{trainInput, trainOutput};

    // Test training one epoch twice, against training two epochs in one call.
    // Expect all three training runs to succeed.
    EXPECT_TRUE(twoCalls.train(singleEpoch, learningRate));
    EXPECT_TRUE(twoCalls.train(singleEpoch, learningRate));
    EXPECT_TRUE(oneCall.train(twoEpochs, learningRate));

    // Test prediction with both models.
    // Expect identical predictions, since neither call started training over.
    for (const auto input : inputs)
    {
        EXPECT_NEAR(twoCalls.predict(input), oneCall.predict(input), Test::ExactTolerance);
    }
}

/**
 * @brief Verify that omitting the learning rate selects the default of 0.01, giving a tenth of
 *        the steps taken with a learning rate of 0.1.
 */
TEST(LinRegFixed, DefaultLearningRate)
{
    constexpr std::size_t epochCount{1U};
    constexpr double biasInput{0.0};
    constexpr double input{1.0};
    constexpr double expectedBias{0.05};
    constexpr double expectedPrediction{0.1};

    const Matrix trainInput{1.0};
    const Matrix trainOutput{5.0};
    LinReg linReg{trainInput, trainOutput};

    // Test training without passing a learning rate.
    // Expect training to succeed using the default learning rate.
    EXPECT_TRUE(linReg.train(epochCount));

    // Test prediction with input 0, which reads back the bias on its own.
    // Expect the bias to have been adjusted to 0.05.
    EXPECT_NEAR(linReg.predict(biasInput), expectedBias, Test::ExactTolerance);

    // Test prediction with input 1.
    // Expect weight * 1 + bias = 0.05 + 0.05 = 0.1.
    EXPECT_NEAR(linReg.predict(input), expectedPrediction, Test::ExactTolerance);
}

/**
 * @brief Verify that an input of zero assigns the bias directly and leaves the weight untouched,
 *        so that every prediction equals the reference value.
 */
TEST(LinRegFixed, ZeroInput)
{
    constexpr std::size_t epochCount{1U};
    constexpr double learningRate{0.1};
    constexpr double referenceOutput{7.0};
    constexpr double inputs[]{0.0, 3.0, -3.0};

    const Matrix trainInput{0.0};
    const Matrix trainOutput{referenceOutput};
    LinReg linReg{trainInput, trainOutput};

    EXPECT_TRUE(linReg.train(epochCount, learningRate));

    // Test prediction across several inputs.
    // Expect every prediction to equal the reference output, since the bias was assigned
    // directly and the weight was left at zero.
    for (const auto input : inputs)
    {
        EXPECT_NEAR(linReg.predict(input), referenceOutput, Test::ExactTolerance);
    }
}

/**
 * @brief Verify that training vectors of different lengths train on the complete sets only,
 *        i.e. on min(inputs, outputs) sets, leaving the surplus values unused.
 */
TEST(LinRegFixed, MismatchedTrainingData)
{
    constexpr std::size_t epochCount{1U};
    constexpr double learningRate{0.1};
    constexpr double biasInput{0.0};
    constexpr double input{1.0};
    constexpr double expectedBias{0.5};
    constexpr double expectedPrediction{1.0};

    const Matrix threeInputs{1.0, 2.0, 3.0};
    const Matrix oneInput{1.0};
    const Matrix threeOutputs{5.0, 9.0, 13.0};
    const Matrix oneOutput{5.0};
    LinReg moreInputs{threeInputs, oneOutput};
    LinReg moreOutputs{oneInput, threeOutputs};

    EXPECT_TRUE(moreInputs.train(epochCount, learningRate));
    EXPECT_TRUE(moreOutputs.train(epochCount, learningRate));

    // Test prediction with both models.
    // Expect the result of training the single complete set (x = 1, y_ref = 5), i.e. the
    // surplus values in the longer vector to have been ignored.
    EXPECT_NEAR(moreInputs.predict(biasInput), expectedBias, Test::ExactTolerance);
    EXPECT_NEAR(moreInputs.predict(input), expectedPrediction, Test::ExactTolerance);
    EXPECT_NEAR(moreOutputs.predict(biasInput), expectedBias, Test::ExactTolerance);
    EXPECT_NEAR(moreOutputs.predict(input), expectedPrediction, Test::ExactTolerance);
}

/**
 * @brief Verify that the model reproduces the training data for y = 2x + 2, i.e. the output
 *        printed by main.cpp.
 */
TEST(LinRegComponent, HappyPath)
{
    constexpr std::size_t setCount{5U};
    constexpr double bias{2.0};
    constexpr double weight{2.0};
    constexpr std::size_t epochCount{1000U};

    Matrix trainInput(setCount);
    Matrix trainOutput(setCount);
    createTrainingData(trainInput, trainOutput, bias, weight);

    LinReg linReg{trainInput, trainOutput};

    EXPECT_TRUE(linReg.train(epochCount));

    // Test prediction with each training input.
    // Expect the model to reproduce the training data.
    for (std::size_t i{}; i < setCount; ++i)
    {
        const auto prediction = linReg.predict(trainInput[i]);
        EXPECT_TRUE(std::isfinite(prediction));
        EXPECT_NEAR(prediction, trainOutput[i], Test::ConvergenceTolerance);
    }
}

/**
 * @brief Verify that the model converges to y = 3x + 2 and generalizes across the range [-5, 5],
 *        most of which was never part of the training data.
 */
TEST(LinRegComponent, Generalization)
{
    constexpr std::size_t setCount{5U};
    constexpr double bias{2.0};
    constexpr double weight{3.0};
    constexpr std::size_t epochCount{1000U};

    Matrix trainInput(setCount);
    Matrix trainOutput(setCount);
    createTrainingData(trainInput, trainOutput, bias, weight);

    LinReg linReg{trainInput, trainOutput};

    EXPECT_TRUE(linReg.train(epochCount));

    constexpr double startVal{-5.0};
    constexpr double endVal{5.0};
    constexpr double step{1.0};

    // Test prediction across the range [-5, 5], most of which was never trained on.
    // Expect the model to reproduce the line for every input.
    for (double input{startVal}; input <= endVal; input += step)
    {
        const auto prediction = linReg.predict(input);
        EXPECT_TRUE(std::isfinite(prediction));
        EXPECT_NEAR(prediction, weight * input + bias, Test::ConvergenceTolerance);
    }
}

/**
 * @brief Verify that the model converges with a negative weight, which only works if the sign of
 *        the error is carried through the weight update.
 */
TEST(LinRegComponent, NegativeParameters)
{
    constexpr std::size_t setCount{5U};
    constexpr double bias{1.0};
    constexpr double weight{-2.0};
    constexpr std::size_t epochCount{1000U};

    Matrix trainInput(setCount);
    Matrix trainOutput(setCount);
    createTrainingData(trainInput, trainOutput, bias, weight);

    LinReg linReg{trainInput, trainOutput};

    EXPECT_TRUE(linReg.train(epochCount));

    constexpr double startVal{-5.0};
    constexpr double endVal{5.0};
    constexpr double step{1.0};

    // Test prediction across the range [-5, 5].
    // Expect the model to reproduce the line, i.e. to have driven the weight negative rather
    // than away from the reference values.
    for (double input{startVal}; input <= endVal; input += step)
    {
        const auto prediction = linReg.predict(input);
        EXPECT_TRUE(std::isfinite(prediction));
        EXPECT_NEAR(prediction, weight * input + bias, Test::ConvergenceTolerance);
    }
}

/**
 * @brief Verify that prediction through ml::lin_reg::Interface dispatches to the concrete class,
 *        which is what lets main.cpp predict without knowing which model it was handed.
 */
TEST(LinRegComponent, PredictionThroughInterface)
{
    constexpr std::size_t setCount{5U};
    constexpr double bias{2.0};
    constexpr double weight{2.0};
    constexpr std::size_t epochCount{1000U};

    Matrix trainInput(setCount);
    Matrix trainOutput(setCount);
    createTrainingData(trainInput, trainOutput, bias, weight);

    LinReg linReg{trainInput, trainOutput};

    EXPECT_TRUE(linReg.train(epochCount));

    const LinRegInterface& linRegInterface{linReg};

    constexpr double startVal{-5.0};
    constexpr double endVal{5.0};
    constexpr double step{1.0};

    // Test prediction through the interface against prediction through the concrete class.
    // Expect identical results, since the interface call dispatches to Fixed::predict().
    for (double input{startVal}; input <= endVal; input += step)
    {
        EXPECT_EQ(linRegInterface.predict(input), linReg.predict(input));
    }
}
} // namespace
