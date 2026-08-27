/**
 * @file Unit and component tests for ml::lin_reg::Adaptive.
 */
#include <algorithm>
#include <cmath>
#include <cstddef>

#include "ml/lin_reg/adaptive.hpp"
#include "ml/lin_reg/fixed.hpp"
#include "ml/lin_reg/interface.hpp"
#include "ml/types.hpp"
#include "qacademy/test/test.hpp"

using LinReg          = ml::lin_reg::Adaptive;
using LinRegFixed     = ml::lin_reg::Fixed;
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

    /** The learning rate the model starts every training run with. */
    static constexpr double InitialLearningRate{0.1};
};

// -----------------------------------------------------------------------------
void createTrainingData(Matrix& trainInput, Matrix& trainOutput, const double bias,
                        const double weight, const double step) noexcept
{
    const auto setCount = std::min(trainInput.size(), trainOutput.size());

    for (std::size_t i{}; i < setCount; ++i)
    {
        trainInput[i]  = static_cast<double>(i) * step;
        trainOutput[i] = weight * trainInput[i] + bias;
    }
}

// -----------------------------------------------------------------------------
double meanAbsoluteError(const LinRegInterface& linReg, const Matrix& trainInput,
                         const Matrix& trainOutput) noexcept
{
    const auto setCount = std::min(trainInput.size(), trainOutput.size());
    double sum{};

    for (std::size_t i{}; i < setCount; ++i)
    {
        sum += std::abs(trainOutput[i] - linReg.predict(trainInput[i]));
    }
    return sum / static_cast<double>(setCount);
}

/**
 * @brief Verify that an untrained model predicts zero for every input.
 */
TEST(LinRegAdaptive, PredictBeforeTraining)
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
TEST(LinRegAdaptive, ZeroEpochCount)
{
    constexpr std::size_t epochCount{0U};
    constexpr double input{1.0};
    constexpr double zero{0.0};

    const Matrix trainInput{1.0};
    const Matrix trainOutput{5.0};
    LinReg linReg{trainInput, trainOutput};

    // Test training for zero epochs.
    // Expect training to fail.
    EXPECT_FALSE(linReg.train(epochCount));

    // Test prediction after the rejected training.
    // Expect the model parameters to be untouched.
    EXPECT_NEAR(linReg.predict(input), zero, Test::ExactTolerance);
}

/**
 * @brief Verify that train() returns false for a precision threshold of zero or less, leaving
 *        the model parameters untouched.
 */
TEST(LinRegAdaptive, NonPositivePrecisionThreshold)
{
    constexpr std::size_t epochCount{10U};
    constexpr double precisionThresholds[]{0.0, -0.5};
    constexpr double input{1.0};
    constexpr double zero{0.0};

    const Matrix trainInput{1.0};
    const Matrix trainOutput{5.0};
    LinReg linReg{trainInput, trainOutput};

    // Test precision thresholds at and below the lower bound.
    // Expect training to fail.
    for (const auto precisionThreshold : precisionThresholds)
    {
        EXPECT_FALSE(linReg.train(epochCount, precisionThreshold));
    }

    // Test prediction after the rejected training.
    // Expect the model parameters to be untouched.
    EXPECT_NEAR(linReg.predict(input), zero, Test::ExactTolerance);
}

/**
 * @brief Verify that train() returns false for a precision threshold of 1.0 or more, which can
 *        never be reached, leaving the model parameters untouched.
 */
TEST(LinRegAdaptive, PrecisionThresholdAtOrAboveOne)
{
    constexpr std::size_t epochCount{10U};
    constexpr double precisionThresholds[]{1.0, 1.5};
    constexpr double input{1.0};
    constexpr double zero{0.0};

    const Matrix trainInput{1.0};
    const Matrix trainOutput{5.0};
    LinReg linReg{trainInput, trainOutput};

    // Test precision thresholds at and above the upper bound.
    // Expect training to fail.
    for (const auto precisionThreshold : precisionThresholds)
    {
        EXPECT_FALSE(linReg.train(epochCount, precisionThreshold));
    }

    // Test prediction after the rejected training.
    // Expect the model parameters to be untouched.
    EXPECT_NEAR(linReg.predict(input), zero, Test::ExactTolerance);
}

/**
 * @brief Verify that train() returns true for a valid epoch count and precision threshold, and
 *        that the learning rate is no longer part of the interface: the model picks it itself.
 */
TEST(LinRegAdaptive, ValidArguments)
{
    constexpr std::size_t singleEpoch{1U};
    constexpr std::size_t manyEpochs{1000U};
    constexpr double lowPrecisionThreshold{0.001};
    constexpr double highPrecisionThreshold{0.999999};

    const Matrix trainInput{1.0};
    const Matrix trainOutput{5.0};
    LinReg linReg{trainInput, trainOutput};

    // Test the smallest valid epoch count, and thresholds just inside each of the two bounds.
    // Expect training to succeed in every case.
    EXPECT_TRUE(linReg.train(singleEpoch));
    EXPECT_TRUE(linReg.train(manyEpochs, lowPrecisionThreshold));
    EXPECT_TRUE(linReg.train(manyEpochs, highPrecisionThreshold));
}

/**
 * @brief Verify that the first epoch runs with the initial learning rate of 0.1, i.e. that a
 *        single epoch on the training set (x = 1, y_ref = 5) adjusts both parameters to 0.5.
 */
TEST(LinRegAdaptive, SingleEpoch)
{
    constexpr std::size_t epochCount{1U};
    constexpr double biasInput{0.0};
    constexpr double input{1.0};
    constexpr double expectedBias{0.5};
    constexpr double expectedPrediction{1.0};

    const Matrix trainInput{1.0};
    const Matrix trainOutput{5.0};
    LinReg linReg{trainInput, trainOutput};

    EXPECT_TRUE(linReg.train(epochCount));

    // Test prediction with input 0, which reads back the bias on its own, since y = k * 0 + m.
    // Expect the bias to have been adjusted by error * 0.1 = 5 * 0.1 = 0.5.
    EXPECT_NEAR(linReg.predict(biasInput), expectedBias, Test::ExactTolerance);

    // Test prediction with input 1.
    // Expect weight * 1 + bias = 0.5 + 0.5 = 1.0.
    EXPECT_NEAR(linReg.predict(input), expectedPrediction, Test::ExactTolerance);
}

/**
 * @brief Verify that the second epoch continues from the parameters left behind by the first, and
 *        still runs with the initial learning rate, since the rate is only revised every tenth
 *        epoch.
 */
TEST(LinRegAdaptive, TwoEpochs)
{
    constexpr std::size_t epochCount{2U};
    constexpr double biasInput{0.0};
    constexpr double input{1.0};
    constexpr double expectedBias{0.9};
    constexpr double expectedPrediction{1.8};

    const Matrix trainInput{1.0};
    const Matrix trainOutput{5.0};
    LinReg linReg{trainInput, trainOutput};

    EXPECT_TRUE(linReg.train(epochCount));

    // Test prediction with input 0, which reads back the bias on its own.
    // Expect the bias to have been adjusted to 0.9.
    EXPECT_NEAR(linReg.predict(biasInput), expectedBias, Test::ExactTolerance);

    // Test prediction with input 1.
    // Expect weight * 1 + bias = 0.9 + 0.9 = 1.8.
    EXPECT_NEAR(linReg.predict(input), expectedPrediction, Test::ExactTolerance);
}

/**
 * @brief Verify that training one epoch twice gives the same result as training two epochs in a
 *        single call, i.e. that train() never starts the model over.
 */
TEST(LinRegAdaptive, RepeatedTrainCalls)
{
    constexpr std::size_t singleEpoch{1U};
    constexpr std::size_t twoEpochs{2U};
    constexpr double inputs[]{0.0, 2.0};

    const Matrix trainInput{1.0};
    const Matrix trainOutput{5.0};
    LinReg twoCalls{trainInput, trainOutput};
    LinReg oneCall{trainInput, trainOutput};

    // Test training one epoch twice, against training two epochs in one call.
    // Expect all three training runs to succeed.
    EXPECT_TRUE(twoCalls.train(singleEpoch));
    EXPECT_TRUE(twoCalls.train(singleEpoch));
    EXPECT_TRUE(oneCall.train(twoEpochs));

    // Test prediction with both models.
    // Expect identical predictions, since neither call started training over.
    for (const auto input : inputs)
    {
        EXPECT_NEAR(twoCalls.predict(input), oneCall.predict(input), Test::ExactTolerance);
    }
}

/**
 * @brief Verify that an input of zero assigns the bias directly and leaves the weight untouched,
 *        so that every prediction equals the reference value.
 */
TEST(LinRegAdaptive, ZeroInput)
{
    constexpr std::size_t epochCount{1U};
    constexpr double referenceOutput{7.0};
    constexpr double inputs[]{0.0, 3.0, -3.0};

    const Matrix trainInput{0.0};
    const Matrix trainOutput{referenceOutput};
    LinReg linReg{trainInput, trainOutput};

    EXPECT_TRUE(linReg.train(epochCount));

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
TEST(LinRegAdaptive, MismatchedTrainingData)
{
    constexpr std::size_t epochCount{1U};
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

    EXPECT_TRUE(moreInputs.train(epochCount));
    EXPECT_TRUE(moreOutputs.train(epochCount));

    // Test prediction with both models.
    // Expect the result of training the single complete set (x = 1, y_ref = 5), i.e. the
    // surplus values in the longer vector to have been ignored.
    EXPECT_NEAR(moreInputs.predict(biasInput), expectedBias, Test::ExactTolerance);
    EXPECT_NEAR(moreInputs.predict(input), expectedPrediction, Test::ExactTolerance);
    EXPECT_NEAR(moreOutputs.predict(biasInput), expectedBias, Test::ExactTolerance);
    EXPECT_NEAR(moreOutputs.predict(input), expectedPrediction, Test::ExactTolerance);
}

/**
 * @brief Verify that the model reproduces the training data it was trained on, here the line
 *        y = 2x + 2, without ever being given a learning rate.
 */
TEST(LinRegAdaptiveComponent, HappyPath)
{
    constexpr std::size_t setCount{5U};
    constexpr double bias{2.0};
    constexpr double weight{2.0};
    constexpr double step{1.0};
    constexpr std::size_t epochCount{1000U};

    Matrix trainInput(setCount);
    Matrix trainOutput(setCount);
    createTrainingData(trainInput, trainOutput, bias, weight, step);

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
TEST(LinRegAdaptiveComponent, Generalization)
{
    constexpr std::size_t setCount{5U};
    constexpr double bias{2.0};
    constexpr double weight{3.0};
    constexpr double step{1.0};
    constexpr std::size_t epochCount{1000U};

    Matrix trainInput(setCount);
    Matrix trainOutput(setCount);
    createTrainingData(trainInput, trainOutput, bias, weight, step);

    LinReg linReg{trainInput, trainOutput};

    EXPECT_TRUE(linReg.train(epochCount));

    constexpr double startVal{-5.0};
    constexpr double endVal{5.0};
    constexpr double inputStep{1.0};

    // Test prediction across the range [-5, 5], most of which was never trained on.
    // Expect the model to reproduce the line for every input.
    for (double input{startVal}; input <= endVal; input += inputStep)
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
TEST(LinRegAdaptiveComponent, NegativeParameters)
{
    constexpr std::size_t setCount{5U};
    constexpr double bias{1.0};
    constexpr double weight{-2.0};
    constexpr double step{1.0};
    constexpr std::size_t epochCount{1000U};

    Matrix trainInput(setCount);
    Matrix trainOutput(setCount);
    createTrainingData(trainInput, trainOutput, bias, weight, step);

    LinReg linReg{trainInput, trainOutput};

    EXPECT_TRUE(linReg.train(epochCount));

    constexpr double startVal{-5.0};
    constexpr double endVal{5.0};
    constexpr double inputStep{1.0};

    // Test prediction across the range [-5, 5].
    // Expect the model to reproduce the line, i.e. to have driven the weight negative rather
    // than away from the reference values.
    for (double input{startVal}; input <= endVal; input += inputStep)
    {
        const auto prediction = linReg.predict(input);
        EXPECT_TRUE(std::isfinite(prediction));
        EXPECT_NEAR(prediction, weight * input + bias, Test::ConvergenceTolerance);
    }
}

/**
 * @brief Verify that the training order is randomized before every epoch, by training several
 *        models on identical data for a single epoch. A different order leaves different
 *        parameters behind, so the models mustn't all agree.
 */
TEST(LinRegAdaptiveComponent, TrainingOrderIsRandomized)
{
    constexpr std::size_t modelCount{10U};
    constexpr std::size_t epochCount{1U};
    constexpr double input{1.0};

    // The training inputs start at 1 rather than 0 on purpose: an input of zero assigns the bias
    // directly, which erases what the order did to the sets processed before it.
    const Matrix trainInput{1.0, 2.0, 3.0, 4.0, 5.0};
    const Matrix trainOutput{4.0, 6.0, 8.0, 10.0, 12.0};

    double firstPrediction{};
    bool allPredictionsEqual{true};

    for (std::size_t i{}; i < modelCount; ++i)
    {
        LinReg linReg{trainInput, trainOutput};
        EXPECT_TRUE(linReg.train(epochCount));

        const auto prediction = linReg.predict(input);
        if (0U == i) { firstPrediction = prediction; }
        else if (prediction != firstPrediction) { allPredictionsEqual = false; }
    }

    // Test the predictions left behind by ten identically trained models.
    // Expect them not all to agree, since each epoch shuffles the training order first.
    EXPECT_FALSE(allPredictionsEqual);
}

/**
 * @brief Verify that training stops once the precision reaches the threshold, i.e. that the mean
 *        absolute error left behind is within 1.0 - precisionThreshold, since the precision is
 *        defined as 1.0 minus the mean absolute error.
 *
 * @note The training data uses inputs a tenth of the usual size, which makes every weight update
 *       a tenth as large. The model then needs enough epochs for the threshold to be what stops
 *       training, rather than converging past every threshold on the first evaluation.
 */
TEST(LinRegAdaptiveComponent, EarlyStopHonoursPrecisionThreshold)
{
    constexpr std::size_t setCount{5U};
    constexpr double bias{2.0};
    constexpr double weight{2.0};
    constexpr double step{0.1};
    constexpr std::size_t epochCount{2000U};
    constexpr double precisionThresholds[]{0.5, 0.9, 0.99};

    Matrix trainInput(setCount);
    Matrix trainOutput(setCount);
    createTrainingData(trainInput, trainOutput, bias, weight, step);

    // Test training with several precision thresholds.
    // Expect the error left behind to be within the error the threshold allows.
    for (const auto threshold : precisionThresholds)
    {
        LinReg linReg{trainInput, trainOutput};
        EXPECT_TRUE(linReg.train(epochCount, threshold));
        EXPECT_TRUE(meanAbsoluteError(linReg, trainInput, trainOutput) <= 1.0 - threshold);
    }
}

/**
 * @brief Verify that a low precision threshold stops training earlier than a high one, leaving a
 *        measurably less accurate model behind.
 */
TEST(LinRegAdaptiveComponent, LowPrecisionThresholdStopsEarlier)
{
    constexpr std::size_t setCount{5U};
    constexpr double bias{2.0};
    constexpr double weight{2.0};
    constexpr double step{0.1};
    constexpr std::size_t epochCount{2000U};
    constexpr double lowPrecisionThreshold{0.5};
    constexpr double highPrecisionThreshold{0.999999};

    Matrix trainInput(setCount);
    Matrix trainOutput(setCount);
    createTrainingData(trainInput, trainOutput, bias, weight, step);

    LinReg stopsEarly{trainInput, trainOutput};
    LinReg stopsLate{trainInput, trainOutput};

    EXPECT_TRUE(stopsEarly.train(epochCount, lowPrecisionThreshold));
    EXPECT_TRUE(stopsLate.train(epochCount, highPrecisionThreshold));

    // Test the error left behind by both models.
    // Expect the model that stopped at the low threshold to be the less accurate of the two.
    EXPECT_TRUE(meanAbsoluteError(stopsEarly, trainInput, trainOutput) >
                meanAbsoluteError(stopsLate, trainInput, trainOutput));
}

/**
 * @brief Verify that the learning rate is raised while progress is slow, by training an adaptive
 *        model and a fixed one on identical data for the same number of epochs. The fixed model
 *        is given the learning rate the adaptive one starts with, so a tie would mean the rate
 *        never moved.
 *
 * @note The comparison needs training data the initial learning rate is too cautious for, which
 *       is what the inputs of a tenth the usual size give: the adaptive model raises its rate
 *       every tenth epoch while the improvement stays below the expected minimum, and pulls ahead.
 */
TEST(LinRegAdaptiveComponent, LearningRateRisesWhileProgressIsSlow)
{
    constexpr std::size_t setCount{5U};
    constexpr double bias{2.0};
    constexpr double weight{2.0};
    constexpr double step{0.1};
    constexpr std::size_t epochCount{200U};

    Matrix trainInput(setCount);
    Matrix trainOutput(setCount);
    createTrainingData(trainInput, trainOutput, bias, weight, step);

    LinReg adaptive{trainInput, trainOutput};
    LinRegFixed fixed{trainInput, trainOutput};

    EXPECT_TRUE(adaptive.train(epochCount));
    EXPECT_TRUE(fixed.train(epochCount, Test::InitialLearningRate));

    // Test the error left behind by both models.
    // Expect the adaptive model to be the more accurate of the two, since it spent most of the
    // run at a higher learning rate than the one it started with.
    EXPECT_TRUE(meanAbsoluteError(adaptive, trainInput, trainOutput) <
                meanAbsoluteError(fixed, trainInput, trainOutput));
}

/**
 * @brief Verify that prediction through ml::lin_reg::Interface dispatches to the concrete class,
 *        which is what lets main.cpp predict without knowing which model it was handed.
 */
TEST(LinRegAdaptiveComponent, PredictionThroughInterface)
{
    constexpr std::size_t setCount{5U};
    constexpr double bias{2.0};
    constexpr double weight{2.0};
    constexpr double step{1.0};
    constexpr std::size_t epochCount{1000U};

    Matrix trainInput(setCount);
    Matrix trainOutput(setCount);
    createTrainingData(trainInput, trainOutput, bias, weight, step);

    LinReg linReg{trainInput, trainOutput};

    EXPECT_TRUE(linReg.train(epochCount));

    const LinRegInterface& linRegInterface{linReg};

    constexpr double startVal{-5.0};
    constexpr double endVal{5.0};
    constexpr double inputStep{1.0};

    // Test prediction through the interface against prediction through the concrete class.
    // Expect identical results, since the interface call dispatches to Adaptive::predict().
    for (double input{startVal}; input <= endVal; input += inputStep)
    {
        EXPECT_EQ(linRegInterface.predict(input), linReg.predict(input));
    }
}
} // namespace
