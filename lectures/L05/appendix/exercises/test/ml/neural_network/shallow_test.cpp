/**
 * @file Component tests for ml::neural_network::Shallow.
 */
#include <cstddef>

#include "ml/dense_layer/stub.hpp"
#include "ml/neural_network/interface.hpp"
#include "ml/neural_network/shallow.hpp"
#include "ml/types.hpp"
#include "qacademy/test/test.hpp"

using DenseLayer             = ml::dense_layer::Stub;
using NeuralNetwork          = ml::neural_network::Shallow;
using NeuralNetworkInterface = ml::neural_network::Interface;
using Matrix1d               = ml::Matrix1d;
using Matrix2d               = ml::Matrix2d;

namespace
{
/**
 * @brief Test parameters.
 */
struct Test
{
    /** Value tolerance when using EXPECT_NEAR(). */
    static constexpr double Tolerance{1e-9};

    /** Number of network inputs, i.e. weights per hidden node. */
    static constexpr std::size_t InputCount{2U};

    /** Number of hidden nodes. */
    static constexpr std::size_t HiddenCount{3U};

    /** Number of output nodes. */
    static constexpr std::size_t OutputCount{1U};

    /** Number of training sets in the XOR pattern below. */
    static constexpr std::size_t SetCount{4U};

    /** A precision threshold inside the valid range, high enough never to be reached. */
    static constexpr double PrecisionThreshold{0.999999};

    /** Epochs between two precision evaluations, mirroring the network's own interval. */
    static constexpr std::size_t EvaluationInterval{100U};

    /** Output value reported by the hidden layer, distinct from the output layer's. */
    static constexpr double HiddenOutputValue{0.25};

    /** Output value reported by the output layer, distinct from the hidden layer's. */
    static constexpr double OutputOutputValue{0.75};
};

/** Training data inputs: the 2-bit XOR pattern. */
const Matrix2d TrainInput{{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};

/** Training data outputs: the 2-bit XOR pattern. */
const Matrix2d TrainOutput{{0.0}, {1.0}, {1.0}, {0.0}};

/**
 * Training data outputs a network of stubs predicts exactly, whatever it is fed: every reference
 * value is the output layer's fixed output, so the measured precision is 1.0 from the first epoch.
 */
const Matrix2d ExactTrainOutput{{Test::OutputOutputValue},
                                {Test::OutputOutputValue},
                                {Test::OutputOutputValue},
                                {Test::OutputOutputValue}};

/**
 * @brief Verify that the network hands back the output layer's output, not the hidden layer's.
 */
TEST(NeuralNetworkShallow, PredictReturnsOutputLayerOutput)
{
    DenseLayer hiddenLayer{Test::HiddenCount, Test::InputCount, Test::HiddenOutputValue};
    DenseLayer outputLayer{Test::OutputCount, Test::HiddenCount, Test::OutputOutputValue};
    NeuralNetwork network{hiddenLayer, outputLayer, TrainInput, TrainOutput};

    const auto& prediction = network.predict(TrainInput[0U]);

    // Test the shape and the value of the prediction.
    // Expect one value per output node, each reporting the output layer's value.
    EXPECT_EQ(prediction.size(), Test::OutputCount);

    for (const auto value : prediction)
    {
        EXPECT_NEAR(value, Test::OutputOutputValue, Test::Tolerance);
    }
}

/**
 * @brief Verify that the prediction is read from the output layer on every call rather than
 *        copied once, by changing the layer's output behind the network's back.
 */
TEST(NeuralNetworkShallow, PredictReadsOutputLayerLive)
{
    constexpr double newOutputValue{-1.5};

    DenseLayer hiddenLayer{Test::HiddenCount, Test::InputCount, Test::HiddenOutputValue};
    DenseLayer outputLayer{Test::OutputCount, Test::HiddenCount, Test::OutputOutputValue};
    NeuralNetwork network{hiddenLayer, outputLayer, TrainInput, TrainOutput};

    EXPECT_NEAR(network.predict(TrainInput[0U])[0U], Test::OutputOutputValue, Test::Tolerance);

    // Test a prediction after the output layer was changed through the reference the network
    // holds. Expect the new value: a network that cached the output would still report the old.
    outputLayer.setOutput(newOutputValue);
    EXPECT_NEAR(network.predict(TrainInput[0U])[0U], newOutputValue, Test::Tolerance);
}

/**
 * @brief Verify that the prediction refers to the output layer's own vector rather than a copy.
 */
TEST(NeuralNetworkShallow, PredictReturnsReferenceToOutputLayer)
{
    DenseLayer hiddenLayer{Test::HiddenCount, Test::InputCount};
    DenseLayer outputLayer{Test::OutputCount, Test::HiddenCount};
    NeuralNetwork network{hiddenLayer, outputLayer, TrainInput, TrainOutput};

    EXPECT_TRUE(&network.predict(TrainInput[0U]) == &outputLayer.output());
}

/**
 * @brief Verify that a prediction feeds both layers exactly once, in one pass through the network.
 */
TEST(NeuralNetworkShallow, PredictFeedsBothLayersOnce)
{
    constexpr std::size_t one{1U};
    constexpr std::size_t predictionCount{3U};

    DenseLayer hiddenLayer{Test::HiddenCount, Test::InputCount};
    DenseLayer outputLayer{Test::OutputCount, Test::HiddenCount};
    NeuralNetwork network{hiddenLayer, outputLayer, TrainInput, TrainOutput};

    // Test the counts after construction.
    // Expect both at zero: building a network mustn't feed anything forward.
    EXPECT_EQ(hiddenLayer.feedforwardCount(), std::size_t{});
    EXPECT_EQ(outputLayer.feedforwardCount(), std::size_t{});

    network.predict(TrainInput[0U]);
    EXPECT_EQ(hiddenLayer.feedforwardCount(), one);
    EXPECT_EQ(outputLayer.feedforwardCount(), one);

    // Test the counts after several further predictions.
    // Expect one pass through each layer per call, and never more than one.
    for (std::size_t i{}; i < predictionCount; ++i)
    {
        network.predict(TrainInput[i]);
    }
    EXPECT_EQ(hiddenLayer.feedforwardCount(), one + predictionCount);
    EXPECT_EQ(outputLayer.feedforwardCount(), one + predictionCount);
}

/**
 * @brief Verify that predictions work on an untrained network.
 */
TEST(NeuralNetworkShallow, PredictBeforeTraining)
{
    DenseLayer hiddenLayer{Test::HiddenCount, Test::InputCount, Test::HiddenOutputValue};
    DenseLayer outputLayer{Test::OutputCount, Test::HiddenCount, Test::OutputOutputValue};
    NeuralNetwork network{hiddenLayer, outputLayer, TrainInput, TrainOutput};

    for (const auto& input : TrainInput)
    {
        EXPECT_NEAR(network.predict(input)[0U], Test::OutputOutputValue, Test::Tolerance);
    }
}

/**
 * @brief Verify that training performs one feedforward per training set per epoch.
 */
TEST(NeuralNetworkShallow, TrainFeedsEveryTrainingSetEveryEpoch)
{
    constexpr std::size_t epochCounts[]{1U, 2U, 10U};

    // Every count stays below the evaluation interval, so no precision pass is added to the tally.
    for (const auto epochCount : epochCounts)
    {
        DenseLayer hiddenLayer{Test::HiddenCount, Test::InputCount};
        DenseLayer outputLayer{Test::OutputCount, Test::HiddenCount};
        NeuralNetwork network{hiddenLayer, outputLayer, TrainInput, TrainOutput};

        EXPECT_TRUE(network.train(epochCount, Test::PrecisionThreshold));
        EXPECT_EQ(hiddenLayer.feedforwardCount(), epochCount * Test::SetCount);
        EXPECT_EQ(outputLayer.feedforwardCount(), epochCount * Test::SetCount);
    }
}

/**
 * @brief Verify that training with an epoch count of zero fails without touching the layers.
 */
TEST(NeuralNetworkShallow, TrainRejectsZeroEpochCount)
{
    constexpr std::size_t zeroEpochs{0U};

    DenseLayer hiddenLayer{Test::HiddenCount, Test::InputCount};
    DenseLayer outputLayer{Test::OutputCount, Test::HiddenCount};
    NeuralNetwork network{hiddenLayer, outputLayer, TrainInput, TrainOutput};

    EXPECT_FALSE(network.train(zeroEpochs, Test::PrecisionThreshold));

    // Test the counts after the rejected call.
    // Expect them untouched: a rejected training run mustn't feed anything forward.
    EXPECT_EQ(hiddenLayer.feedforwardCount(), std::size_t{});
    EXPECT_EQ(outputLayer.feedforwardCount(), std::size_t{});
}

/**
 * @brief Verify that training fails for precision thresholds outside the range (0.0, 1.0).
 */
TEST(NeuralNetworkShallow, TrainRejectsInvalidPrecisionThreshold)
{
    constexpr std::size_t epochCount{5U};
    constexpr double invalidThresholds[]{0.0, -0.1, -1.0, 1.0, 1.5, 100.0};

    for (const auto precisionThreshold : invalidThresholds)
    {
        DenseLayer hiddenLayer{Test::HiddenCount, Test::InputCount};
        DenseLayer outputLayer{Test::OutputCount, Test::HiddenCount};
        NeuralNetwork network{hiddenLayer, outputLayer, TrainInput, TrainOutput};

        EXPECT_FALSE(network.train(epochCount, precisionThreshold));
        EXPECT_EQ(hiddenLayer.feedforwardCount(), std::size_t{});
    }
}

/**
 * @brief Verify that training succeeds for precision thresholds inside the range (0.0, 1.0).
 */
TEST(NeuralNetworkShallow, TrainAcceptsValidPrecisionThreshold)
{
    constexpr std::size_t epochCount{2U};
    constexpr double validThresholds[]{0.001, 0.1, 0.5, 0.999999};

    // The epoch count stays below the evaluation interval, so no threshold is reached early here,
    // however low it is set: the network trains for every epoch it was given.
    for (const auto precisionThreshold : validThresholds)
    {
        DenseLayer hiddenLayer{Test::HiddenCount, Test::InputCount};
        DenseLayer outputLayer{Test::OutputCount, Test::HiddenCount};
        NeuralNetwork network{hiddenLayer, outputLayer, TrainInput, TrainOutput};

        EXPECT_TRUE(network.train(epochCount, precisionThreshold));
        EXPECT_EQ(hiddenLayer.feedforwardCount(), epochCount * Test::SetCount);
    }
}

/**
 * @brief Verify that the precision threshold defaults to a value inside the valid range.
 */
TEST(NeuralNetworkShallow, TrainDefaultPrecisionThreshold)
{
    constexpr std::size_t epochCount{3U};

    DenseLayer hiddenLayer{Test::HiddenCount, Test::InputCount};
    DenseLayer outputLayer{Test::OutputCount, Test::HiddenCount};
    NeuralNetwork network{hiddenLayer, outputLayer, TrainInput, TrainOutput};

    EXPECT_TRUE(network.train(epochCount));
    EXPECT_EQ(hiddenLayer.feedforwardCount(), epochCount * Test::SetCount);
}

/**
 * @brief Verify that the precision is evaluated once every hundredth epoch, and never on the
 *        first, by counting the extra pass each evaluation makes over the training data.
 */
TEST(NeuralNetworkShallow, TrainEvaluatesPrecisionEveryHundredthEpoch)
{
    constexpr std::size_t evaluationCounts[]{0U, 1U, 2U};

    // A stub reports the same output whatever it is fed, so the precision never reaches the
    // threshold and every run trains for the full epoch count.
    for (const auto evaluationCount : evaluationCounts)
    {
        const auto epochCount = (evaluationCount * Test::EvaluationInterval) + 1U;

        DenseLayer hiddenLayer{Test::HiddenCount, Test::InputCount};
        DenseLayer outputLayer{Test::OutputCount, Test::HiddenCount};
        NeuralNetwork network{hiddenLayer, outputLayer, TrainInput, TrainOutput};

        EXPECT_TRUE(network.train(epochCount, Test::PrecisionThreshold));

        // Test the tally against the training passes plus one pass per evaluation.
        // Expect no evaluation at all for a single epoch: the first epoch is never evaluated.
        const auto expectedCount = (epochCount + evaluationCount) * Test::SetCount;
        EXPECT_EQ(hiddenLayer.feedforwardCount(), expectedCount);
        EXPECT_EQ(outputLayer.feedforwardCount(), expectedCount);
    }
}

/**
 * @brief Verify that training stops as soon as the measured precision reaches the threshold,
 *        rather than running out the epoch count.
 */
TEST(NeuralNetworkShallow, TrainStopsWhenPrecisionThresholdIsReached)
{
    constexpr std::size_t epochCount{5U * Test::EvaluationInterval};
    constexpr std::size_t evaluationEpoch{Test::EvaluationInterval + 1U};

    DenseLayer hiddenLayer{Test::HiddenCount, Test::InputCount, Test::HiddenOutputValue};
    DenseLayer outputLayer{Test::OutputCount, Test::HiddenCount, Test::OutputOutputValue};
    NeuralNetwork network{hiddenLayer, outputLayer, TrainInput, ExactTrainOutput};

    EXPECT_TRUE(network.train(epochCount, Test::PrecisionThreshold));

    // Test the tally against the epochs run up to the first evaluation, plus that evaluation's own
    // pass. Expect training to have stopped there: every reference value is already predicted
    // exactly, so the precision is 1.0 and nothing is left to learn.
    const auto expectedCount = (evaluationEpoch + 1U) * Test::SetCount;
    EXPECT_EQ(hiddenLayer.feedforwardCount(), expectedCount);
    EXPECT_EQ(outputLayer.feedforwardCount(), expectedCount);
    EXPECT_TRUE(hiddenLayer.feedforwardCount() < epochCount * Test::SetCount);
}

/**
 * @brief Verify that predictions still work once training has run.
 */
TEST(NeuralNetworkShallow, PredictAfterTraining)
{
    constexpr std::size_t epochCount{4U};

    DenseLayer hiddenLayer{Test::HiddenCount, Test::InputCount, Test::HiddenOutputValue};
    DenseLayer outputLayer{Test::OutputCount, Test::HiddenCount, Test::OutputOutputValue};
    NeuralNetwork network{hiddenLayer, outputLayer, TrainInput, TrainOutput};

    EXPECT_TRUE(network.train(epochCount, Test::PrecisionThreshold));
    hiddenLayer.clearFeedforwardCount();
    outputLayer.clearFeedforwardCount();

    // Test a prediction made after training.
    // Expect it to behave exactly as before training, feeding each layer once.
    EXPECT_NEAR(network.predict(TrainInput[0U])[0U], Test::OutputOutputValue, Test::Tolerance);
    EXPECT_EQ(hiddenLayer.feedforwardCount(), std::size_t{1U});
    EXPECT_EQ(outputLayer.feedforwardCount(), std::size_t{1U});
}

/**
 * @brief Verify that the network is usable through ml::neural_network::Interface.
 */
TEST(NeuralNetworkShallow, UsableThroughInterface)
{
    DenseLayer hiddenLayer{Test::HiddenCount, Test::InputCount, Test::HiddenOutputValue};
    DenseLayer outputLayer{Test::OutputCount, Test::HiddenCount, Test::OutputOutputValue};
    NeuralNetwork network{hiddenLayer, outputLayer, TrainInput, TrainOutput};
    NeuralNetworkInterface& networkInterface{network};

    const auto& prediction = networkInterface.predict(TrainInput[0U]);
    EXPECT_EQ(prediction.size(), Test::OutputCount);
    EXPECT_NEAR(prediction[0U], Test::OutputOutputValue, Test::Tolerance);
    EXPECT_EQ(hiddenLayer.feedforwardCount(), std::size_t{1U});
}
} // namespace
