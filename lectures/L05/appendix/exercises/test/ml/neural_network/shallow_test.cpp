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

    /** A learning rate inside the valid range. */
    static constexpr double LearningRate{0.1};

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

    for (const auto epochCount : epochCounts)
    {
        DenseLayer hiddenLayer{Test::HiddenCount, Test::InputCount};
        DenseLayer outputLayer{Test::OutputCount, Test::HiddenCount};
        NeuralNetwork network{hiddenLayer, outputLayer, TrainInput, TrainOutput};

        EXPECT_TRUE(network.train(epochCount, Test::LearningRate));
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

    EXPECT_FALSE(network.train(zeroEpochs, Test::LearningRate));

    // Test the counts after the rejected call.
    // Expect them untouched: a rejected training run mustn't feed anything forward.
    EXPECT_EQ(hiddenLayer.feedforwardCount(), std::size_t{});
    EXPECT_EQ(outputLayer.feedforwardCount(), std::size_t{});
}

/**
 * @brief Verify that training fails for learning rates outside the range (0.0, 1.0).
 */
TEST(NeuralNetworkShallow, TrainRejectsInvalidLearningRate)
{
    constexpr std::size_t epochCount{5U};
    constexpr double invalidLearningRates[]{0.0, -0.1, -1.0, 1.0, 1.5, 100.0};

    for (const auto learningRate : invalidLearningRates)
    {
        DenseLayer hiddenLayer{Test::HiddenCount, Test::InputCount};
        DenseLayer outputLayer{Test::OutputCount, Test::HiddenCount};
        NeuralNetwork network{hiddenLayer, outputLayer, TrainInput, TrainOutput};

        EXPECT_FALSE(network.train(epochCount, learningRate));
        EXPECT_EQ(hiddenLayer.feedforwardCount(), std::size_t{});
    }
}

/**
 * @brief Verify that training succeeds for learning rates just inside the range (0.0, 1.0).
 */
TEST(NeuralNetworkShallow, TrainAcceptsValidLearningRate)
{
    constexpr std::size_t epochCount{2U};
    constexpr double validLearningRates[]{0.001, 0.1, 0.5, 0.999};

    for (const auto learningRate : validLearningRates)
    {
        DenseLayer hiddenLayer{Test::HiddenCount, Test::InputCount};
        DenseLayer outputLayer{Test::OutputCount, Test::HiddenCount};
        NeuralNetwork network{hiddenLayer, outputLayer, TrainInput, TrainOutput};

        EXPECT_TRUE(network.train(epochCount, learningRate));
        EXPECT_EQ(hiddenLayer.feedforwardCount(), epochCount * Test::SetCount);
    }
}

/**
 * @brief Verify that the learning rate defaults to a value inside the valid range.
 */
TEST(NeuralNetworkShallow, TrainDefaultLearningRate)
{
    constexpr std::size_t epochCount{3U};

    DenseLayer hiddenLayer{Test::HiddenCount, Test::InputCount};
    DenseLayer outputLayer{Test::OutputCount, Test::HiddenCount};
    NeuralNetwork network{hiddenLayer, outputLayer, TrainInput, TrainOutput};

    EXPECT_TRUE(network.train(epochCount));
    EXPECT_EQ(hiddenLayer.feedforwardCount(), epochCount * Test::SetCount);
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

    EXPECT_TRUE(network.train(epochCount, Test::LearningRate));
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
