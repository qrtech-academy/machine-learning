/**
 * @file Unit and component tests for ml::dense_layer::Dense.
 */
#include <cmath>
#include <cstddef>

#include "ml/dense_layer/dense.hpp"
#include "ml/dense_layer/interface.hpp"
#include "ml/neural_network/shallow.hpp"
#include "ml/types.hpp"
#include "ml/utils.hpp"
#include "qacademy/test/test.hpp"

using ActFunc             = ml::ActFunc;
using DenseLayer          = ml::dense_layer::Dense;
using DenseLayerInterface = ml::dense_layer::Interface;
using NeuralNetwork       = ml::neural_network::Shallow;
using Matrix1d            = ml::Matrix1d;
using Matrix2d            = ml::Matrix2d;

namespace
{
/**
 * @brief Test parameters.
 */
struct Test
{
    /** Value tolerance for hand-computed results when using EXPECT_NEAR(). */
    static constexpr double ExactTolerance{1e-9};

    /** Node count used by most tests. */
    static constexpr std::size_t NodeCount{3U};

    /** Weight count per node used by most tests. Deliberately unequal to NodeCount. */
    static constexpr std::size_t WeightCount{2U};

    /** A learning rate inside the valid range. */
    static constexpr double LearningRate{0.1};

    /** Hidden nodes used by the convergence test. */
    static constexpr std::size_t HiddenCount{8U};

    /** Epochs used by the convergence test. */
    static constexpr std::size_t EpochCount{10000U};

    /** Precision the convergence test trains towards, i.e. a mean absolute error of 0.01. */
    static constexpr double PrecisionThreshold{0.99};

    /** Mean absolute error the convergence test must reach. */
    static constexpr double ConvergenceThreshold{0.05};

    /** Training runs the convergence test is allowed, each starting from fresh parameters. */
    static constexpr std::size_t TrainAttempts{3U};
};

// -----------------------------------------------------------------------------
Matrix1d recoverBias(DenseLayer& denseLayer) noexcept
{
    const Matrix1d zeroInput(denseLayer.weightCount(), 0.0);
    denseLayer.feedforward(zeroInput);
    return denseLayer.output();
}

// -----------------------------------------------------------------------------
double weightedSum(const Matrix2d& weights, const Matrix1d& bias, const Matrix1d& input,
                   const std::size_t node) noexcept
{
    auto sum = bias[node];

    for (std::size_t i{}; i < input.size(); ++i)
    {
        sum += weights[node][i] * input[i];
    }
    return sum;
}

// -----------------------------------------------------------------------------
double meanAbsoluteError(NeuralNetwork& network, const Matrix2d& input,
                         const Matrix2d& reference) noexcept
{
    double sum{};

    for (std::size_t i{}; i < input.size(); ++i)
    {
        sum += std::fabs(reference[i][0U] - network.predict(input[i])[0U]);
    }
    return sum / static_cast<double>(input.size());
}

/**
 * @brief Verify that the layer reports the node and weight counts it was constructed with.
 */
TEST(DenseLayerDense, ConstructedDimensions)
{
    const DenseLayer denseLayer{Test::NodeCount, Test::WeightCount};
    EXPECT_EQ(denseLayer.nodeCount(), Test::NodeCount);
    EXPECT_EQ(denseLayer.weightCount(), Test::WeightCount);
}

/**
 * @brief Verify that the output, error, and weight matrices are built to the right shape.
 */
TEST(DenseLayerDense, ConstructedMatrixShapes)
{
    const DenseLayer denseLayer{Test::NodeCount, Test::WeightCount};
    EXPECT_EQ(denseLayer.output().size(), Test::NodeCount);
    EXPECT_EQ(denseLayer.error().size(), Test::NodeCount);
    EXPECT_EQ(denseLayer.weights().size(), Test::NodeCount);

    for (const auto& nodeWeights : denseLayer.weights())
    {
        EXPECT_EQ(nodeWeights.size(), Test::WeightCount);
    }
}

/**
 * @brief Verify that bias and weights start randomized inside the range [0.0, 1.0].
 */
TEST(DenseLayerDense, ConstructedParametersAreRandomized)
{
    DenseLayer denseLayer{Test::NodeCount, Test::WeightCount, ActFunc::None};
    const auto bias = recoverBias(denseLayer);
    bool biasVaries{false};

    for (const auto value : bias)
    {
        EXPECT_TRUE(std::isfinite(value));
        EXPECT_TRUE((0.0 <= value) && (1.0 >= value));
        biasVaries = biasVaries || (value != bias[0U]);
    }

    // Test that the bias values aren't all the same number.
    // Expect variation: a bias vector left at its default is all zeros, which passes the range
    // check above while quietly dropping one of the layer's two trainable parameters.
    EXPECT_TRUE(biasVaries);

    bool weightsVary{false};

    for (const auto& nodeWeights : denseLayer.weights())
    {
        for (const auto weight : nodeWeights)
        {
            EXPECT_TRUE(std::isfinite(weight));
            EXPECT_TRUE((0.0 <= weight) && (1.0 >= weight));
            weightsVary = weightsVary || (weight != denseLayer.weights()[0U][0U]);
        }
    }

    // Test that the weights aren't all the same number.
    // Expect variation: nodes that start identical compute identical errors and stay identical.
    EXPECT_TRUE(weightsVary);
}

/**
 * @brief Verify that two layers built with the same arguments get different random parameters.
 */
TEST(DenseLayerDense, LayersAreIndependentlyRandomized)
{
    const DenseLayer first{Test::NodeCount, Test::WeightCount};
    const DenseLayer second{Test::NodeCount, Test::WeightCount};
    bool differs{false};

    for (std::size_t i{}; (i < Test::NodeCount) && !differs; ++i)
    {
        for (std::size_t j{}; (j < Test::WeightCount) && !differs; ++j)
        {
            differs = (first.weights()[i][j] != second.weights()[i][j]);
        }
    }
    EXPECT_TRUE(differs);
}

/**
 * @brief Verify that feedforward computes the bias plus the weighted sum of the input.
 */
TEST(DenseLayerDense, FeedforwardComputesWeightedSum)
{
    const Matrix1d input{2.0, -3.0};

    DenseLayer denseLayer{Test::NodeCount, Test::WeightCount, ActFunc::None};
    const auto bias = recoverBias(denseLayer);

    EXPECT_TRUE(denseLayer.feedforward(input));

    // Test each node's output against a sum computed by hand from weights() and the bias.
    // Expect an exact match: with ActFunc::None the activation function is the identity.
    for (std::size_t i{}; i < Test::NodeCount; ++i)
    {
        const auto expected = weightedSum(denseLayer.weights(), bias, input, i);
        EXPECT_TRUE(std::isfinite(denseLayer.output()[i]));
        EXPECT_NEAR(denseLayer.output()[i], expected, Test::ExactTolerance);
    }
}

/**
 * @brief Verify that feedforward applies the layer's activation function to the weighted sum.
 */
TEST(DenseLayerDense, FeedforwardAppliesActivationFunction)
{
    // Large negative inputs against non-negative weights drive every sum below zero, so this
    // exercises ReLU's clamp rather than just its pass-through branch.
    const Matrix1d negativeInput{-50.0, -50.0};
    const Matrix1d positiveInput{2.0, 3.0};

    DenseLayer denseLayer{Test::NodeCount, Test::WeightCount, ActFunc::Relu};

    // ReLU passes non-negative values through unchanged, and randomStartVal() never returns a
    // negative number, so a zero input recovers the bias here exactly as ActFunc::None would.
    const auto bias = recoverBias(denseLayer);

    for (const auto& input : {negativeInput, positiveInput})
    {
        EXPECT_TRUE(denseLayer.feedforward(input));

        for (std::size_t i{}; i < Test::NodeCount; ++i)
        {
            const auto sum = weightedSum(denseLayer.weights(), bias, input, i);
            EXPECT_NEAR(denseLayer.output()[i], ml::actFuncOutput(ActFunc::Relu, sum),
                        Test::ExactTolerance);
        }
    }
}

/**
 * @brief Verify that feeding the same input twice produces the same output.
 */
TEST(DenseLayerDense, FeedforwardIsDeterministic)
{
    const Matrix1d input{0.5, -0.25};
    DenseLayer denseLayer{Test::NodeCount, Test::WeightCount, ActFunc::Tanh};

    EXPECT_TRUE(denseLayer.feedforward(input));
    const auto firstOutput = denseLayer.output();

    EXPECT_TRUE(denseLayer.feedforward(input));

    for (std::size_t i{}; i < Test::NodeCount; ++i)
    {
        EXPECT_NEAR(denseLayer.output()[i], firstOutput[i], Test::ExactTolerance);
    }
}

/**
 * @brief Verify that feedforward alone doesn't change any trainable parameter.
 */
TEST(DenseLayerDense, FeedforwardDoesNotTrain)
{
    const Matrix1d input{1.0, 2.0};
    DenseLayer denseLayer{Test::NodeCount, Test::WeightCount, ActFunc::None};

    const auto weightsBefore = denseLayer.weights();
    const auto biasBefore    = recoverBias(denseLayer);

    EXPECT_TRUE(denseLayer.feedforward(input));
    const auto biasAfter = recoverBias(denseLayer);

    for (std::size_t i{}; i < Test::NodeCount; ++i)
    {
        EXPECT_NEAR(biasAfter[i], biasBefore[i], Test::ExactTolerance);

        for (std::size_t j{}; j < Test::WeightCount; ++j)
        {
            EXPECT_NEAR(denseLayer.weights()[i][j], weightsBefore[i][j], Test::ExactTolerance);
        }
    }
}

/**
 * @brief Verify that an output layer computes its error as the reference minus its own output.
 */
TEST(DenseLayerDense, BackpropagateOutputLayerComputesError)
{
    const Matrix1d input{1.5, -0.5};
    const Matrix1d reference{1.0, 0.0, -1.0};

    DenseLayer denseLayer{Test::NodeCount, Test::WeightCount, ActFunc::None};
    EXPECT_TRUE(denseLayer.feedforward(input));
    const auto output = denseLayer.output();

    EXPECT_TRUE(denseLayer.backpropagate(reference));

    // Test each node's error against the raw deviation.
    // Expect an exact match: ActFunc::None has a derivative of 1.0, so nothing scales the error.
    for (std::size_t i{}; i < Test::NodeCount; ++i)
    {
        EXPECT_NEAR(denseLayer.error()[i], reference[i] - output[i], Test::ExactTolerance);
    }
}

/**
 * @brief Verify that the error is scaled by the activation function's derivative taken at the
 *        pre-activation weighted sum, not at the layer's output.
 */
TEST(DenseLayerDense, BackpropagateUsesPreActivationDerivative)
{
    const Matrix1d input{1.5, -0.5};
    const Matrix1d reference{1.0, 0.0, -1.0};

    DenseLayer denseLayer{Test::NodeCount, Test::WeightCount, ActFunc::Tanh};

    // Recover the bias through the activation function: a zero input makes the weighted sum the
    // bias, so the output is tanh(bias), which std::atanh() inverts.
    const auto tanhOfBias = recoverBias(denseLayer);
    Matrix1d bias(Test::NodeCount, 0.0);

    for (std::size_t i{}; i < Test::NodeCount; ++i)
    {
        bias[i] = std::atanh(tanhOfBias[i]);
    }

    EXPECT_TRUE(denseLayer.feedforward(input));
    EXPECT_TRUE(denseLayer.backpropagate(reference));

    for (std::size_t i{}; i < Test::NodeCount; ++i)
    {
        const auto preActivation = weightedSum(denseLayer.weights(), bias, input, i);
        const auto delta         = ml::actFuncDelta(ActFunc::Tanh, preActivation);
        const auto expected      = (reference[i] - denseLayer.output()[i]) * delta;

        EXPECT_TRUE(std::isfinite(expected));
        EXPECT_NEAR(denseLayer.error()[i], expected, Test::ExactTolerance);
    }
}

/**
 * @brief Verify that a hidden layer computes its error from the next layer's error and weights.
 */
TEST(DenseLayerDense, BackpropagateHiddenLayerComputesError)
{
    constexpr std::size_t nextNodeCount{2U};
    const Matrix1d input{1.0, -2.0};
    const Matrix1d nextReference{1.0, -1.0};

    // ActFunc::Tanh rather than None, so this also pins down which value the derivative is taken
    // at. See the note on BackpropagateUsesPreActivationDerivative above.
    DenseLayer hiddenLayer{Test::NodeCount, Test::WeightCount, ActFunc::Tanh};
    DenseLayer nextLayer{nextNodeCount, Test::NodeCount, ActFunc::None};

    const auto tanhOfBias = recoverBias(hiddenLayer);
    Matrix1d bias(Test::NodeCount, 0.0);

    for (std::size_t i{}; i < Test::NodeCount; ++i)
    {
        bias[i] = std::atanh(tanhOfBias[i]);
    }

    // Give the next layer a non-zero error to propagate back, otherwise every expected value
    // below would be zero and a broken implementation would pass.
    EXPECT_TRUE(hiddenLayer.feedforward(input));
    EXPECT_TRUE(nextLayer.feedforward(hiddenLayer.output()));
    EXPECT_TRUE(nextLayer.backpropagate(nextReference));
    EXPECT_TRUE(hiddenLayer.backpropagate(nextLayer));

    for (std::size_t i{}; i < Test::NodeCount; ++i)
    {
        double sum{};

        for (std::size_t j{}; j < nextNodeCount; ++j)
        {
            sum += nextLayer.error()[j] * nextLayer.weights()[j][i];
        }
        const auto preActivation = weightedSum(hiddenLayer.weights(), bias, input, i);
        const auto expected      = sum * ml::actFuncDelta(ActFunc::Tanh, preActivation);

        EXPECT_TRUE(std::isfinite(hiddenLayer.error()[i]));
        EXPECT_NEAR(hiddenLayer.error()[i], expected, Test::ExactTolerance);
    }
}

/**
 * @brief Verify that optimization adjusts every bias and weight by the error, scaled by the
 *        learning rate and, for weights, by the corresponding input.
 */
TEST(DenseLayerDense, OptimizeUpdatesBiasAndWeights)
{
    const Matrix1d input{2.0, -1.0};
    const Matrix1d reference{1.0, 0.0, -1.0};

    DenseLayer denseLayer{Test::NodeCount, Test::WeightCount, ActFunc::None};
    EXPECT_TRUE(denseLayer.feedforward(input));
    EXPECT_TRUE(denseLayer.backpropagate(reference));

    const auto error         = denseLayer.error();
    const auto weightsBefore = denseLayer.weights();
    const auto biasBefore    = recoverBias(denseLayer);

    EXPECT_TRUE(denseLayer.optimize(input, Test::LearningRate));
    const auto biasAfter = recoverBias(denseLayer);

    for (std::size_t i{}; i < Test::NodeCount; ++i)
    {
        const auto changeRate = error[i] * Test::LearningRate;
        EXPECT_NEAR(biasAfter[i], biasBefore[i] + changeRate, Test::ExactTolerance);

        for (std::size_t j{}; j < Test::WeightCount; ++j)
        {
            EXPECT_NEAR(denseLayer.weights()[i][j], weightsBefore[i][j] + changeRate * input[j],
                        Test::ExactTolerance);
        }
    }
}

/**
 * @brief Verify that a rejected optimization leaves every parameter untouched.
 */
TEST(DenseLayerDense, OptimizeRejectedLeavesParametersUnchanged)
{
    const Matrix1d input{2.0, -1.0};
    const Matrix1d reference{1.0, 0.0, -1.0};
    const Matrix1d wrongSized(Test::WeightCount + 1U, 1.0);

    DenseLayer denseLayer{Test::NodeCount, Test::WeightCount, ActFunc::None};
    EXPECT_TRUE(denseLayer.feedforward(input));
    EXPECT_TRUE(denseLayer.backpropagate(reference));

    const auto weightsBefore = denseLayer.weights();
    const auto biasBefore    = recoverBias(denseLayer);

    EXPECT_FALSE(denseLayer.optimize(wrongSized, Test::LearningRate));
    EXPECT_FALSE(denseLayer.optimize(input, 0.0));
    EXPECT_FALSE(denseLayer.optimize(input, 1.0));

    const auto biasAfter = recoverBias(denseLayer);

    for (std::size_t i{}; i < Test::NodeCount; ++i)
    {
        EXPECT_NEAR(biasAfter[i], biasBefore[i], Test::ExactTolerance);

        for (std::size_t j{}; j < Test::WeightCount; ++j)
        {
            EXPECT_NEAR(denseLayer.weights()[i][j], weightsBefore[i][j], Test::ExactTolerance);
        }
    }
}

/**
 * @brief Verify that feedforward requires one input value per weight.
 */
TEST(DenseLayerDense, FeedforwardChecksInputSize)
{
    DenseLayer denseLayer{Test::NodeCount, Test::WeightCount};

    EXPECT_TRUE(denseLayer.feedforward(Matrix1d(Test::WeightCount, 1.0)));
    EXPECT_FALSE(denseLayer.feedforward(Matrix1d(Test::WeightCount - 1U, 1.0)));
    EXPECT_FALSE(denseLayer.feedforward(Matrix1d(Test::WeightCount + 1U, 1.0)));
    EXPECT_FALSE(denseLayer.feedforward(Matrix1d{}));

    // Test an input sized to the node count rather than the weight count.
    // Expect rejection: mixing the two counts up is the easiest mistake to make here.
    EXPECT_FALSE(denseLayer.feedforward(Matrix1d(Test::NodeCount, 1.0)));
}

/**
 * @brief Verify that output-layer backpropagation requires one reference value per node.
 */
TEST(DenseLayerDense, BackpropagateChecksReferenceSize)
{
    DenseLayer denseLayer{Test::NodeCount, Test::WeightCount};

    EXPECT_TRUE(denseLayer.backpropagate(Matrix1d(Test::NodeCount, 1.0)));
    EXPECT_FALSE(denseLayer.backpropagate(Matrix1d(Test::NodeCount - 1U, 1.0)));
    EXPECT_FALSE(denseLayer.backpropagate(Matrix1d(Test::NodeCount + 1U, 1.0)));
    EXPECT_FALSE(denseLayer.backpropagate(Matrix1d{}));
    EXPECT_FALSE(denseLayer.backpropagate(Matrix1d(Test::WeightCount, 1.0)));
}

/**
 * @brief Verify that hidden-layer backpropagation requires one weight per node in the next layer.
 */
TEST(DenseLayerDense, BackpropagateChecksNextLayerWeightCount)
{
    DenseLayer denseLayer{Test::NodeCount, Test::WeightCount};
    DenseLayer matching{1U, Test::NodeCount};
    DenseLayer mismatched{1U, Test::NodeCount + 1U};

    EXPECT_TRUE(denseLayer.backpropagate(matching));
    EXPECT_FALSE(denseLayer.backpropagate(mismatched));
}

/**
 * @brief Verify that optimization requires one input value per weight.
 */
TEST(DenseLayerDense, OptimizeChecksInputSize)
{
    DenseLayer denseLayer{Test::NodeCount, Test::WeightCount};

    EXPECT_TRUE(denseLayer.optimize(Matrix1d(Test::WeightCount, 1.0), Test::LearningRate));
    EXPECT_FALSE(denseLayer.optimize(Matrix1d(Test::WeightCount - 1U, 1.0), Test::LearningRate));
    EXPECT_FALSE(denseLayer.optimize(Matrix1d(Test::WeightCount + 1U, 1.0), Test::LearningRate));
    EXPECT_FALSE(denseLayer.optimize(Matrix1d{}, Test::LearningRate));
    EXPECT_FALSE(denseLayer.optimize(Matrix1d(Test::NodeCount, 1.0), Test::LearningRate));
}

/**
 * @brief Verify that optimization requires a learning rate inside the range (0.0, 1.0).
 */
TEST(DenseLayerDense, OptimizeChecksLearningRate)
{
    constexpr double validRates[]{1e-9, 0.01, 0.5, 0.999};
    constexpr double invalidRates[]{0.0, -1e-9, -0.5, 1.0, 1.5, 100.0};

    DenseLayer denseLayer{Test::NodeCount, Test::WeightCount};
    const Matrix1d input(Test::WeightCount, 1.0);

    for (const auto learningRate : validRates)
    {
        EXPECT_TRUE(denseLayer.optimize(input, learningRate));
    }
    for (const auto learningRate : invalidRates)
    {
        EXPECT_FALSE(denseLayer.optimize(input, learningRate));
    }
}

/**
 * @brief Verify that everything above works through a ml::dense_layer::Interface reference.
 */
TEST(DenseLayerDense, UsableThroughInterface)
{
    const Matrix1d input(Test::WeightCount, 1.0);
    const Matrix1d reference(Test::NodeCount, 1.0);

    DenseLayer denseLayer{Test::NodeCount, Test::WeightCount};
    DenseLayerInterface& layerInterface{denseLayer};

    EXPECT_EQ(layerInterface.nodeCount(), Test::NodeCount);
    EXPECT_EQ(layerInterface.weightCount(), Test::WeightCount);
    EXPECT_EQ(layerInterface.output().size(), Test::NodeCount);
    EXPECT_TRUE(layerInterface.feedforward(input));
    EXPECT_TRUE(layerInterface.backpropagate(reference));
    EXPECT_TRUE(layerInterface.optimize(input, Test::LearningRate));
    EXPECT_FALSE(layerInterface.feedforward(Matrix1d{}));
}

/**
 * @brief Verify that initParams() draws a new set of trainable parameters, leaving the
 *        layer's dimensions and every value's range intact.
 */
TEST(DenseLayerDense, InitParamsDrawsNewValues)
{
    DenseLayer denseLayer{Test::NodeCount, Test::WeightCount, ActFunc::None};

    const auto biasBefore    = recoverBias(denseLayer);
    const auto weightsBefore = denseLayer.weights();

    denseLayer.initParams();

    const auto biasAfter    = recoverBias(denseLayer);
    const auto weightsAfter = denseLayer.weights();

    // Test the shape of the layer after the reset.
    // Expect it unchanged: initParams() replaces values, it doesn't resize anything.
    EXPECT_EQ(denseLayer.nodeCount(), Test::NodeCount);
    EXPECT_EQ(denseLayer.weightCount(), Test::WeightCount);
    EXPECT_EQ(biasAfter.size(), Test::NodeCount);
    EXPECT_EQ(weightsAfter.size(), Test::NodeCount);

    // Test the new values against the range randomStartVal() draws from.
    // Expect every one of them inside [0.0, 1.0], exactly as after construction.
    std::size_t changedCount{};

    for (std::size_t i{}; i < Test::NodeCount; ++i)
    {
        EXPECT_TRUE((0.0 <= biasAfter[i]) && (1.0 >= biasAfter[i]));

        if (biasAfter[i] != biasBefore[i]) { ++changedCount; }

        for (std::size_t j{}; j < Test::WeightCount; ++j)
        {
            EXPECT_TRUE((0.0 <= weightsAfter[i][j]) && (1.0 >= weightsAfter[i][j]));

            if (weightsAfter[i][j] != weightsBefore[i][j]) { ++changedCount; }
        }
    }

    // Test how many of the nine values the reset replaced.
    // Expect at least one: a method that left every one of them alone reset nothing. Drawing the
    // same nine values twice from std::rand() is possible in theory and never seen in practice.
    EXPECT_TRUE(changedCount > std::size_t{});
}

/**
 * @brief Verify that a network of two dense layers actually learns, by training it on 2-bit XOR
 *        and comparing its error against its own error before training.
 */
TEST(DenseLayerDense, NetworkLearnsXorPattern)
{
    const Matrix2d trainInput{{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
    const Matrix2d trainOutput{{0.0}, {1.0}, {1.0}, {0.0}};

    DenseLayer hiddenLayer{Test::HiddenCount, trainInput[0U].size(), ActFunc::Tanh};
    DenseLayer outputLayer{1U, Test::HiddenCount, ActFunc::Tanh};
    NeuralNetwork network{hiddenLayer, outputLayer, trainInput, trainOutput};

    const auto errorBeforeTraining = meanAbsoluteError(network, trainInput, trainOutput);
    EXPECT_TRUE(std::isfinite(errorBeforeTraining));

    // Train until the network converges, giving it a few attempts to get there. A network that
    // settles into a bad set of parameters can't train its way out of them, and train() resets
    // both layers before its first epoch, so every attempt starts from newly drawn values.
    double errorAfterTraining{};

    for (std::size_t attempt{}; attempt < Test::TrainAttempts; ++attempt)
    {
        EXPECT_TRUE(network.train(Test::EpochCount, Test::PrecisionThreshold));
        errorAfterTraining = meanAbsoluteError(network, trainInput, trainOutput);

        if (errorAfterTraining < Test::ConvergenceThreshold) { break; }
    }
    EXPECT_TRUE(std::isfinite(errorAfterTraining));

    // Test the error against the threshold and against the untrained baseline.
    // Expect both: the threshold alone would pass on a network that started out lucky.
    EXPECT_TRUE(errorAfterTraining < Test::ConvergenceThreshold);
    EXPECT_TRUE(errorAfterTraining < errorBeforeTraining);

    for (std::size_t i{}; i < trainInput.size(); ++i)
    {
        const auto prediction = network.predict(trainInput[i])[0U];
        EXPECT_TRUE(std::isfinite(prediction));
        EXPECT_NEAR(prediction, trainOutput[i][0U], Test::ConvergenceThreshold);
    }
}
} // namespace
