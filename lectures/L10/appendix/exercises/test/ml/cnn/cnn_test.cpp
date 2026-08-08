/**
 * @file Component tests for ml::cnn::Cnn. See the suite's README for the chosen thresholds.
 */
#include <cmath>
#include <cstddef>

#include "ml/act_func/type.hpp"
#include "ml/cnn/cnn.hpp"
#include "ml/cnn/interface.hpp"
#include "ml/conv_layer/conv.hpp"
#include "ml/conv_layer/max_pool.hpp"
#include "ml/dense_layer/dense.hpp"
#include "ml/factory/factory.hpp"
#include "ml/flatten_layer/flatten.hpp"
#include "ml/types.hpp"
#include "qacademy/test/test.hpp"

using ActFunc  = ml::act_func::Type;
using Cnn      = ml::cnn::Cnn;
using Matrix1d = ml::Matrix1d;
using Matrix2d = ml::Matrix2d;
using Matrix3d = ml::Matrix3d;

namespace
{
/**
 * @brief Test parameters.
 */
struct Test
{
    /** Value tolerance for exact comparisons when using EXPECT_NEAR(). */
    static constexpr double ExactTolerance{1e-9};

    /** Conv layer input size, i.e. the image size. */
    static constexpr std::size_t InputSize{4U};

    /** Conv layer kernel size. */
    static constexpr std::size_t KernelSize{2U};

    /** Max pooling layer pool size. */
    static constexpr std::size_t PoolSize{2U};

    /** Output node count, one per digit the network can recognize. */
    static constexpr std::size_t OutputSize{4U};

    /** Hidden node count used by the convergence test. */
    static constexpr std::size_t HiddenCount{8U};

    /** Epochs used by the convergence tests. */
    static constexpr std::size_t EpochCount{5000U};

    /** Learning rate used by the convergence tests. */
    static constexpr double LearningRate{0.01};

    /** Mean absolute error the network with a hidden layer must reach. */
    static constexpr double ConvergenceThreshold{0.25};

    /** Mean absolute error the single dense layer network must reach. */
    static constexpr double SingleLayerThreshold{0.5};

    /** A learning rate large enough to move the kernel visibly in a single step. */
    static constexpr double PipelineLearningRate{0.1};

    /** Learning rates outside the range training accepts. */
    static constexpr double ZeroLearningRate{0.0};
    static constexpr double NegativeLearningRate{-0.1};
    static constexpr double UpperBoundLearningRate{1.0};
    static constexpr double AboveUpperLearningRate{1.5};

    /** Index of the first training set, used wherever any one image will do. */
    static constexpr std::size_t FirstDigit{0U};
};

// -----------------------------------------------------------------------------
Matrix3d trainingInput() noexcept
{
    // The digits 0 - 3 as 4x4 images, the set the cnn_work demo trains on.
    return Matrix3d{
        {{1, 1, 1, 1}, {1, 0, 0, 1}, {1, 0, 0, 1}, {1, 1, 1, 1}},
        {{0, 1, 0, 0}, {0, 1, 0, 0}, {0, 1, 0, 0}, {0, 1, 0, 0}},
        {{1, 0, 0, 1}, {0, 1, 1, 0}, {0, 1, 1, 0}, {1, 0, 0, 1}},
        {{0, 0, 1, 0}, {0, 0, 1, 0}, {1, 1, 1, 1}, {0, 0, 1, 0}},
    };
}

// -----------------------------------------------------------------------------
Matrix2d trainingOutput() noexcept
{
    // One-hot encoded: the target digit's node is 1.0, the rest -1.0.
    return Matrix2d{
        {1.0, -1.0, -1.0, -1.0},
        {-1.0, 1.0, -1.0, -1.0},
        {-1.0, -1.0, 1.0, -1.0},
        {-1.0, -1.0, -1.0, 1.0},
    };
}

// -----------------------------------------------------------------------------
double meanAbsoluteError(ml::cnn::Interface& cnn, const Matrix3d& input,
                         const Matrix2d& reference) noexcept
{
    double sum{};
    std::size_t count{};

    for (std::size_t i{}; i < input.size(); ++i)
    {
        const auto& prediction = cnn.predict(input[i]);

        for (std::size_t j{}; j < prediction.size(); ++j)
        {
            sum += std::fabs(reference[i][j] - prediction[j]);
            ++count;
        }
    }
    return 0U == count ? 0.0 : sum / static_cast<double>(count);
}

// -----------------------------------------------------------------------------
std::size_t strongestNode(const Matrix1d& prediction) noexcept
{
    std::size_t index{};

    for (std::size_t i{1U}; i < prediction.size(); ++i)
    {
        if (prediction[i] > prediction[index]) { index = i; }
    }
    return index;
}

/**
 * @brief Verify that the network reports the image size it takes and the node count it predicts.
 */
TEST(CnnComponent, ConstructedDimensions)
{
    auto factory = ml::factory::create();
    const Cnn cnn{*factory,       Test::InputSize,  Test::KernelSize, ActFunc::Tanh,
                  Test::PoolSize, Test::OutputSize, ActFunc::Tanh};

    EXPECT_EQ(cnn.inputSize(), Test::InputSize);
    EXPECT_EQ(cnn.outputSize(), Test::OutputSize);
}

/**
 * @brief Verify that the layers are sized to fit each other at other sizes too.
 */
TEST(CnnComponent, LayersAreSizedToFitEachOther)
{
    constexpr std::size_t inputSize{8U};
    constexpr std::size_t poolSize{4U};
    constexpr std::size_t outputSize{2U};

    auto factory = ml::factory::create();
    const Cnn cnn{*factory, inputSize,  Test::KernelSize, ActFunc::Tanh,
                  poolSize, outputSize, ActFunc::Tanh};

    EXPECT_EQ(cnn.inputSize(), inputSize);
    EXPECT_EQ(cnn.outputSize(), outputSize);
}

/**
 * @brief Verify that adding a dense layer re-points the network's output at the new last layer.
 */
TEST(CnnComponent, AddDenseLayerChangesOutputSize)
{
    constexpr std::size_t addedOutputSize{3U};

    auto factory = ml::factory::create();
    Cnn cnn{*factory,       Test::InputSize,  Test::KernelSize, ActFunc::Tanh,
            Test::PoolSize, Test::OutputSize, ActFunc::Tanh};

    EXPECT_EQ(cnn.outputSize(), Test::OutputSize);

    cnn.addDenseLayer(addedOutputSize, ActFunc::Tanh);
    EXPECT_EQ(cnn.outputSize(), addedOutputSize);
}

/**
 * @brief Verify that a prediction holds one value per output node, even before training.
 */
TEST(CnnComponent, PredictReturnsOneValuePerOutputNode)
{
    auto factory = ml::factory::create();
    Cnn cnn{*factory,       Test::InputSize,  Test::KernelSize, ActFunc::Tanh,
            Test::PoolSize, Test::OutputSize, ActFunc::Tanh};

    EXPECT_EQ(cnn.predict(trainingInput()[Test::FirstDigit]).size(), Test::OutputSize);
}

/**
 * @brief Verify that predicting twice with the same image yields the same answer.
 */
TEST(CnnComponent, PredictIsRepeatable)
{
    auto factory = ml::factory::create();
    Cnn cnn{*factory,       Test::InputSize,  Test::KernelSize, ActFunc::Tanh,
            Test::PoolSize, Test::OutputSize, ActFunc::Tanh};

    const auto first  = cnn.predict(trainingInput()[Test::FirstDigit]);
    const auto second = cnn.predict(trainingInput()[Test::FirstDigit]);

    EXPECT_EQ(first.size(), second.size());

    for (std::size_t i{}; i < first.size(); ++i)
    {
        EXPECT_NEAR(first[i], second[i], Test::ExactTolerance);
    }
}

/**
 * @brief Verify that training rejects a learning rate outside the range it accepts.
 */
TEST(CnnComponent, TrainRejectsInvalidLearningRate)
{
    constexpr std::size_t epochCount{1U};

    auto factory = ml::factory::create();
    Cnn cnn{*factory,       Test::InputSize,  Test::KernelSize, ActFunc::Tanh,
            Test::PoolSize, Test::OutputSize, ActFunc::Tanh};

    EXPECT_FALSE(cnn.train(trainingInput(), trainingOutput(), epochCount, Test::ZeroLearningRate));
    EXPECT_FALSE(
        cnn.train(trainingInput(), trainingOutput(), epochCount, Test::NegativeLearningRate));
    EXPECT_FALSE(
        cnn.train(trainingInput(), trainingOutput(), epochCount, Test::AboveUpperLearningRate));
    EXPECT_FALSE(
        cnn.train(trainingInput(), trainingOutput(), epochCount, Test::UpperBoundLearningRate));
}

/**
 * @brief Verify that training rejects an epoch count of zero.
 */
TEST(CnnComponent, TrainRejectsZeroEpochs)
{
    auto factory = ml::factory::create();
    Cnn cnn{*factory,       Test::InputSize,  Test::KernelSize, ActFunc::Tanh,
            Test::PoolSize, Test::OutputSize, ActFunc::Tanh};

    EXPECT_FALSE(cnn.train(trainingInput(), trainingOutput(), 0U, Test::LearningRate));
}

/**
 * @brief Verify that training rejects empty training data.
 */
TEST(CnnComponent, TrainRejectsEmptyTrainingData)
{
    constexpr std::size_t epochCount{1U};

    auto factory = ml::factory::create();
    Cnn cnn{*factory,       Test::InputSize,  Test::KernelSize, ActFunc::Tanh,
            Test::PoolSize, Test::OutputSize, ActFunc::Tanh};

    EXPECT_FALSE(cnn.train(Matrix3d{}, Matrix2d{}, epochCount, Test::LearningRate));
    EXPECT_FALSE(cnn.train(trainingInput(), Matrix2d{}, epochCount, Test::LearningRate));
    EXPECT_FALSE(cnn.train(Matrix3d{}, trainingOutput(), epochCount, Test::LearningRate));
}

/**
 * @brief Verify that training succeeds and actually changes the network.
 */
TEST(CnnComponent, TrainChangesThePrediction)
{
    constexpr std::size_t epochCount{100U};

    auto factory = ml::factory::create();
    Cnn cnn{*factory,       Test::InputSize,  Test::KernelSize, ActFunc::Tanh,
            Test::PoolSize, Test::OutputSize, ActFunc::Tanh};

    const auto before = cnn.predict(trainingInput()[Test::FirstDigit]);
    EXPECT_TRUE(cnn.train(trainingInput(), trainingOutput(), epochCount, Test::LearningRate));
    const auto after = cnn.predict(trainingInput()[Test::FirstDigit]);

    bool changed{false};

    for (std::size_t i{}; i < before.size(); ++i)
    {
        if (before[i] != after[i]) { changed = true; }
    }
    EXPECT_TRUE(changed);
}

/**
 * @brief Verify that the network learns to recognize all four digits.
 */
TEST(CnnComponent, LearnsToRecognizeAllFourDigits)
{
    const auto input     = trainingInput();
    const auto reference = trainingOutput();

    auto factory = ml::factory::create();
    Cnn cnn{*factory,       Test::InputSize,   Test::KernelSize, ActFunc::Tanh,
            Test::PoolSize, Test::HiddenCount, ActFunc::Tanh};
    cnn.addDenseLayer(Test::OutputSize, ActFunc::Tanh);

    const auto errorBefore = meanAbsoluteError(cnn, input, reference);

    EXPECT_TRUE(cnn.train(input, reference, Test::EpochCount, Test::LearningRate));

    // Test each digit in turn.
    for (std::size_t i{}; i < input.size(); ++i)
    {
        // Expect the node matching the digit to be the strongest one.
        EXPECT_EQ(strongestNode(cnn.predict(input[i])), i);
    }

    // Expect the error below the threshold, and below where it started.
    const auto errorAfter = meanAbsoluteError(cnn, input, reference);
    EXPECT_TRUE(errorAfter < Test::ConvergenceThreshold);
    EXPECT_TRUE(errorAfter < errorBefore);
}

/**
 * @brief Verify that the demo's own topology, a single dense layer, trains to a low error.
 */
TEST(CnnComponent, SingleDenseLayerNetworkTrainsToLowError)
{
    const auto input     = trainingInput();
    const auto reference = trainingOutput();

    auto factory = ml::factory::create();
    Cnn cnn{*factory,       Test::InputSize,  Test::KernelSize, ActFunc::Tanh,
            Test::PoolSize, Test::OutputSize, ActFunc::Tanh};

    const auto errorBefore = meanAbsoluteError(cnn, input, reference);

    EXPECT_TRUE(cnn.train(input, reference, Test::EpochCount, Test::LearningRate));

    const auto errorAfter = meanAbsoluteError(cnn, input, reference);
    EXPECT_TRUE(errorAfter < Test::SingleLayerThreshold);
    EXPECT_TRUE(errorAfter < errorBefore);
}

/**
 * @brief Verify that backpropagation carries from the dense layer back to the conv kernel.
 *
 *        Wired by hand rather than through Cnn, which keeps its layers private.
 */
TEST(CnnComponent, BackpropagationReachesTheConvLayerThroughEveryLayer)
{
    ml::conv_layer::Conv convLayer{Test::InputSize, Test::KernelSize, ActFunc::Tanh};
    ml::conv_layer::MaxPool maxPool{convLayer.outputSize(), Test::PoolSize};
    ml::flatten_layer::Flatten flattenLayer{maxPool.outputSize()};
    ml::dense_layer::Dense denseLayer{flattenLayer.outputSize(), Test::OutputSize, ActFunc::Tanh};

    // Expect each layer to consume exactly what the one before it produces.
    EXPECT_EQ(maxPool.inputSize(), convLayer.outputSize());
    EXPECT_EQ(flattenLayer.inputSize(), maxPool.outputSize());
    EXPECT_EQ(denseLayer.inputSize(), flattenLayer.outputSize());

    const auto input        = trainingInput()[Test::FirstDigit];
    const auto reference    = trainingOutput()[Test::FirstDigit];
    const auto kernelBefore = convLayer.kernel();
    const auto biasBefore   = convLayer.bias();

    // Feed forward along the whole chain.
    EXPECT_TRUE(convLayer.feedforward(input));
    EXPECT_TRUE(maxPool.feedforward(convLayer.output()));
    EXPECT_TRUE(flattenLayer.feedforward(maxPool.output()));
    EXPECT_TRUE(denseLayer.feedforward(flattenLayer.output()));

    // Backpropagate back along it, in reverse.
    EXPECT_TRUE(denseLayer.backpropagate(reference));
    EXPECT_TRUE(flattenLayer.backpropagate(denseLayer.inputGradients()));
    EXPECT_TRUE(maxPool.backpropagate(flattenLayer.inputGradients()));
    EXPECT_TRUE(convLayer.backpropagate(maxPool.inputGradients()));

    EXPECT_TRUE(convLayer.optimize(Test::PipelineLearningRate));

    // Expect the conv parameters to have moved; a dropped gradient would leave them untouched.
    bool parametersMoved{convLayer.bias() != biasBefore};

    for (std::size_t i{}; i < kernelBefore.size(); ++i)
    {
        for (std::size_t j{}; j < kernelBefore.size(); ++j)
        {
            if (convLayer.kernel()[i][j] != kernelBefore[i][j]) { parametersMoved = true; }
        }
    }
    EXPECT_TRUE(parametersMoved);
}

/**
 * @brief Verify that the network can be used through cnn::Interface.
 */
TEST(CnnComponent, UsableThroughInterface)
{
    auto factory = ml::factory::create();
    Cnn cnn{*factory,       Test::InputSize,  Test::KernelSize, ActFunc::Tanh,
            Test::PoolSize, Test::OutputSize, ActFunc::Tanh};
    ml::cnn::Interface& network{cnn};

    EXPECT_EQ(network.inputSize(), Test::InputSize);
    EXPECT_EQ(network.outputSize(), Test::OutputSize);
    EXPECT_EQ(network.predict(trainingInput()[Test::FirstDigit]).size(), Test::OutputSize);
}
} // namespace
