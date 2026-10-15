/**
 * @file Shallow network implementation details.
 */
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>

#include "ml/dense_layer/interface.hpp"
#include "ml/neural_network/shallow.hpp"
#include "ml/types.hpp"
#include "ml/utils.hpp"

namespace ml::neural_network
{
namespace
{
// -----------------------------------------------------------------------------
void updateLearningRate(double& learningRate, double& prevPrecision,
                        const double currPrecision) noexcept
{
    constexpr double min{0.01};
    constexpr double max{0.25};
    constexpr double minImprovement{0.1};
    constexpr double step{0.05};
    constexpr double halve{0.5};
    const auto delta = currPrecision - prevPrecision;

    // Increase the learning rate if the training is going too slow.
    // Decrease the learning rate if performance is decreasing.
    if (0.0 < delta)
    {
        if (minImprovement > delta) { learningRate += step; }
    }
    else { learningRate *= halve; }

    // Clamp the learning rate between [0.01, 0.25].
    learningRate  = std::clamp(learningRate, min, max);
    prevPrecision = currPrecision;
}

// -----------------------------------------------------------------------------
[[nodiscard]] bool isEvaluationEpoch(const std::size_t epoch) noexcept
{
    constexpr std::size_t evaluationInterval{100U};
    const auto firstEpoch = (0U == epoch);
    const auto noRest     = (0U == (epoch % evaluationInterval));
    return !firstEpoch && noRest;
}

// -----------------------------------------------------------------------------
[[nodiscard]] double averageAbs(const Matrix1d& x, const Matrix1d& y) noexcept
{
    double sum{};
    const auto count = std::min(x.size(), y.size());
    if (0U == count) { return 0.0; }

    for (std::size_t i{}; i < count; ++i)
    {
        sum += std::fabs(x[i] - y[i]);
    }
    return sum / count;
}
} // namespace

// -----------------------------------------------------------------------------
Shallow::Shallow(dense_layer::Interface& hiddenLayer, dense_layer::Interface& outputLayer,
                 const Matrix2d& trainInput, const Matrix2d& trainOutput) noexcept
    : myTrainOrder{}
    , myHiddenLayer{hiddenLayer}
    , myOutputLayer{outputLayer}
    , myTrainInput{trainInput}
    , myTrainOutput{trainOutput}
{
    if (myOutputLayer.weightCount() != myHiddenLayer.nodeCount())
    {
        std::fprintf(stderr, "Dimension mismatch between the dense layers!\n");
        std::terminate();
    }
    const auto setCount = std::min(myTrainInput.size(), myTrainOutput.size());

    if (0U == setCount)
    {
        std::fprintf(stderr, "Cannot train the network without training sets!\n");
        std::terminate();
    }
    initTrainOrder(setCount);
    initRandGen();
}

// -----------------------------------------------------------------------------
const Matrix1d& Shallow::predict(const Matrix1d& input) noexcept
{
    // Perform a feedforward, return the predicted output.
    feedforward(input);
    return myOutputLayer.output();
}

// -----------------------------------------------------------------------------
bool Shallow::train(const std::size_t epochCount, const double precisionThreshold) noexcept
{
    static constexpr double initialLearningRate{0.01};

    // Check the input arguments, return false if invalid.
    if (0U == epochCount) { return false; }
    const bool thresholdValid{(0.0 < precisionThreshold) && (1.0 > precisionThreshold)};
    if (!thresholdValid) { return false; }

    double learningRate{initialLearningRate};
    double prevPrecision{};
    myHiddenLayer.initParams();
    myOutputLayer.initParams();

    for (std::size_t epoch{}; epoch < epochCount; ++epoch)
    {
        randomizeTrainOrder();

        // Go through each training set one by one.
        for (const auto i : myTrainOrder)
        {
            const auto& input     = myTrainInput[i];
            const auto& reference = myTrainOutput[i];

            // Run feedforward -> backpropagation -> optimization through the layers.
            if (!feedforward(input)) { return false; }
            if (!backpropagate(reference)) { return false; }
            if (!optimize(input, learningRate)) { return false; }
        }
        // Stop training once the precision threshold has been reached.
        if (isEvaluationEpoch(epoch))
        {
            const auto currPrecision = precision();

            if (precisionThreshold <= currPrecision)
            {
                std::printf("Reached precision %g after %zu epochs!\n", currPrecision, epoch + 1U);
                return true;
            }
            updateLearningRate(learningRate, prevPrecision, currPrecision);
        }
    }
    return true;
}

// -----------------------------------------------------------------------------
void Shallow::initTrainOrder(const std::size_t setCount) noexcept
{
    myTrainOrder.resize(setCount);

    for (std::uint32_t i{}; i < setCount; ++i)
    {
        myTrainOrder[i] = i;
    }
}

// -----------------------------------------------------------------------------
void Shallow::randomizeTrainOrder() noexcept
{
    for (std::size_t i{}; i < myTrainOrder.size(); ++i)
    {
        const auto r    = std::rand() % myTrainOrder.size();
        const auto temp = myTrainOrder[i];
        myTrainOrder[i] = myTrainOrder[r];
        myTrainOrder[r] = temp;
    }
}

// -----------------------------------------------------------------------------
bool Shallow::feedforward(const Matrix1d& input) noexcept
{
    if (!myHiddenLayer.feedforward(input)) { return false; }
    return myOutputLayer.feedforward(myHiddenLayer.output());
}

// -----------------------------------------------------------------------------
bool Shallow::backpropagate(const Matrix1d& reference) noexcept
{
    if (!myOutputLayer.backpropagate(reference)) { return false; }
    return myHiddenLayer.backpropagate(myOutputLayer);
}

// -----------------------------------------------------------------------------
bool Shallow::optimize(const Matrix1d& input, const double learningRate) noexcept
{
    if (!myHiddenLayer.optimize(input, learningRate)) { return false; }
    return myOutputLayer.optimize(myHiddenLayer.output(), learningRate);
}

// -----------------------------------------------------------------------------
double Shallow::precision() noexcept
{
    double sum{};

    for (std::size_t i{}; i < myTrainOrder.size(); ++i)
    {
        const auto& input      = myTrainInput[i];
        const auto& reference  = myTrainOutput[i];
        const auto& prediction = predict(input);

        sum += averageAbs(reference, prediction);
    }
    const auto avg = sum / myTrainOrder.size();
    return 1.0 - avg;
}
} // namespace ml::neural_network
