/**
 * @file Dense layer implementation details.
 */
#include <cstddef>
#include <cstdio>
#include <exception>

#include "ml/dense_layer/dense.hpp"
#include "ml/types.hpp"
#include "ml/utils.hpp"

namespace ml::dense_layer
{
// -----------------------------------------------------------------------------
Dense::Dense(const std::size_t nodeCount, const std::size_t weightCount,
             const ActFunc actFunc) noexcept
    : myWeights{}
    , myBias{}
    , myOutput{}
    , myPreActOutput{}
    , myError{}
    , myActFunc{actFunc}
{
    // Check the layer parameters, terminate if invalid.
    if ((0U == nodeCount) || (0U == weightCount))
    {
        std::fprintf(stderr, "Invalid dense layer parameters: "
                             "nodeCount and weightCount must be greater than 0!\n");
        std::terminate();
    }

    // Size the matrices, then initialize the trainable parameters with random values.
    myWeights.resize(nodeCount, Matrix1d(weightCount));
    myBias.resize(nodeCount);
    myOutput.resize(nodeCount);
    myPreActOutput.resize(nodeCount);
    myError.resize(nodeCount);

    initRandGen();
    initParams();
}

// -----------------------------------------------------------------------------
const Matrix1d& Dense::output() const noexcept { return myOutput; }

// -----------------------------------------------------------------------------
const Matrix1d& Dense::error() const noexcept { return myError; }

// -----------------------------------------------------------------------------
const Matrix2d& Dense::weights() const noexcept { return myWeights; }

// -----------------------------------------------------------------------------
std::size_t Dense::nodeCount() const noexcept { return myOutput.size(); }

// -----------------------------------------------------------------------------
std::size_t Dense::weightCount() const noexcept { return myWeights[0U].size(); }

// -----------------------------------------------------------------------------
bool Dense::feedforward(const Matrix1d& input) noexcept
{
    // Check the input arguments, return false if invalid.
    if (input.size() != weightCount())
    {
        std::fprintf(stderr, "Input dimension mismatch: expected %zu, actual: %zu!\n",
                     weightCount(), input.size());
        return false;
    }

    // Compute new outputs for each node in the layer.
    for (std::size_t i{}; i < nodeCount(); ++i)
    {
        auto sum = myBias[i];

        for (std::size_t j{}; j < weightCount(); ++j)
        {
            sum += input[j] * myWeights[i][j];
        }
        myPreActOutput[i] = sum;
        myOutput[i]       = actFuncOutput(myActFunc, sum);
    }
    return true;
}

// -----------------------------------------------------------------------------
bool Dense::backpropagate(const Matrix1d& reference) noexcept
{
    // Check the input arguments, return false if invalid.
    if (reference.size() != nodeCount())
    {
        std::fprintf(stderr, "Output dimension mismatch: expected %zu, actual: %zu!\n", nodeCount(),
                     reference.size());
        return false;
    }

    // Compute error values for each node in the layer.
    for (std::size_t i{}; i < nodeCount(); ++i)
    {
        const auto delta = actFuncDelta(myActFunc, myPreActOutput[i]);
        const auto error = reference[i] - myOutput[i];
        myError[i]       = error * delta;
    }
    return true;
}

// -----------------------------------------------------------------------------
bool Dense::backpropagate(const Interface& nextLayer) noexcept
{
    // Check the input arguments, return false if invalid.
    if (nextLayer.weightCount() != nodeCount())
    {
        std::fprintf(stderr, "Layer dimension mismatch: expected %zu, actual: %zu!\n", nodeCount(),
                     nextLayer.weightCount());
        return false;
    }

    // Compute error values for each node in the layer.
    for (std::size_t i{}; i < nodeCount(); ++i)
    {
        const auto delta = actFuncDelta(myActFunc, myPreActOutput[i]);
        double error{};

        for (std::size_t j{}; j < nextLayer.nodeCount(); ++j)
        {
            error += nextLayer.error()[j] * nextLayer.weights()[j][i];
        }
        myError[i] = error * delta;
    }
    return true;
}

// -----------------------------------------------------------------------------
bool Dense::optimize(const Matrix1d& input, const double learningRate) noexcept
{
    // Check the input arguments, return false if invalid.
    const bool lrValid{(0.0 < learningRate) && (1.0 > learningRate)};

    if (!lrValid)
    {
        std::fprintf(stderr, "Invalid learning rate %g!\n", learningRate);
        return false;
    }
    if (input.size() != weightCount())
    {
        std::fprintf(stderr, "Input dimension mismatch: expected %zu, actual: %zu!\n",
                     weightCount(), input.size());
        return false;
    }

    // Adjust all trainable parameters in the layer.
    for (std::size_t i{}; i < nodeCount(); ++i)
    {
        const auto changeRate = myError[i] * learningRate;
        myBias[i] += changeRate;

        for (std::size_t j{}; j < weightCount(); ++j)
        {
            myWeights[i][j] += changeRate * input[j];
        }
    }
    return true;
}

// -----------------------------------------------------------------------------
void Dense::initParams() noexcept
{
    // Initialize all parameters with values in range [-1.0, 1.0].
    for (std::size_t i{}; i < nodeCount(); ++i)
    {
        myBias[i] = randomStartVal();

        for (std::size_t j{}; j < weightCount(); ++j)
        {
            myWeights[i][j] = randomStartVal();
        }
    }
}
} // namespace ml::dense_layer
