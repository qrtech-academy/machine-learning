/**
 * @file Fixed linear regression model implementation details.
 */
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <exception>

#include "ml/lin_reg/fixed.hpp"
#include "ml/types.hpp"
#include "ml/utils.hpp"

namespace ml::lin_reg
{
// -----------------------------------------------------------------------------
Fixed::Fixed(const Matrix1d& trainIn, const Matrix1d& trainOut) noexcept
    : myTrainOrder{}
    , myTrainIn{trainIn}
    , myTrainOut{trainOut}
    , myBias{}
    , myWeight{}
{
    const auto setCount = std::min(trainIn.size(), trainOut.size());

    if (0U == setCount)
    {
        std::fprintf(stderr, "Cannot create linear regression model without training sets!\n");
        std::terminate();
    }

    // Initialize the training order matrix with 0, 1, 2... setCount-1.
    myTrainOrder.resize(setCount);

    for (std::size_t i{}; i < myTrainOrder.size(); ++i)
    {
        myTrainOrder[i] = static_cast<std::uint32_t>(i);
    }
    initRandGen();
}

// -----------------------------------------------------------------------------
double Fixed::predict(const double input) const noexcept { return myWeight * input + myBias; }

// -----------------------------------------------------------------------------
bool Fixed::train(const std::size_t epochCount, const double learningRate,
                  const double precisionThreshold) noexcept
{
    constexpr std::size_t evaluationInterval{10U};

    // Check the epoch count, return false if invalid.
    if (0U == epochCount) { return false; }

    // Check the learning rate, return false if invalid.
    if ((0.0 >= learningRate) || (1.0 <= learningRate)) { return false; }

    if ((0.0 >= precisionThreshold) || (1.0 <= precisionThreshold)) { return false; }

    for (std::size_t epoch{}; epoch < epochCount; ++epoch)
    {
        shuffle();

        // Iterate through each training set one by one.
        for (const auto i : myTrainOrder)
        {
            const auto input  = myTrainIn[i];
            const auto output = myTrainOut[i];
            optimize(input, output, learningRate);
        }

        // Evaluate the precision every tenth epoch.
        const bool evaluate{(0U < epoch) && (0U == (epoch % evaluationInterval))};
        if (evaluate)
        {
            const auto currentPrecision = precision();

            if (precisionThreshold <= currentPrecision)
            {
                std::printf("Target precision %.2f reached after %zu epochs!\n", currentPrecision,
                            epoch + 1U);
                break;
            }
        }
    }
    return true;
}

// -----------------------------------------------------------------------------
void Fixed::optimize(const double input, const double output, const double learningRate) noexcept
{
    // If input = 0, the output is equal to the target bias value.
    if (0U == input) { myBias = output; }
    else
    {
        // Perform prediction, calculate error, adjust parameters accordingly.
        const auto prediction = predict(input);
        const auto error      = output - prediction;

        myBias += error * learningRate;
        myWeight += error * learningRate * input;
    }
}

// -----------------------------------------------------------------------------
void Fixed::shuffle() noexcept
{
    // Iterate through trainOrder, swap index i and r.
    for (std::size_t i{}; i < myTrainOrder.size(); ++i)
    {
        const auto r    = std::rand() % myTrainOrder.size();
        const auto temp = myTrainOrder[i];
        myTrainOrder[i] = myTrainOrder[r];
        myTrainOrder[r] = temp;
    }
}

// -----------------------------------------------------------------------------
double Fixed::precision() const noexcept
{
    double errSum{};

    // Ackumulate the absolute error of each training set.
    for (std::size_t i{}; i < myTrainOrder.size(); ++i)
    {
        const auto input  = myTrainIn[i];
        const auto output = myTrainOut[i];
        const auto err    = std::abs(output - predict(input));
        errSum += err;
    }
    const auto avgError = errSum / myTrainOrder.size();
    return 1.0 - avgError;
}
} // namespace ml::lin_reg
