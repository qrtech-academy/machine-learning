/**
 * @file Fixed linear regression model implementation details.
 */
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <exception>

#include "ml/lin_reg/fixed.hpp"
#include "ml/types.hpp"

namespace ml::lin_reg
{
// -----------------------------------------------------------------------------
Fixed::Fixed(const Matrix1d& trainIn, const Matrix1d& trainOut) noexcept
    : myTrainIn{trainIn}
    , myTrainOut{trainOut}
    , mySetCount{std::min(trainIn.size(), trainOut.size())}
    , myBias{}
    , myWeight{}
{
    if (0U == mySetCount)
    {
        std::fprintf(stderr, "Cannot create linear regression model without training sets!\n");
        std::terminate();
    }
}

// -----------------------------------------------------------------------------
double Fixed::predict(const double input) const noexcept { return myWeight * input + myBias; }

// -----------------------------------------------------------------------------
bool Fixed::train(const std::size_t epochCount, const double learningRate) noexcept
{
    // Check the epoch count, return false if invalid.
    if (0U == epochCount) { return false; }

    // Check the learning rate, return false if invalid.
    if ((0.0 >= learningRate) || (1.0 <= learningRate)) { return false; }

    for (std::size_t epoch{}; epoch < epochCount; ++epoch)
    {
        // Iterate through each training set one by one.
        for (std::size_t i{}; i < mySetCount; ++i)
        {
            const auto input  = myTrainIn[i];
            const auto output = myTrainOut[i];
            optimize(input, output, learningRate);
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
} // namespace ml::lin_reg
