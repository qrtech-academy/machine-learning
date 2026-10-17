/**
 * @file Utility function details.
 */
#include <cmath>
#include <cstdlib>
#include <ctime>

#include "ml/utils.hpp"

namespace ml
{
// -----------------------------------------------------------------------------
void initRandGen() noexcept
{
    static bool initialized{false};
    if (initialized) { return; }
    std::srand(std::time(nullptr));
    initialized = true;
}

// -----------------------------------------------------------------------------
double randomStartVal() noexcept
{
    constexpr double randMax{static_cast<double>(RAND_MAX)};
    const auto ratio = std::rand() / randMax; // Range [0.0, 1.0].
    return 2.0 * ratio - 1.0;                 // Rescaled to range [-1.0, 1.0].
}

// -----------------------------------------------------------------------------
double actFuncOutput(const ActFunc actFunc, const double input) noexcept
{
    switch (actFunc)
    {
        case ActFunc::Relu:
        {
            return 0.0 < input ? input : 0.0;
        }
        case ActFunc::Tanh:
        {
            return std::tanh(input);
        }
        default:
        {
            return input;
        }
    }
}

// -----------------------------------------------------------------------------
double actFuncDelta(const ActFunc actFunc, const double input) noexcept
{
    switch (actFunc)
    {
        case ActFunc::Relu:
        {
            return 0.0 < input ? 1.0 : 0.0;
        }
        case ActFunc::Tanh:
        {
            const auto out = std::tanh(input);
            return 1.0 - out * out;
        }
        default:
        {
            return 1.0;
        }
    }
}
} // namespace ml
