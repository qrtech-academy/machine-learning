/**
 * @file Application entry point.
 */
#include <cstdio>

#include "ml/lin_reg/adaptive.hpp"
#include "ml/types.hpp"

namespace
{
/**
 * @brief Evaluate the model with the given inputs.
 *
 * @param[in] linReg Model to evaluate.
 * @param[in] inputs Input values for the evaluation.
 */
void evaluateModel(const ml::lin_reg::Interface& linReg, const ml::Matrix1d& inputs) noexcept
{
    for (const auto& input : inputs)
    {
        std::printf("Input: %g, ", input);
        std::printf("prediction: %g\n", linReg.predict(input));
    }
}
} // namespace

/**
 * @brief Train and predict with a linear regression model.
 *
 * @return 0 if the model was trained, -1 otherwise.
 */
int main()
{
    constexpr std::size_t epochCount{100U};

    // Create linear regression model holding five training sets (y = 3x + 2).
    const ml::Matrix1d trainIn{0.0, 1.0, 2.0, 3.0, 4.0};
    const ml::Matrix1d trainOut{2.0, 5.0, 8.0, 11.0, 14.0};
    ml::lin_reg::Adaptive linReg{trainIn, trainOut};

    // Train the model, print the results on success.
    const auto trained = linReg.train(epochCount);

    if (trained) { evaluateModel(linReg, trainIn); }
    else { std::fprintf(stderr, "Training failed!\n"); }
    return trained ? 0 : -1;
}
