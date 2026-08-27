/**
 * @file Linear regression interface.
 */
#pragma once

namespace ml::lin_reg
{
/**
 * @brief Linear regression interface.
 */
class Interface
{
public:
    /**
     * @brief Destructor.
     */
    virtual ~Interface() noexcept = default;

    /**
     * @brief Predict based on the given input.
     *
     * @param[in] input Input value.
     *
     * @return The predicted output.
     */
    [[nodiscard]] virtual double predict(double input) const noexcept = 0;
};
} // namespace ml::lin_reg
