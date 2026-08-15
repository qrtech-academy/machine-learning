/**
 * @file Fixed linear regression model.
 */
#pragma once

#include <cstddef>

#include "ml/lin_reg/interface.hpp"
#include "ml/types.hpp"

namespace ml::lin_reg
{
/**
 * @brief Fixed linear regression model.
 */
class Fixed final : public Interface
{
public:
    /**
     * @brief Constructor.
     */
    explicit Fixed(const Matrix1d& trainIn, const Matrix1d& trainOut) noexcept;

    /**
     * @brief Destructor.
     */
    ~Fixed() noexcept override = default;

    /**
     * @brief Predict based on the given input.
     *
     * @param[in] input Input value.
     *
     * @return The predicted output.
     */
    [[nodiscard]] double predict(double input) const noexcept override;

    /**
     * @brief Train the model with the given parameters.
     *
     * @param[in] epochCount Must be greater than 0.
     * @param[in] learningRate Must be in range (0.0, 1.0).
     */
    bool train(std::size_t epochCount, double learningRate = 0.01) noexcept;

    Fixed()                        = delete; // No default constructor.
    Fixed(const Fixed&)            = delete; // No copy constructor.
    Fixed(Fixed&&)                 = delete; // No move constructor.
    Fixed& operator=(const Fixed&) = delete; // No copy assignment.
    Fixed& operator=(Fixed&&)      = delete; // No move assignment.

private:
    void optimize(double input, double output, double learningRate) noexcept;

    /** Training inputs. */
    const Matrix1d& myTrainIn;

    /** Training outputs. */
    const Matrix1d& myTrainOut;

    /** Number of complete training sets. */
    const std::size_t mySetCount;

    /** Bias value. */
    double myBias;

    /** Weight value. */
    double myWeight;
};
} // namespace ml::lin_reg
