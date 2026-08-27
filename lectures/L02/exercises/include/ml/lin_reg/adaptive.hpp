/**
 * @file Adaptive linear regression model.
 */
#pragma once

#include <cstddef>

#include "ml/lin_reg/interface.hpp"
#include "ml/types.hpp"

namespace ml::lin_reg
{
/**
 * @brief Adaptive linear regression model.
 *
 *        This class is non-copyable and non-movable.
 */
class Adaptive final : public Interface
{
public:
    /**
     * @brief Constructor.
     */
    explicit Adaptive(const Matrix1d& trainIn, const Matrix1d& trainOut) noexcept;

    /**
     * @brief Destructor.
     */
    ~Adaptive() noexcept override = default;

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
     * @param[in] epochCount Number of epochs to train the model. Must be greater than 0.
     * @param[in] precisionThreshold Minimum precision to stop training (default = 0.999999).
     *
     * @note The training is stopped once the specified precision is reached.
     */
    bool train(std::size_t epochCount, double precisionThreshold = 0.999999) noexcept;

    Adaptive()                           = delete; // No default constructor.
    Adaptive(const Adaptive&)            = delete; // No copy constructor.
    Adaptive(Adaptive&&)                 = delete; // No move constructor.
    Adaptive& operator=(const Adaptive&) = delete; // No copy assignment.
    Adaptive& operator=(Adaptive&&)      = delete; // No move assignment.

private:
    void optimize(double input, double output, double learningRate) noexcept;
    void shuffle() noexcept;
    [[nodiscard]] double precision() const noexcept;

    /** Matrix holding the training order via index. */
    MatrixU32 myTrainOrder;

    /** Training inputs. */
    const Matrix1d& myTrainIn;

    /** Training outputs. */
    const Matrix1d& myTrainOut;

    /** Bias value. */
    double myBias;

    /** Weight value. */
    double myWeight;
};
} // namespace ml::lin_reg
