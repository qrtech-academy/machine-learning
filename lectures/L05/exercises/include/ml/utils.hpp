/**
 * @file Utility functions.
 */
#pragma once

#include "ml/types.hpp"

namespace ml
{
/**
 * @brief Initialize random generator.
 *
 *        This operation is only done once.
 */
void initRandGen() noexcept;

/**
 * @brief Get a random starting value for trainable parameters.
 *
 * @return Random value in the range [-1.0, 1.0].
 */
[[nodiscard]] double randomStartVal() noexcept;

/**
 * @brief Compute the output of an activation function.
 *
 * @param[in] actFunc Activation function to use.
 * @param[in] input Input value, i.e. a node's weighted sum.
 *
 * @return The activation function's output for the given input.
 */
[[nodiscard]] double actFuncOutput(ActFunc actFunc, double input) noexcept;

/**
 * @brief Compute the derivative of an activation function.
 *
 * @param[in] actFunc Activation function to use.
 * @param[in] input Input value, i.e. a node's weighted sum, not the activation function's output.
 *
 * @return The activation function's derivative at the given input.
 */
[[nodiscard]] double actFuncDelta(ActFunc actFunc, double input) noexcept;

} // namespace ml
