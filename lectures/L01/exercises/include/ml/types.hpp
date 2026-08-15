/**
 * @file Machine learning type definitions..
 */
#pragma once

#include <vector>

namespace ml
{
/** One-dimensional matrix. */
using Matrix1d = std::vector<double>;

/** Two-dimensional matrix. */
using Matrix2d = std::vector<Matrix1d>;

/** Three-dimensional matrix. */
using Matrix3d = std::vector<Matrix2d>;
} // namespace ml
