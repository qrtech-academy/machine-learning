/**
 * @file Utility function details.
 */
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
} // namespace ml
