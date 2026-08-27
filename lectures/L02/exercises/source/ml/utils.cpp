/**
 * @brief Miscellaneous machine learning utility functions.
 */
#include <cstdlib>
#include <ctime>

#include "ml/utils.hpp"

namespace ml
{
// -----------------------------------------------------------------------------
void initRandGen() noexcept
{
    // Initialize the random generator with the current time as seed once only.
    static bool initialized{false};
    if (initialized) { return; }
    std::srand(std::time(nullptr));
    initialized = true;
}
} // namespace ml
