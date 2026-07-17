/**
 * @file Machine learning factory implementation details.
 */
#include <cstddef>
#include <memory>

#include "ml/act_func/none.hpp"
#include "ml/act_func/relu.hpp"
#include "ml/act_func/tanh.hpp"
#include "ml/conv_layer/conv.hpp"
#include "ml/conv_layer/max_pool.hpp"
#include "ml/dense_layer/dense.hpp"
#include "ml/factory/factory.hpp"
#include "ml/factory/stub.hpp"
#include "ml/flatten_layer/flatten.hpp"

namespace ml::factory
{
// -----------------------------------------------------------------------------
ActFuncPtr Factory::actFunc(const act_func::Type type) noexcept
{
    switch (type)
    {
        case act_func::Type::Relu:
            return std::make_unique<act_func::Relu>();
        case act_func::Type::Tanh:
            return std::make_unique<act_func::Tanh>();
        default:
            return std::make_unique<act_func::None>();
    }
}

// -----------------------------------------------------------------------------
ConvLayerPtr Factory::convLayer(const std::size_t inputSize, const std::size_t kernelSize,
                                const act_func::Type actFunc) noexcept
{
    return std::make_unique<conv_layer::Conv>(inputSize, kernelSize, actFunc);
}

// -----------------------------------------------------------------------------
DenseLayerPtr Factory::denseLayer(const std::size_t inputSize, const std::size_t outputSize,
                                  const act_func::Type actFunc) noexcept
{
    return std::make_unique<dense_layer::Dense>(inputSize, outputSize, actFunc);
}

// -----------------------------------------------------------------------------
FlattenLayerPtr Factory::flattenLayer(const std::size_t inputSize) noexcept
{
    return std::make_unique<flatten_layer::Flatten>(inputSize);
}

// -----------------------------------------------------------------------------
ConvLayerPtr Factory::maxPoolLayer(const std::size_t inputSize, const std::size_t poolSize) noexcept
{
    return std::make_unique<conv_layer::MaxPool>(inputSize, poolSize);
}

// -----------------------------------------------------------------------------
FactoryPtr create(const bool stub) noexcept
{
    if (stub) { return std::make_unique<Stub>(); }
    return std::make_unique<Factory>();
}
} // namespace ml::factory
