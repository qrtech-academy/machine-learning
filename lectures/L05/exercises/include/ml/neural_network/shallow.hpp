/**
 * @file Shallow network implementation.
 */
#pragma once

#include <cstddef>

#include "ml/neural_network/interface.hpp"
#include "ml/types.hpp"

// clang-format off
/** Dense layer interface. */
namespace ml::dense_layer { class Interface; }
// clang-format on

namespace ml::neural_network
{
/**
 * @brief Shallow network implementation.
 *
 *        This class is non-copyable and non-movable.
 */
class Shallow final : public Interface
{
public:
    /**
     * @brief Constructor.
     *
     * @param[in] hiddenLayer The network's single hidden layer.
     * @param[in] outputLayer The network's output layer.
     * @param[in] trainInput Training input values. Must not be empty.
     * @param[in] trainOutput Training output values. Must not be empty.
     *
     * @note The node count of the hidden layer must match the weight count of the output layer.
     */
    explicit Shallow(dense_layer::Interface& hiddenLayer, dense_layer::Interface& outputLayer,
                     const Matrix2d& trainInput, const Matrix2d& trainOutput) noexcept;

    /**
     * @brief Destructor.
     */
    ~Shallow() noexcept override = default;

    /**
     * @brief Predict based on the given input.
     *
     * @param[in] input Input to base the prediction on.
     *
     * @return Predicted output.
     */
    [[nodiscard]] const Matrix1d& predict(const Matrix1d& input) noexcept override;

    /**
     * @brief Train the network.
     *
     * @param[in] epochCount Number of epochs to train the network.
     * @param[in] precisionThreshold Threshold, we terminate the training when reached.
     *
     * @return True if training was performed, false if the input arguments were invalid.
     */
    bool train(std::size_t epochCount, double precisionThreshold = 0.999999) noexcept;

    Shallow()                          = delete; // No default constructor.
    Shallow(const Shallow&)            = delete; // No copy constructor.
    Shallow(Shallow&&)                 = delete; // No move constructor.
    Shallow& operator=(const Shallow&) = delete; // No copy assignment.
    Shallow& operator=(Shallow&&)      = delete; // No move assignment.

private:
    void initTrainOrder(std::size_t setCount) noexcept;
    void randomizeTrainOrder() noexcept;
    bool feedforward(const Matrix1d& input) noexcept;
    bool backpropagate(const Matrix1d& reference) noexcept;
    bool optimize(const Matrix1d& input, double learningRate) noexcept;
    [[nodiscard]] double precision() noexcept;

    /** Training order indexes. */
    MatrixU32 myTrainOrder;

    /** The network's single hidden layer. */
    dense_layer::Interface& myHiddenLayer;

    /** The network's output layer. */
    dense_layer::Interface& myOutputLayer;

    /** Input training data. */
    const Matrix2d& myTrainInput;

    /** Output training data. */
    const Matrix2d& myTrainOutput;
};
} // namespace ml::neural_network
