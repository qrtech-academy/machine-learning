"""Simple convolutional layer demo."""

import random

# pylint: disable = consider-using-enumerate

Matrix1d = list[float]
Matrix2d = list[Matrix1d]


def create_matrix2d(size: int) -> Matrix2d:
    """Create a two-dimensional matrix.

    Args:
        size: Size of the matrix.

    Returns:
        New matrix initialized with zeros.
    """
    matrix: Matrix2d = []
    for _ in range(size):
        row = [0.0] * size
        matrix.append(row)
    return matrix


def init_matrix2d(matrix: Matrix2d) -> None:
    """Initialize matrix with zeros.

    Args:
        matrix: The matrix to initialize.
    """
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            matrix[i][j] = 0.0


def relu_output(num: float) -> float:
    """Get ReLU output based on the given input.

    Args:
        num: ReLU input.

    Returns: The corresponding ReLU output.
    """
    return num if num > 0 else 0


def relu_delta(num: float) -> float:
    """Get ReLU delta based on the given input.

    Args:
        num: ReLU input.

    Returns: The corresponding ReLU delta.
    """
    return 1.0 if num > 0 else 0


class ConvLayer:
    """Convolutional layer implementation."""

    # pylint: disable = too-many-instance-attributes

    def __init__(self, input_size: int, kernel_size: int) -> None:

        if kernel_size == 0 or input_size == 0 or input_size < kernel_size:
            raise ValueError(
                "Cannot create convolutional layer: invalid input arguments!"
            )

        pad_offset = kernel_size // 2
        padded_size = input_size + 2 * pad_offset

        self.input_padded = create_matrix2d(padded_size)
        self.input_gradients_padded = create_matrix2d(padded_size)
        self.input_gradients = create_matrix2d(input_size)
        self.kernel = create_matrix2d(kernel_size)
        self.kernel_gradients = create_matrix2d(kernel_size)
        self.output = create_matrix2d(input_size)
        self.preactivation_output = create_matrix2d(input_size)
        self.bias = random.random()
        self.bias_gradient = 0.0

        for ki in range(len(self.kernel)):
            for kj in range(len(self.kernel)):
                self.kernel[ki][kj] = random.random()

    def feedforward(self, input_data: Matrix2d) -> bool:
        """Perform feedforward operation.

        Args:
            input_data: Input data.

        Returns:
            True on success, false on failure.
        """
        if len(input_data) != len(self.output):
            return False

        if not self._pad_input(input_data):
            return False

        # Perform convolution and apply the activation function.
        for i in range(len(self.output)):
            for j in range(len(self.output)):
                num = self.bias
                for ki in range(len(self.kernel)):
                    for kj in range(len(self.kernel)):
                        num += self.input_padded[i + ki][j + kj] * self.kernel[ki][kj]
                self.preactivation_output[i][j] = num
                self.output[i][j] = relu_output(num)
        return True

    def backpropagate(self, output_gradients: Matrix2d) -> bool:
        """Perform backpropagation.

        Args:
            output_gradients: Output gradients from the next layer.

        Returns:
            True on success, false on failure.
        """
        if len(output_gradients) != len(self.output):
            return False

        init_matrix2d(self.input_gradients_padded)
        init_matrix2d(self.input_gradients)
        init_matrix2d(self.kernel_gradients)
        self.bias_gradient = 0.0

        for i in range(len(self.output)):
            # Check if the output gradient matrix is square, return false if not.
            if len(output_gradients[i]) != len(output_gradients):
                return False
            for j in range(len(self.output)):
                delta = output_gradients[i][j] * relu_delta(
                    self.preactivation_output[i][j]
                )
                self.bias_gradient += delta
                for ki in range(len(self.kernel)):
                    for kj in range(len(self.kernel)):
                        self.input_gradients_padded[i + ki][j + kj] += (
                            self.kernel[ki][kj] * delta
                        )
                        self.kernel_gradients[ki][kj] += (
                            self.input_padded[i + ki][j + kj] * delta
                        )
        self._extract_input_gradients()
        return True

    def optimize(self, learning_rate: float) -> bool:
        """Perform optimization.

        Args:
            learning_rate: Learning rate used to adjust the trainable parameters.

        Returns:
            True on success, false on failure.
        """
        if learning_rate <= 0.0:
            return False

        self.bias += self.bias_gradient * learning_rate
        for ki in range(len(self.kernel)):
            for kj in range(len(self.kernel)):
                self.kernel[ki][kj] += self.kernel_gradients[ki][kj] * learning_rate
        return True

    def _pad_input(self, input_data: Matrix2d) -> bool:
        init_matrix2d(self.input_padded)
        offset = len(self.kernel) // 2

        # Copy the input data to the padded input matrix.
        for i in range(len(input_data)):
            # Check if the input matrix is square, return false if not.
            if len(input_data[i]) != len(input_data):
                return False
            for j in range(len(input_data)):
                self.input_padded[i + offset][j + offset] = input_data[i][j]
        return True

    def _extract_input_gradients(self) -> None:
        offset = len(self.kernel) // 2

        # Extract the input gradients from the corresponding padded matrix.
        for i in range(len(self.output)):
            for j in range(len(self.output)):
                self.input_gradients[i][j] = self.input_gradients_padded[i + offset][
                    j + offset
                ]


def print_comparison(label: str, computed: Matrix2d, expected: Matrix2d) -> None:
    """Print a computed matrix next to the expected one from appendix B.

    Args:
        label: Text describing what's being printed.
        computed: Matrix holding the computed values.
        expected: Matrix holding the expected values.
    """
    width = len(computed) * 8
    print(f"\n{label}:")
    print("   ", f"{'computed:':<{width}}", " expected (appendix B):")

    # Print the two matrices side by side, row by row.
    for computed_row, expected_row in zip(computed, expected):
        computed_text = " ".join(f"{num:7.2f}" for num in computed_row)
        expected_text = " ".join(f"{num:7.2f}" for num in expected_row)
        print("   ", computed_text, " ", expected_text)


def main() -> None:
    """Create and demonstrate a simple convolutional layer.

    The layer is driven with the same values as the hand-training example in L06's appendix B,
    so every result printed below can be compared against the numbers worked out by hand there.
    """

    # Input image from appendix B, resembling the digit 0 made up of ones.
    input_data = [
        [1, 1, 1, 1],
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [1, 1, 1, 1],
    ]

    # Gradients the max pooling layer sends back in appendix B.
    output_gradients = [
        [0, 10, 20, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 30, 0, 40],
    ]

    # The results appendix B works out by hand, used to check the computed ones.
    expected_output = [
        [1.3, 1.9, 1.9, 1.9],
        [1.7, 1.7, 1.1, 1.9],
        [1.7, 1.3, 0.5, 1.7],
        [1.7, 2.1, 1.9, 2.3],
    ]
    expected_input_gradients = [
        [6, 20, 16, 0],
        [0, 0, 0, 0],
        [6, 12, 8, 16],
        [18, 24, 24, 32],
    ]
    expected_kernel = [
        [0.23, 0.44],
        [0.70, 0.90],
    ]
    expected_bias = 0.6
    learning_rate = 0.001

    # Create a convolutional layer: 4x4 input, 2x2 kernel.
    conv_layer = ConvLayer(4, 2)

    # Use appendix B's kernel and bias instead of the randomized ones, so the results below are
    # the same on every run and can be compared against the hand-worked numbers.
    conv_layer.kernel = [[0.2, 0.4], [0.6, 0.8]]
    conv_layer.bias = 0.5

    print("Convolution input data (2D):")
    for row in input_data:
        print("  ", " ".join(f"{num:.1f}" for num in row))

    if not conv_layer.feedforward(input_data):
        print("Feedforward failed, aborting program!")
        return
    print_comparison("Convolution output (2D)", conv_layer.output, expected_output)

    print("\nConvolution output gradients (2D):")
    for row in output_gradients:
        print("  ", " ".join(f"{num:.1f}" for num in row))

    if not conv_layer.backpropagate(output_gradients):
        print("Backpropagation failed, aborting program!")
        return
    print_comparison(
        "Input gradients after backpropagation (2D)",
        conv_layer.input_gradients,
        expected_input_gradients,
    )

    # Perform optimization, which adjusts the kernel and bias using the computed gradients.
    if not conv_layer.optimize(learning_rate):
        print("Optimization failed, aborting program!")
        return
    print_comparison(
        "Kernel after optimization (2D)", conv_layer.kernel, expected_kernel
    )
    print("\nBias after optimization:")
    print(
        f"    computed: {conv_layer.bias:.2f}, expected (appendix B): {expected_bias:.2f}"
    )


# Invoke the main function if this is the startup script.
if __name__ == "__main__":
    main()
