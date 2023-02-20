"""
My implementation of a dense layer - simplifies storing the variables and keeps the code cleaned

created for neural_network class
"""

from typing import Any

import numpy as np

import Models.NeuralNetwork.utils as utils


class Dense:
    """
    Dense layer for a Neural Network
    """

    def __init__(self, input_shape: int, num_units: int):
        """
        Dense layer class - it is initalises with random parameters in [0,1)

        Args:
            input_shape (): size of the input vector
            num_units (): number of neural units
        """
        self.num_units = num_units
        self.weights = np.random.rand(input_shape, num_units)
        self.biases = np.random.random(num_units)

        self.activation_fun = "relu"
        self.last_input = None
        self.last_activations = None
        self.deltas = None
        self.gradients_w = np.zeros((input_shape, num_units))
        self.gradients_b = np.zeros(num_units)

    def calculate_activations(self, input_: np.ndarray) -> np.ndarray:
        """
        Returns activations of this layer from the input
        Args:
            input_ (): Shape (m,) input vector

        Returns:
            Shape (num_units,) activations
        """
        match self.activation_fun:
            case "relu":
                activation_function = utils.relu
            case "sigmoid":
                activation_function = utils.sigmoid
            case _:
                raise ValueError(
                    "Input not supported - only relu and sigmoid activation functions are supported"
                )

        pre_activation = np.matmul(input_, self.weights) + self.biases
        self.last_activations = activation_function(pre_activation)

        return self.last_activations

    def derivative(self) -> Any:
        """
        Returns the derivative function for the type of activation in this layer
        Returns:
            derivative function
        """
        if self.activation_fun == "relu":
            return utils.relu_der(self.last_activations)
        else:
            return utils.sigmoid_der(self.last_activations)

    def compute_gradient(self, previous_layers_activations: np.ndarray) -> None:
        """
        Calculates the gradients for each parameter of the model, using a recursive formula
        and sums it with the previous gradients of the batch to average later

        See Also the function for deltas
        Args:
            previous_layers_activations (): (m,) shape activations from the previous layer
        Returns:

        """

        self.gradients_w = self.gradients_w + np.outer(
            previous_layers_activations, self.deltas
        )
        self.gradients_b = self.gradients_b + self.deltas

    def update_params(self, size: int, step_size: float) -> None:
        """
        Updates the parameters of this layer using mean gradients from the batch
        Args:
            size (): size of the batch
            step_size (): step size for the update

        Returns:

        """

        self.weights = self.weights - step_size * self.gradients_w / size
        self.biases = self.biases - step_size * self.gradients_b / size

        # Set parameters for zero
        self.gradients_w = self.gradients_w * 0
        self.gradients_b = self.gradients_b * 0
