"""
Script with a Neural Network class, currently it only supports Dense layers, relu and sigmoid activation functions, 
flat (m,) shape vectors as input, but the layers are easy to add

sigmoid function is default for output layer
"""
from typing import List

import numpy as np

import Models.NeuralNetwork.dense_layer as dense_layer
import Models.NeuralNetwork.utils as utils


class NeuralNetwork:
    """
    My implementation of a neural network

        input - (m,) shape vector

        layer types - Dense

        activation functions - relu and sigmoid
    """

    def __init__(self, input_shape: int, layers_units: list):
        """
        Initialises the neural network with len(layers_units) layers, each with layer_units

        Args:
            input_shape (): shape of the (m,) input to the network
            layers_units (): len(layers_units) layers, each with layer_units
        """

        self.layers = [dense_layer.Dense(input_shape, layers_units[0])]
        last_input = layers_units[0]

        for units in layers_units[1:]:
            self.layers.append(dense_layer.Dense(last_input, units))
            last_input = units

        # manually set last layer's activation to sigmoid
        self.layers[-1].activation_fun = "sigmoid"

    def forward_prop(self, inputs: np.ndarray) -> np.ndarray:
        """
        Propagates the input through each layer
        Args:
            inputs (): (m,) shap input vector

        Returns:
            (m,) shape network output
        """

        for layer in self.layers:
            inputs = layer.calculate_activations(inputs)

        return inputs

    def _calculate_deltas(self, y_diff: np.ndarray) -> None:
        """
        Calculates the deltas for each layer, for the use of the recursive formula for backpropagation

        #TODO: write out the formula
        Args:
            y_diff (): (m,) shape difference of y - y_expected

        Returns:

        """

        last_layer = self.layers[-1]
        delta_last = y_diff * last_layer.derivative()
        last_layer.deltas = delta_last

        for i in range(len(self.layers[:-1])):
            layer = self.layers[-2 - i]
            delta_last = np.matmul(last_layer.weights, delta_last) * layer.derivative()
            last_layer = layer
            last_layer.deltas = delta_last

    def calculate_gradients(self, y_diff, x: np.ndarray) -> None:
        """
        Calculates gradients for each parameter
        Args:
            y_diff (): (m,) shape difference of y - y_expected
            x (): (m,) shape input to the network

        Returns:

        """
        # TODO: previous layer activations can be stored in layers
        self._calculate_deltas(y_diff)

        last_activations = x

        for layer in self.layers:
            layer.compute_gradient(last_activations)
            last_activations = layer.last_activations

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Predicts binary classes based on current weights
        Args:
            inputs (): (m,n) shape n data points

        Returns:
            (n,) shape vector with binary labels
        """
        outputs = [self.forward_prop(input_)[0] for input_ in inputs]
        return np.array([1 if a >= 0.5 else 0 for a in outputs])

    def precision(self, y_expected: np.ndarray, X: np.ndarray) -> float:
        """
        Calculates accuracy of the model
        Args:
            y_expected (): true labels
            X (): Shape(m,) dimensional input to the model

        Returns: accuracy of the model

        """
        vals = y_expected == self.predict(X)

        return round(np.count_nonzero(vals) / len(vals), 4)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        num_epochs: int = 10,
        step_size: float = 1e-1,
    ) -> List[float]:
        """
        Iteratively fits parameters of the model using backpropagation.

        Args:
            X (): (m,n) shape n data points
            y (): (n,) shape vector with {0,1} class for each data point
            num_epochs (): number of training epochs
            step_size (): step size

        Returns:
            Cost function for each epoch
        """
        J = []
        outputs = np.zeros(X.shape[0])

        for epoch in range(num_epochs):
            for i, x in enumerate(X):
                outputs[i] = self.forward_prop(x)
                self.calculate_gradients(np.array([outputs[i] - y[i]]), x)

            for layer in self.layers:
                layer.update_params(X.shape[0], step_size)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, cost = {utils.binary_cross_entropy(y, outputs)}")

            J.append(utils.binary_cross_entropy(y, outputs))

        return J
