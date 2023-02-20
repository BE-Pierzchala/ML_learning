"""
Script with utility functions for the Neural Network class
"""

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    """
    Calculates the Rectified Linear Unit (relu) for each element of the input
    Args:
        x (): (m,) shape vector

    Returns:
        (m,) shape vector
    """
    return np.array([max(0, element) for element in x])


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Calculates the sigmoid function for each element of the input
    Args:
        x (): (m,) shape vector

    Returns:
        (m,) shape vector
    """
    return 1 / (1 + np.exp(-x))


def relu_der(x: np.ndarray) -> np.ndarray:
    """
    Calculates the derivative of the Rectified Linear Unit (relu) function for each element of the input
    Args:
        x (): (m,) shape vector

    Returns:
        (m,) shape vector
    """
    return np.array([1 if y >= 0 else 0 for y in x])


def sigmoid_der(x: np.ndarray) -> np.ndarray:
    """
    Calculates the derivative of sigmoid function on each element of the input
    Args:
        x (): (m,) shape vector

    Returns:
        (m,) shape vector
    """
    return x * (1 - x)


def binary_cross_entropy(y: np.ndarray, predictions: np.ndarray) -> float:
    """
    Calculates the cost using binary cross-entropy function
    Args:
        y (): Expected outputs
        predictions (): Predicted outputs

    Returns:
        cost for the inputs
    """
    losses = -y * np.log(predictions + 1e-10) - (1 - y) * np.log(
        1 + 1e-10 - predictions
    )
    return np.sum(losses) / len(y)
