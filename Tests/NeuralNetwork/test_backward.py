"""
Script for testing elements of backpropagation in my network
"""

import numpy as np
import pytest

from Models.NeuralNetwork.neural_network import NeuralNetwork


@pytest.fixture
def x():
    """
    shaape (3,) sample input
    Returns:

    """
    return np.array([1, 1, 1])


@pytest.fixture
def y():
    """
    shape (1,) sample network output
    Returns:

    """
    return np.array([1])


@pytest.fixture
def deltas():
    """
    List of calculated deltas for each node in the network
    Returns:

    """
    return [np.array([-0.25, 0.25]), np.array([-0.125])]


@pytest.fixture
def gradients_w():
    """
    List of calculated gradients for each weight in the network
    Returns:

    """
    return [
        np.array([[-0.25, 0.25], [-0.25, 0.25], [-0.25, 0.25]]),
        np.array([[-1 / 16], [-3 / 16]]),
    ]


@pytest.fixture
def NN():
    """
    Neural network with hard coded initial parameters
    Returns:

    """
    network = NeuralNetwork(3, [2, 1])
    network.layers[0].weights = np.array([[0, 0.5], [0, 0.5], [0, 0.5]])
    network.layers[0].biases = np.array([0.5, 0])

    network.layers[1].weights = np.array([[2], [-2]])
    network.layers[1].biases = np.array([2])

    return network


def test_deltas(NN, x, y, deltas):
    """
    Tests whether the network calculates the deltas properly
    Args:
        NN ():
        x ():
        y ():
        deltas ():

    Returns:

    """
    output = NN.forward_prop(x)
    NN._calculate_deltas(output - y)
    for layer, delta in zip(NN.layers, deltas):
        assert (layer.deltas == delta).all()


def test_gradient(NN, x, y, deltas, gradients_w):
    """
    Tests whether the network calculates the gradients correctly for each parameter
    Args:
        NN ():
        x ():
        y ():
        deltas ():
        gradients_w ():

    Returns:

    """
    output = NN.forward_prop(x)
    NN.calculate_gradients(output - y, x)

    for layer, grad_w in zip(NN.layers, gradients_w):
        assert (layer.gradients_b == layer.deltas).all()
        assert (layer.gradients_w == grad_w).all()
