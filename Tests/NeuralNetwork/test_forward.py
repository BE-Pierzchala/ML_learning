"""
Script for testing the Forward propagation in my neural network
"""

import numpy as np
import pytest

import Models.NeuralNetwork.utils as utils
from Models.NeuralNetwork.dense_layer import Dense
from Models.NeuralNetwork.neural_network import NeuralNetwork


@pytest.fixture
def x():
    """
    Single input to the network size (3,)
    Returns:

    """
    return np.array([1, 1, 1])


@pytest.fixture
def Xs():
    """
    Inputs to the network size (3,2)
    Returns:

    """
    return np.array([[1, 1, 1], [2, 2, 2]])


@pytest.fixture
def NN():
    """
    Neural network with hard coded initial parameters
    Returns:

    """
    network = NeuralNetwork(3, [2, 1])
    network.layers[0].weights = np.array([[1, 2], [1, 2], [1, 2]])
    network.layers[0].biases = np.array([3, 3])

    network.layers[1].weights = np.array([[3], [-2]])
    network.layers[1].biases = np.array([2])

    return network


@pytest.fixture
def rand_NN():
    """
    Randomly initialised Neural Network with the same architecture as NN
    Returns:

    """
    return NeuralNetwork(3, [2, 1])


def test_network(rand_NN, NN):
    """
    Tests whether randomly initialised network has the same architecture as the hard coded one
    Args:
        rand_NN ():
        NN ():

    Returns:

    """
    for l1, l2 in zip(rand_NN.layers, NN.layers):
        assert l1.weights.shape == l2.weights.shape
        assert l1.biases.shape == l2.biases.shape


@pytest.mark.parametrize(
    "input_", [(np.array([0])), (np.array([-10, -4, 2])), (np.array([0, 1]))]
)
def test_sigmoid(input_):
    """
    Tests the sigmoid implementation
    Args:
        input_ ():

    Returns:

    """
    assert (utils.sigmoid(input_) == 1 / (1 + np.exp(-input_))).all()


@pytest.mark.parametrize(
    "input_", [(np.array([0])), (np.array([-10, -4, 2])), (np.array([0, 1]))]
)
def test_relu(input_):
    """
    Tests the Rectified Linear Unit (relu) implementation
    Args:
        input_ ():

    Returns:

    """
    assert (
        utils.relu(input_) == np.array([max(0, element) for element in input_])
    ).all()


def test_forward_pass(x, NN):
    """
    Checks whether the network correctly calculates the activations
    Args:
        x ():
        NN ():

    Returns:

    """
    output = NN.forward_prop(x)

    assert (NN.layers[0].last_activations == utils.relu(np.array([6, 9]))).all()
    assert (output == utils.sigmoid(np.array([2]))).all()


def test_full_inputs(Xs, NN):
    """
    Tests whether the network has the correct input shape for multidimensional inputs
    Args:
        Xs ():
        NN ():

    Returns:

    """
    for x in Xs:
        assert x.shape[0] == NN.layers[0].weights.shape[0]
