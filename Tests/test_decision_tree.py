import numpy as np
import pytest

import Models.Trees.utils as utils


@pytest.fixture
def X():
    """
    Sample data
    """
    np.random.seed(42)
    X = np.random.rand(4, 3)
    X.T[1] = np.array([1 if x >= 0.6 else 0 for x in X.T[1]])
    X.T[2] = np.array([1 if x >= 0.8 else 0 for x in X.T[2]])
    return X


@pytest.fixture
def y(X):
    """
    Sample outputs
    """
    return np.array([1 if x[1] == 1 else 0 for x in X])


@pytest.mark.parametrize(
    "input_, output",
    [(np.array([]), 0.0), (np.array([0]), 0.0), (np.array([0, 1, 1, 0]), 1.0)],
)
def test_entropy(input_, output):
    """
    Tests entropy calculations
    """
    assert utils.compute_binary_entropy(input_) == output


@pytest.mark.parametrize(
    "node_indices, output, feature",
    [
        ([], ([], []), 1),
        ([0, 1, 2], ([0, 2], [1]), 1),
        ([1, 3], ([], [1, 3]), 1),
        ([1, 3], ([3], [1]), 2),
        ([0, 1, 2, 3], ([0, 2], [1, 3]), 1),
        ([0, 1, 2, 3], ([3], [0, 1, 2]), 2),
    ],
)
def test_binary_split(X, node_indices, output, feature):
    """
    Tests impllemented binary split
    """
    assert utils.binary_split_dataset(X, node_indices, feature) == output


@pytest.mark.parametrize(
    "node_indices, feature, output",
    [
        ([], 1, 0.0),
        ([0, 1, 2, 3], 1, 1.0),
        ([0, 1, 3], 2, 0.2516291673878228),
        ([0, 1, 2, 3], 2, 0.31127812445913283),
    ],
)
def test_information_gain(X, y, node_indices, feature, output):
    """
    Test information gain function
    """
    assert utils.compute_information_gain(X, y, node_indices, feature) == output


@pytest.mark.parametrize(
    "node_indices, output", [([], -1), ([0, 1], 0), ([0, 1, 3], 0), ([0, 1, 2, 3], 0)]
)
def test_best_split(X, y, node_indices, output):
    """
    Tests implementation of best split
    """
    assert utils.get_best_split(X[:, [1, 2]], y, node_indices) == output
