"""

"""
from typing import List

import numpy as np


def compute_binary_entropy(y: np.ndarray) -> float:
    """
    Computes the binary cross entropy for an ensemble y

    Args:
       y (ndarray): (m,) shape vector with bin

    Returns:
        entropy (float): Entropy at that node

    """
    if len(y):
        p1 = np.mean(y)
        if 0 < p1 < 1:
            return -p1 * np.log2(p1) - (1 - p1) * np.log2(1 - p1)
    return 0.0


def binary_split_dataset(
    X, node_indices: list, feature: int
) -> tuple[List[int], List[int]]:
    """
    Splits the data at the given node into left and right branches

    Args:
        X (ndarray):             Data matrix of shape(n_samples, n_features)
        node_indices (list):     List containing the active indices. I.e, the samples being considered at this step.
        feature (int):           Index of feature to split on

    Returns:
        left_indices (list):     Indices with feature value == 1
        right_indices (list):    Indices with feature value == 0
    """

    left_indices = []
    right_indices = []

    for index in node_indices:
        if X[index, feature] == 1:
            left_indices.append(index)
        else:
            right_indices.append(index)

    return left_indices, right_indices


def compute_information_gain(X, y, node_indices, feature) -> float:
    """
    Compute the information of splitting the node on a given feature

    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.

    Returns:
        cost (float):        Cost computed

    """
    left_indices, right_indices = binary_split_dataset(X, node_indices, feature)

    y_node = y[node_indices]
    y_left = y[left_indices]
    y_right = y[right_indices]

    if not node_indices:
        return 0
    information_gain = 0

    left_w = len(left_indices) / len(node_indices)
    right_w = 1 - left_w

    information_gain = (
        compute_binary_entropy(y_node)
        - left_w * compute_binary_entropy(y_left)
        - right_w * compute_binary_entropy(y_right)
    )

    return information_gain


def get_best_split(X, y, node_indices):
    """
    Returns the optimal feature and threshold value
    to split the node data

    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.

    Returns:
        best_feature (int):     The index of the best feature to split
    """

    num_features = X.shape[1]

    best_feature = -1
    inf_gain = 0

    for feature in range(num_features):
        gain = compute_information_gain(X, y, node_indices, feature)

        if gain > inf_gain:
            best_feature = feature
            inf_gain = gain

    return best_feature
