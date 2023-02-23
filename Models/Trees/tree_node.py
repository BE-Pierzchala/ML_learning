"""
Implementation for tree node used for building decision trees
"""
import numpy as np

import Models.Trees.utils as utils

class Node:

    def __init__(self, X: np.ndarray, y: np.ndarray, node_indices: list, parent = None, max_depth = 3):
        self.parent = parent
        self.split_feature = utils.get_best_split(X, y, node_indices)

        if max_depth or len(node_indices) > 1:
            left_nodes, right_nodes = utils.binary_split_dataset(X, node_indices, self.split_feature)
            self.left_child = Node(X, y, left_nodes, parent = self, max_depth = max_depth - 1)
            self.right_child = Node(X, y, right_nodes, parent = self, max_depth = max_depth - 1)
        else:
            self = node_indices
