"""
Class for logistical regression
"""

from typing import List

import numpy as np
import tqdm


class LogisticRegressor:
    """
    Class for logistical regression
    """

    def __init__(self):
        self.weights = None
        self.bias = None
        self.lambda_ = None

    def initialise(self, X: np.ndarray, lambda_: float) -> None:
        """
        Initialises model's parameters
        Args:
            X (): input vector
        """
        self.weights = np.random.random(X.shape[1]).T
        self.bias = 0.0
        self.lambda_ = lambda_

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Calculates the sigmoid function for binary classification
        Args:
            z (): predictions in range [0,1]

        Returns:

        """
        return 1 / (1 + np.exp(-z))

    def score(self, y_expected: np.ndarray, X: np.ndarray) -> float:
        """
        Calculates accuracy of the model
        Args:
            y_expected (): true labels
            X (): Shape(m,) dimensional input to the model

        Returns: accuracy of the modeel

        """
        vals = y_expected == self.predict(X)

        return round(np.count_nonzero(vals) / len(vals), 4)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts a label given an input
        Args:
            X (): Shape(m,) dimensional input to the model

        Returns:
            Predicted labels
        """

        predictions = self._predict(X)

        return [0 if pred < 0.5 else 1 for pred in predictions]

    def _predict(self, X: np.ndarray) -> float:
        """
        Predicts expected y (in [0,1] range) value for input x
        Args:
            X (): Shape(m,) dimensional input to the model

        Returns:
            (): expected y value
        """
        return self.sigmoid(np.dot(X, self.weights) + self.bias)

    def compute_cost(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Method for calculating total cost using the square distance metric

        Args:
            X (): Shape(m,) dimensional input to the model
            y (): Shape(m,) Labels

        Returns:
            (flloat): total cost of using w,b as parameters for linear regression to fit X,y
        """
        size = X.shape[0]

        predictions = self._predict(X)
        losses = -y * np.log(predictions) - (1 - y) * np.log(1 - predictions)

        return np.sum(losses) / size + self.lambda_ * np.sum(
            np.square(self.weights)
        ) / (2 * size)

    def compute_gradient(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """
        Computes the gradient for linear regression
        Args:
            X (): Shape(m,) dimensional input to the model
            y (): Shape(m,) Labels

        Returns:
            dj_dw: Gradient of the cost w.r.t w
            dj_db: Gradient of the cost w.r.t b
        """
        size = y.shape[0]

        dj_dw = (
            sum([(self._predict(x_) - y_) * x_ for (x_, y_) in zip(X, y)]) / size
            + self.lambda_ / size * self.weights
        )
        dj_db = sum([self._predict(x_) - y_ for (x_, y_) in zip(X, y)]) / size

        return dj_dw, dj_db

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        step_size: float = 1e-3,
        num_iters: int = 1000,
        lambda_: float = 0.0,
    ) -> np.ndarray:
        """
        Performs batch gradient descent to learn parameters of the model. Updates them by taking num_iters steps with
        learning rate steep_size.
        Args:
            lambda_ (): regularisation parameter
            X (): Shape(m,) dimensional input to the model
            y (): Shape(m,) Labels
            step_size (): learning rate
            num_iters (): number of steps for the model to take

        Returns:
            J: Cost history
        """
        self.initialise(X, lambda_)
        J = np.ones(num_iters)

        for i in tqdm.tqdm(range(num_iters), desc="Training"):
            dj_dw, dj_db = self.compute_gradient(X, y)
            self.weights = self.weights - step_size * dj_dw
            self.bias = self.bias - step_size * dj_db

            J[i] = self.compute_cost(X, y)

        print(f"Final parameters: w {self.weights}, b {self.bias}")
        return J
