"""
Class for linear regression
"""

from typing import List

import numpy as np
import tqdm


class LinearRegressor:
    """
    Class for linear regression
    """

    def __init__(self):
        self.weights = None
        self.bias = 0
        self.lambda_ = None

    def initialise(self, X: np.ndarray, lambda_: float) -> None:
        """
        Initialises the models parameters
        Args:
            lambda_ (): parameter for regularisation
            X (): input vector

        Returns:

        """
        self.weights = np.random.random(X.shape[1]).T
        self.lambda_ = lambda_

    def score(self, y_expected: np.ndarray, X: np.ndarray) -> float:
        """
        Calculates the score for the model as R^2
        Args:
            y_expected (): expected values
            X (): Shape(m,) dimensional input to the model

        Returns:
            Score of the model
        """
        res = np.sum((y_expected - self.predict(X)) ** 2)
        tot = sum((y_expected - np.mean(y_expected)) ** 2)
        return 1 - res / tot

    def predict(self, X: np.ndarray) -> float:
        """
        Predicts expected y value for input x and parameters w,b
        Args:
            X (): Shape(m,) dimensional input to the model
        Returns:
            (): expected y value
        """
        return np.dot(X, self.weights) + self.bias

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
        cost = sum([(self.predict(x_) - y_) ** 2 for (x_, y_) in zip(X, y)]) / (
            2 * size
        )

        return cost + self.lambda_ / (2 * size) * np.sum(np.square(self.weights))

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
            sum([(self.predict(x_) - y_) * x_ for (x_, y_) in zip(X, y)]) / size
            + self.lambda_ / size * self.weights
        )
        dj_db = sum([self.predict(x_) - y_ for (x_, y_) in zip(X, y)]) / size

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
        Performs batch gradient descent to learn the parameters of the model. Updates them by taking num_iters steps
        with learning rate steep_size.
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
