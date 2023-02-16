"""
Class for linear regression
"""

import numpy as np
from typing import List
import tqdm

class LinearRegressor:
    def __init__(self):
        pass

    def predict(self, x: np.ndarray, weights: np.ndarray, bias: float) -> float:
        """
        Predicts expected y value for input x and parameters w,b
        Args:
            x ():
            weights ():
            bias ():

        Returns:
            (): expected y value
        """
        return np.dot(x, weights) + bias

    def compute_cost(self, X: np.ndarray, y: np.ndarray, weights: float, bias: float) -> float:
        """
        Method for calculating total cost using the square distance metric

        Args:
            X (): Shape(m,) dimensional input to the model
            y (): Shape(m,) Labels
            weights (): weights of the model
            bias (): bias of the model

        Returns:
            (flloat): total cost of using w,b as parameters for linear regression to fit X,y
        """
        size = X.shape[0]
        return sum([(self.predict(x_, weights, bias) - y_)**2 for (x_, y_) in zip(X, y)])/(2*size)

    def compute_gradient(self, X: np.ndarray, y: np.ndarray, weights: float, bias: float) -> tuple[np.ndarray, float]:
        """
        Computes the gradient for linear regression
        Args:
            X (): Shape(m,) dimensional input to the model
            y (): Shape(m,) Labels
            weights (): weights of the model
            bias (): bias of the model

        Returns:
            dj_dw: Gradient of the cost w.r.t w
            dj_db: Gradient of the cost w.r.t b
        """
        size = y.shape[0]

        dj_dw = sum([(self.predict(x_, weights, bias) - y_)*x_ for (x_, y_) in zip(X, y)])/size
        dj_db = sum([self.predict(x_, weights, bias) - y_ for (x_, y_) in zip(X, y)])/size

        return dj_dw, dj_db

    def gradient_descent(self,
                         X: np.ndarray,
                         y: np.ndarray,
                         weights_in: float,
                         bias_in: float,
                         step_size: float = 1e-3,
                         num_iters: int = 1000) -> tuple[np.ndarray, float, List[float]]:
        """
        Performs batch gradient descent to learn parameters of the model. Updates them by taking num_iters steps with
        learning rate steep_size.
        Args:
            X (): Shape(m,) dimensional input to the model
            y (): Shape(m,) Labels
            weights_in (): weights of the model
            bias_in (): bias of the model
            step_size (): learning rate
            num_iters (): number of steps for the model to take

        Returns:
            w: updated values of the parameter
            b: updated values of the parameter
            J: Cost history
        """
        J = []
        ws = []
        w = weights_in.copy()
        b = bias_in

        for i in tqdm.tqdm(range(num_iters), desc='Training'):
            dj_dw, dj_db = self.compute_gradient(X, y, w, b)

            w = w - step_size*dj_dw
            b = b - step_size*dj_db

            J.append(self.compute_cost(X, y, w, b))
            ws.append(w)

        print(f"Final cost: {J[-1]}")
        print(f"Final parameters: w {w}, b {b}")
        return w, b, J, ws
