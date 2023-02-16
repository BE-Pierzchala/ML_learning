"""
Script for testing Linear Regression (LR)
"""

import os
import numpy as np
import Models.linear_regression as linear_regression
import plotly.graph_objects as go
import plotly.express as px

if __name__ == '__main__':

    # Create data
    size = 100
    x = np.linspace(0, 2, size)
    y = x - 4*x**2 + 0.1*x**3 + (1 - 2 * np.random.random(size))/4

    X = np.array([x, x**2, x**3]).T

    # Initialise parameters
    w = np.random.random(X.shape[1]).T
    b = 0.

    LR = linear_regression.LinearRegressor()

    step_size = 1e-3
    num_iters = int(1e3)

    w, b, J, ws = LR.gradient_descent(X, y, w, b, step_size, num_iters)

    predictions = np.array([LR.predict(x_, w, b) for x_ in X])

    current_path = os.getcwd()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x = x, y = y, mode='markers', name = 'Data'))
    fig.add_trace(go.Scatter(x=x, y=predictions, mode='lines', name = 'Prediction'))

    fig.write_image(current_path + "/LR.png")

    fig = px.scatter(x = np.linspace(1, num_iters, num_iters), y = J)
    fig.write_image(current_path + "/LR_cost.png")