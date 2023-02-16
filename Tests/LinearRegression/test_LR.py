"""
Script for testing Linear Regression (LR)

It's very simplified, the step size is constant
"""

import os

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import Models.linear_regression as linear_regression

if __name__ == "__main__":
    # Create data
    size = 300
    x = np.linspace(0, 2 * np.pi, size) + np.random.random(size) / 4

    y = np.sin(x) + (1 - 2 * np.random.random(size)) / 4
    X = np.array([x, x**3, x**5, x**7, x**9, x**11, x**13, x**15])

    # Normalise inputs
    for i, feat in enumerate(X):
        X[i] = (feat - np.mean(feat)) / np.std(feat)
    y = (y - np.mean(y)) / np.std(y)
    X = X.T

    # Initialise parameters
    w = np.random.random(X.shape[1]).T
    b = 0.0

    LR = linear_regression.LinearRegressor()

    step_size = 3 * 1e-3
    num_iters = int(1e4)

    w, b, J, ws = LR.gradient_descent(X, y, w, b, step_size, num_iters)

    predictions = np.array([LR.predict(x_, w, b) for x_ in X])

    current_path = os.getcwd()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Data"))
    fig.add_trace(go.Scatter(x=x, y=predictions, mode="lines", name="Prediction"))

    fig.write_image(current_path + "/LR.png")

    fig = px.scatter(x=np.linspace(1, len(J), len(J)), y=J, log_x=True)
    fig.write_image(current_path + "/LR_cost.png")
