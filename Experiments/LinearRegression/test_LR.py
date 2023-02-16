"""
Script for testing Linear Regression (LR)

It's very simplified, the step size is constant
"""

import os

import numpy as np
import pandas as pd
import plotly.express as px

import Models.linear_regression as linear_regression
from config import DATA_PATH, EPERIMENTS_PATH

if __name__ == "__main__":
    # Get data, create numeric columns or drop them
    data = pd.read_csv(DATA_PATH / "insurance.csv")
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    data.sex = data.sex.apply(lambda val: 1 if val == "female" else 0)
    data.smoker = data.smoker.apply(lambda val: 1 if val == "yes" else 0)

    y = data.charges
    data.drop(columns=["region", "charges"], inplace=True)
    X = data.to_numpy()

    # Normalise inputs
    for i in range(X.shape[1]):
        feat = X[:, i]
        X[:, i] = (feat - np.mean(feat)) / np.std(feat)
    y_mean = np.mean(y)
    y_std = np.std(y)
    # y = (y - y_mean) / y_std

    size = X.shape[0]
    split_point = int(0.8 * size)
    y_train, y_test = y[:split_point], y[:split_point]
    X_train, X_test = X[:split_point], X[:split_point]

    # Initialise parameters
    w = np.random.random(X_train.shape[1]).T
    b = 0.0

    LR = linear_regression.LinearRegressor()

    step_size = 1e-1
    num_iters = int(1e3)

    w, b, J, ws = LR.gradient_descent(X_train, y_train, w, b, step_size, num_iters)
    with open(EPERIMENTS_PATH / "LinearRegression/parameters.txt", "w") as file:
        np.savetxt(file, np.concatenate((w, np.array([b]))))

    predictions_train = LR.predict(X_test, w, b)
    prediction_test = LR.predict(X_test, w, b)

    print(f"R^2 on test: {LR.get_R2(y_train, predictions_train)}")
    print(f"Loss on train: {LR.get_R2(y_test, prediction_test)}")

    for i in range(5):
        # print(f"Predicted cost: {LR.predict(X_test[i], w, b)*y_std + y_mean}, real cost: {y_test[i]*y_std + y_mean}")
        print(f"Predicted cost: {LR.predict(X_test[i], w, b)}, real cost: {y_test[i]}")

    fig = px.scatter(x=np.linspace(1, len(J), len(J)), y=J, log_x=True, log_y=True)
    fig.write_image(EPERIMENTS_PATH / "LinearRegression/LR_cost.png")
