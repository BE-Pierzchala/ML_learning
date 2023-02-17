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
    train = pd.read_csv(DATA_PATH / "insurance_train.csv")
    test = pd.read_csv(DATA_PATH / "insurance_test.csv")

    # My class is written in numpy arrays
    y_train, X_train = train.charges, train.drop(columns=["charges"])
    y_test, X_test = test.charges, test.drop(columns=["charges"])

    y_train, X_train = y_train.to_numpy(), X_train.to_numpy()
    y_test, X_test = y_test.to_numpy(), X_test.to_numpy()

    # Initialise parameters
    LR = linear_regression.LinearRegressor()
    J = LR.fit(X_train, y_train, step_size=1e-2, num_iters=int(1e3))

    with open(EPERIMENTS_PATH / "LinearRegression/parameters.txt", "w") as file:
        np.savetxt(file, np.concatenate((LR.weights, np.array([LR.bias]))))

    print(f"Score on test: {LR.score(y_train, X_train)}")
    print(f"Score on train: {LR.score(y_test, X_test)}")

    fig = px.scatter(x=np.linspace(1, len(J), len(J)), y=J, log_x=True, log_y=True)
    fig.write_image(EPERIMENTS_PATH / "LinearRegression/LR_cost.png")
