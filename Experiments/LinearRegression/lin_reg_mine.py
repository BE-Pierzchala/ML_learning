"""
Script for testing Linear Regression (LinearRegression)

It's very simplified, the step size is constant
"""

import numpy as np
import pandas as pd
import plotly.express as px

import Models.linear_regression as linear_regression
from config import DATA_PATH, EPERIMENTS_PATH

if __name__ == "__main__":
    data_path = DATA_PATH / "LinearRegression"

    train = pd.read_csv(data_path / "insurance_train.csv")
    test = pd.read_csv(data_path / "insurance_test.csv")

    target = "charges"

    # My class is written in numpy arrays
    y_train, X_train = train[target], train.drop(columns=[target])
    y_test, X_test = test[target], test.drop(columns=[target])

    y_train, X_train = y_train.to_numpy(), X_train.to_numpy()
    y_test, X_test = y_test.to_numpy(), X_test.to_numpy()

    # Initialise parameters
    LR = linear_regression.LinearRegressor()
    J = LR.fit(X_train, y_train, step_size=1e-2, num_iters=int(1e3), lambda_=0.1)

    with open(EPERIMENTS_PATH / "LinearRegression/parameters.txt", "w") as file:
        np.savetxt(file, np.concatenate((LR.weights, np.array([LR.bias]))))

    print(f"Score on test: {LR.score(y_train, X_train)}")
    print(f"Score on train: {LR.score(y_test, X_test)}")

    fig = px.scatter(
        x=np.linspace(1, len(J), len(J)),
        y=J,
        log_x=True,
        log_y=True,
        labels={"x": "iteration", "y": "cost"},
    )
    fig.write_image(EPERIMENTS_PATH / "LinearRegression/cost.png")
