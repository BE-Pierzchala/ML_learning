"""
Script for evaluating my Logistic Regressor
"""

import numpy as np
import pandas as pd
import plotly.express as px

import Models.logistic_regression as logistic_regression
from config import DATA_PATH, EPERIMENTS_PATH

data_path = DATA_PATH / "LogisticRegression"


train = pd.read_csv(data_path / "adult_train.csv")
test = pd.read_csv(data_path / "adult_test.csv")

target = "income"

# My class is written in numpy arrays
y_train, X_train = train[target], train.drop(columns=[target])
y_test, X_test = test[target], test.drop(columns=[target])

y_train, X_train = y_train.to_numpy(), X_train.to_numpy()
y_test, X_test = y_test.to_numpy(), X_test.to_numpy()

LR = logistic_regression.LogisticRegressor()

J = LR.fit(X_train, y_train, step_size=3 * 3 * 1e-2, num_iters=int(1e3))

# Save models parameters
with open(EPERIMENTS_PATH / "LogisticRegression/parameters.txt", "w") as file:
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
fig.write_image(EPERIMENTS_PATH / "LogisticRegression/cost.png")
