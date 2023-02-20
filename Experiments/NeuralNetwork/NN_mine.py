"""
Script for evaluating my neural network class, it plots and saves the loss function through thee training.
Prints the test training scores.
"""

import numpy as np
import pandas as pd
import plotly.express as px

from config import DATA_PATH, EPERIMENTS_PATH
from Models.NeuralNetwork.neural_network import NeuralNetwork

# Use the same binary classification data set
data_path = DATA_PATH / "LogisticRegression"


train = pd.read_csv(data_path / "adult_train.csv")
test = pd.read_csv(data_path / "adult_test.csv")

target = "income"

y_train, X_train = train[target], train.drop(columns=[target])
y_test, X_test = test[target], test.drop(columns=[target])

# My class is written in numpy arrays
y_train, X_train = y_train.to_numpy(), X_train.to_numpy()
y_test, X_test = y_test.to_numpy(), X_test.to_numpy()

NN = NeuralNetwork(X_train.shape[1], [5, 5, 1])

J = NN.fit(X_train, y_train, step_size=1e-1, num_epochs=int(1e2))

# Prints scores on test and train sets
print(f"Score on test: {NN.precision(y_train, X_train)}")
print(f"Score on train: {NN.precision(y_test, X_test)}")

fig = px.scatter(
    x=np.linspace(1, len(J), len(J)),
    y=J,
    log_x=True,
    log_y=True,
    labels={"x": "iteration", "y": "cost"},
)
fig.write_image(EPERIMENTS_PATH / "NeuralNetwork/cost.png")
