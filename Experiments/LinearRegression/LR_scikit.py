"""
Script for preformin linear regression using Scikit
"""

import pandas as pd
from sklearn.linear_model import LinearRegression

from config import DATA_PATH

train = pd.read_csv(DATA_PATH / "insurance_train.csv")
test = pd.read_csv(DATA_PATH / "insurance_test.csv")

y_train, X_train = train.charges, train.drop(columns=["charges"])
y_test, X_test = test.charges, test.drop(columns=["charges"])

reg = LinearRegression().fit(X_train, y_train)

print(f"Final parameters: {reg.coef_}")
print(f"Train score: {reg.score(X_train, y_train)}")
print(f"Test score: {reg.score(X_test, y_test)}")
