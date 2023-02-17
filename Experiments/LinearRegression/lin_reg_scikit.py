"""
Script for preformin linear regression using Scikit
"""

import pandas as pd
from sklearn.linear_model import LinearRegression

from config import DATA_PATH

data_path = DATA_PATH / "LinearRegression"

train = pd.read_csv(data_path / "insurance_train.csv")
test = pd.read_csv(data_path / "insurance_test.csv")

target = "charges"

y_train, X_train = train[target], train.drop(columns=[target])
y_test, X_test = test[target], test.drop(columns=[target])

reg = LinearRegression().fit(X_train, y_train)

print(f"Final parameters: {reg.coef_}")
print(f"Train score: {reg.score(X_train, y_train)}")
print(f"Test score: {reg.score(X_test, y_test)}")
