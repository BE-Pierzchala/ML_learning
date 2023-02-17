"""
Script for evaluating Scikit's Logistic Regressor
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression

from config import DATA_PATH

data_path = DATA_PATH / "LogisticRegression"

train = pd.read_csv(data_path / "adult_train.csv")
test = pd.read_csv(data_path / "adult_test.csv")

target = "income"

y_train, X_train = train[target], train.drop(columns=[target])
y_test, X_test = test[target], test.drop(columns=[target])

reg = LogisticRegression().fit(X_train, y_train)

print(f"Final parameters: {reg.coef_}")
print(f"Train score: {reg.score(X_train, y_train)}")
print(f"Test score: {reg.score(X_test, y_test)}")
