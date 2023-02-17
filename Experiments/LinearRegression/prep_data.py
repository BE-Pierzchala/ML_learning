"""
Script for normalising thea features and splitting the data into train / test samples
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import DATA_PATH

data = pd.read_csv(DATA_PATH / "insurance.csv")
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

data.sex = data.sex.apply(lambda val: 1 if val == "female" else 0)
data.smoker = data.smoker.apply(lambda val: 1 if val == "yes" else 0)

y = data.charges
X = data.drop(columns=["region", "charges"])

scaler = StandardScaler()
X[X.columns] = scaler.fit_transform(X)

X["charges"] = y

train, test = train_test_split(X, test_size=0.2, random_state=42)
print(len(train), len(test))

train.to_csv(DATA_PATH / "insurance_train.csv", index=False)
test.to_csv(DATA_PATH / "insurance_test.csv", index=False)
