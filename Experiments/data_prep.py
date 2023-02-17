"""
Script for changing categorical variables to numerical,
normalising features and splitting the data into train / test samples

Takes 3 positional arguments:
folder name
data file name
target variable
"""

import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config import DATA_PATH

folder = sys.argv[1]
filename = sys.argv[2]
target_var = sys.argv[3]

data_path = DATA_PATH / folder
data = pd.read_csv(data_path / filename)

data.dropna(inplace=True)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Make categorical columns numerical
cat_cols = data.select_dtypes(include=["object"]).columns.tolist()
data[cat_cols] = data[cat_cols].apply(LabelEncoder().fit_transform)


y = data[target_var]
X = data.drop(columns=[target_var])

# Scale
scaler = StandardScaler()
X[X.columns] = scaler.fit_transform(X)

X[target_var] = y

train, test = train_test_split(X, test_size=0.2, random_state=42)

fname = filename.split(".")[0]

train.to_csv(data_path / (fname + "_train.csv"), index=False)
test.to_csv(data_path / (fname + "_test.csv"), index=False)
