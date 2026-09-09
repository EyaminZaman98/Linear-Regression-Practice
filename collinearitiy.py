import pandas as pd
import numpy as np 
import seaborn as sns
from pprint import pprint 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression

df = pd.read_csv("Credit.csv")

print("First 5 rows of the dataset: ")
print(df.head())

X = df.drop("Balance", axis = 1)
X = pd.get_dummies(X, drop_first = True)
y = df["Balance"]

print("\nColumns after dummy encoding: ")
print(X.columns)

linear_coef = []

# Loop over all predictors
for i in X:
    x = X[[i]]
    model = LinearRegression()
    model.fit(x, y)
    linear_coef.append(model.coef_)

# Multiple Linear Regression
multi_model = LinearRegression()
multi_model.fit(X, y)
multi_coef = multi_model.coef_

print("\nBy simple (one variable) linear regression for each variable: ")
for i in range(len(X.columns)):
    pprint(f"Value of beta{i+1} = {linear_coef[i][0]:.2f}")

print("\nBy multi-linear regression on all variable:")
for i in range(len(X.columns)):
    pprint(f"Value of beta{i+1} = {multi_coef[i]:.2f}")

# Correlation Matrix
corrMatrix = X.corr()
print("\nCorrelation Matrix:")
print(corrMatrix)

# Heatmap
sns.heatmap(corrMatrix , annot = True)
plt.title("Correlation Heatmap")
plt.show()

