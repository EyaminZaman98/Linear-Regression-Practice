import pandas as pd
#import matplotlib.pyplot as plt

from helper import set_dataframe, fit_and_plot_linear, fit_and_plot_multi

# Read the CSV file
df = pd.read_csv("Advertising.csv")
print(df.head())

# Give dataframe to helper.py
set_dataframe(df)


# Create empty dataframe
df_results = pd.DataFrame( columns=["Predictor", "R2 Train", "R2 Test"] )


# Predictors
predictors = ["TV", "radio", "newspaper"]


# Simple Linear Regression
for predictor in predictors:

    r2_train, r2_test = fit_and_plot_linear(df[[predictor]])

    df_results.loc[len(df_results)] = [predictor, r2_train, r2_test]


# Multi-Linear Regression
r2_train, r2_test = fit_and_plot_multi()


# Store Multi-Linear Regression results
df_results.loc[len(df_results)] = [ "All", r2_train ,r2_test ]


# Print results
print("\nR-squared results:")
print(df_results)
