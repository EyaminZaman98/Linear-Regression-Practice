import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


df = None


def set_dataframe(data):
    global df
    df = data


def fit_and_plot_linear(x):

    # Response variable
    y = df["sales"]

    # Split data
    x_train, x_test, y_train, y_test = train_test_split( x , y , train_size=0.6 , random_state=42 )

    # Create model
    model = LinearRegression()

    # Train model
    model.fit(x_train, y_train)

    # Predictions
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # R-squared
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    # Plot
    plt.figure(figsize=(8, 5))

    plt.scatter( x_train , y_train , label="Train Data" , color="k" , alpha=0.6 )

    plt.scatter( x_test, y_test, label="Test Data", color="c", alpha=0.4)

    # Sort X values
    x_sorted = np.sort(x.values, axis=0)

    # Predict sorted X values
    y_sorted = model.predict(x_sorted)

    # Regression line
    plt.plot( x_sorted, y_sorted, label="Linear Regression" )

    plt.xlabel(x.columns[0])
    plt.ylabel("sales")
    plt.title(f"{x.columns[0]} vs sales")

    plt.legend()
    plt.show()

    return r2_train, r2_test


def fit_and_plot_multi():

    # Predictors
    x = df[["TV", "radio", "newspaper"]]

    # Response
    y = df["sales"]

    # Split data
    x_train, x_test, y_train, y_test = train_test_split( x, y, train_size=0.6, random_state=42 )

    # Create model
    model = LinearRegression()

    # Train model
    model.fit(x_train, y_train)

    # Predictions
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # R-squared
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    return r2_train, r2_test