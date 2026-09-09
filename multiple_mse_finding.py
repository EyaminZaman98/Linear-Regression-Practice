import pandas as pd 

from prettytable import PrettyTable 
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("Advertising.csv")
mse_list = []

cols = [
    ["TV"],
    ["newspaper"],
    ['radio'],
    ["TV", "radio"],
    ["TV", "newspaper"],
    ["newspaper","radio"],
    ["TV","radio","newspaper"]
]

for i in cols:
    x = df[i]
    y = df["sales"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 0)

    model = LinearRegression()
    model.fit(x_train, y_train)

    y_test_pred = model.predict(x_test)
    MSE = mean_squared_error(y_test, y_test_pred)
    mse_list.append(MSE)

t = PrettyTable(["Predictors", "MSE"])

for i in range(len(mse_list)):
    t.add_row([cols[i], round(mse_list[i], 3)])

print(t)
