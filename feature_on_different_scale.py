import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

df = pd.read_csv("Advertising.csv" , index_col = 0)
print(df.head())

x = df.drop("sales" , axis = 1)
y = df["sales"]

model = LinearRegression().fit(x,y)

print(f'{"Model Coefficients":>9}')

for col , coef in zip(x.columns , model.coef_):
    print(f'{col:>9}: {coef:>6.3f}')

print(f'\nR^2: {model.score(x,y):.4}')

#---------------------------

# Scale the entire DataFrame by 1000

#---------------------------

df *= 1000
print(df.head())

x = df.drop("sales" , axis = 1)
y = df["sales"]

model2 = LinearRegression().fit(x , y)

for col, coef in zip(x.columns , model2.coef_):
    print(f"{col:>9}: {coef:>6.3f}")

print(f"\nR^2 score : {model2.score(x , y):.4}")

#......................................................

# Visualize coefficients

#......................................................
plt.figure(figsize = (8,3))
cols = x.columns
coefs = model2.coef_
plt.barh(cols,coefs)
plt.axvline(0 , c = "k" , ls = '--' , alpha = 0.5)
plt.ylabel("Predictor")
plt.xlabel("Coefficient Values")
plt.title(
    "Coefficients of Linear Model Predicting Sales\n"
    "from Newspaper, Radio, and TV Advertising Budgets(in Dollar)"
)
plt.show()

#---------------------------------------------
# Create x2 with different currencies
#---------------------------------------------

x2 = pd.DataFrame()

x2["TV (Rupee)"] = 200 * df["TV"]
x2["radio (Won)"] = 1175 * df["radio"]
x2["newspaper (Cedi)"] = 6 * df["newspaper"]

model3 = LinearRegression().fit(x2 , y)
print(f"{"Model Coefficients":>16}")

for col , coef in zip(x2.columns , model3.coef_):
    print(f"{col:>16}: {coef:>8.5f}")

print(f"\nR^2: {model3.score(x2, y):.4}")

#-------------------------------------------------
# Visualize coefficients with different currencies
#-------------------------------------------------

plt.figure(figsize = (8,3))
plt.barh(x2.columns , model3.coef_)
plt.axvline(0 , c = "k" , ls = "--" , alpha = 0.5)
plt.ylabel("Predictor")
plt.xlabel("Coefficient Values")
plt.title(
    "Coefficients of Linear Model Predicting Sales\n"
    "from Newspaper, Radio, and TV Advertising Budgets(Different Currencies)"
)
plt.show()

#------------------------------------------------------------------
# Compare both coefficient plots using shared x-axis
#------------------------------------------------------------------
fig , axes = plt.subplots( 2, 1, figsize = (8,6) , sharex = True)
axes[0].barh(x.columns , model.coef_)
axes[0].set_title("Dollars")

axes[1].barh(x2.columns , model3.coef_)
axes[1].set_title("Different Currencies")
for ax in axes:
    ax.axvline(0 , c = "k" , ls = "--" , alpha = 0.5)

axes[0].set_ylabel("Predictor")
axes[1].set_xlabel("Coefficient Values")
plt.show()

