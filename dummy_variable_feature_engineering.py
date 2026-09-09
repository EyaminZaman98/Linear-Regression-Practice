import numpy as np 
import pandas as pd 
import seaborn as sns  # *
import matplotlib.pyplot as plt 

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the Credit data
df = pd.read_csv("Credit.csv")
print(df.head())
print("\nData types: ")
print(df.dtypes)

# Separate features (x) and response (y)
x = df.drop('Balance', axis = 1)   # *
y = df['Balance']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 42)

#Try fitting using all features
try:
    test_model = LinearRegression().fit(x_train, y_train)
except Exception as e:
    print("\nError!: ")
    print(e)


# Model 1: Numeric features only
numeric_features = ['Income', 'Limit', 'Rating', 'Cards', 'Age', 'Education']
model1 = LinearRegression().fit(x_train[numeric_features], y_train)

# Train R2
train_score = model1.score(x_train[numeric_features], y_train)

#Test R2
test_score = model1.score(x_test[numeric_features], y_test)
print("\nModel 1")
print(f"Train R2: {train_score}")
print(f"Test R2: {test_score}")

# Inspect categorical variables
categorical_features = ['Own', 'Student', 'Married', 'Region']

print("\nCategorical variables: ")
for feature in categorical_features:
    print( feature + " : " , list(x_train[feature].unique())) # *

# Create design matrices
# Convert categorical variables into dummy variables
# drop_first = True removes one category from each categorical
# variable to avoid redundant columns

x_train_design = pd.get_dummies( x_train, drop_first = True ) # *
x_test_design = pd.get_dummies( x_test, drop_first = True )

# Make sure test has exactly the same columns as train
x_test_design = x_test_design.reindex( columns = x_train_design.columns, fill_value = 0 )  # *
print("\nTraining design matrix: ")
print(x_train_design.head())
print("\nDesign matrix data types: ")
print(x_train_design.dtypes)

# Model 2 : Numerical + Categorical features
model2 = LinearRegression().fit(x_train_design, y_train)

# Train R2
train_score = model2.score(x_train_design, y_train)

# Test R2
test_score = model2.score(x_test_design, y_test)

print("\nModel 2")
print(f"Train R2: {train_score}")
print(f"Test R2: {test_score}")

coefs = pd.DataFrame( model2.coef_, index = x_train_design.columns, columns = ['beta_value']) # *

print("\nModel 2 coefficients: ")
print(coefs)

# Plot coefficients
plt.figure(figsize = (10, 5))

sns.barplot( data = coefs.T, orient = 'h' ) # *
plt.title("Model Coefficients")
plt.tight_layout()
plt.show()


# Find most important categorical feature

# get dummy-variable coefficients only
categorical_columns = [ column for column in x_train_design.columns if any(column.startswith(feature + '_') for feature in categorical_features)] # *
categorical_coefs = coefs.loc[categorical_columns]   # *
print("\nCategorical coefficients: ")
print(categorical_coefs)

# find the categorical dummy with largest absolute coefficient
most_important_dummy = categorical_coefs['beta_value'].abs().idxmax() # *
print("\nMost important categorical dummy: " + most_important_dummy)

# Determine the original categorical feature
if most_important_dummy.startswith('Own_'): # *
    best_cat_feature = 'Own'
elif most_important_dummy.startswith('Student_'):
    best_cat_feature = 'Student'
elif most_important_dummy.startswith('Married_'):
    best_cat_feature = 'Married'
elif most_important_dummy.startswith('Region_'):
    best_cat_feature = 'Region'

print("Most important categorical feature: "+ best_cat_feature)

best_dummy_columns = [ column for column in x_train_design.columns if column.startswith(best_cat_feature + '_')] # *
print(f"Dummy columns for {best_cat_feature} : {best_dummy_columns}")
best_dummy = best_dummy_columns[0]

print(f"Using: {best_dummy}")

features = ['Income', best_dummy]

model3 = LinearRegression().fit(x_train_design[features], y_train)

beta0 = model3.intercept_

beta1 = model3.coef_[features.index('Income')]

beta2 = model3.coef_[features.index(best_dummy)]

coefs_model3 = pd.DataFrame([beta0, beta1, beta2], index = ['intercept', 'Income', best_dummy] , columns = ['beta_value'])

print("\nModel 3 coefficients: ")
print(coefs_model3)


# Plot Model 3 coefficients
plt.figure(figsize = (8,4))
sns.barplot(data = coefs_model3.T, orient = 'h')
plt.title("Model 3 Coefficients")
plt.tight_layout()
plt.show()

x_space = np.linspace(x['Income'].min() , x['Income'].max(), 1000)

# Balance = beta0 + beta1 * Income + beta2 * 1
y_hat_yes = (beta0 + beta1*x_space + beta2*1)

# Balance = beta0 + beta1 * Income + beta2 * 0
y_hat_no = ( beta0 + beta1*x_space + beta2*0)

# Plot Prediction Lines
plot_data = pd.concat([x_train_design, y_train], axis = 1)
plt.figure(figsize = (10, 6))

ax = sns.scatterplot( data = plot_data, x = "Income", y = "Balance", hue = best_dummy , alpha = 0.8)

ax.plot( x_space, y_hat_no, label = "0")
ax.plot( x_space, y_hat_yes, label = "1")

plt.title(f"Balance vs Income by {best_cat_feature}")
plt.xlabel("Income")
plt.ylabel("Balance")
plt.legend()
plt.tight_layout()
plt.show()
