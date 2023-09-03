""" Cited source:
    - https://www.kaggle.com/datasets/mirichoi0218/insurance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler  
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

# Dataset contains age, sex, BMI, how many children, if person smokes, region, and medical costs.
# We're attempting to predict the insurance costs.

# Read dataset using pandas.
data = pd.read_csv("datasets\insurance.csv")
print(data.head)

# Convert categorical features into numerical features, as ML models can't work with categorical figs
# (e.g., red, big, True, etc).
label_encoder = LabelEncoder()
data["sex"] = label_encoder.fit_transform(data["sex"])
data["smoker"] = label_encoder.fit_transform(data["smoker"])
data["region"] = label_encoder.fit_transform(data["region"])

# Select X and Y.
X = data[["age", "sex", "bmi", "children", "smoker", "region"]]
Y = data["charges"]

# Split data into train and test set.
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

# Build model.
model = LinearRegression()

# Scale the train and test values for X.
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Train linear regression model.
model.fit(x_train_scaled, y_train)

# Predict against x_test_scaled.
y_pred = model.predict(x_test_scaled)

# Compare the actual and predicted values.
compare = pd.DataFrame()
compare["Actual"] = y_test
compare["Predict"] = y_pred
compare["Compare"] = abs(y_test - y_pred)
print(compare)

# Print the evaluations.
mean = np.mean(y_pred, axis = 0)
std_devs = np.mean(y_pred, axis = 0)
mse = metrics.mean_squared_error(y_test, y_pred)
print("Average cost of the insurance: ", mean)
print("Standard deviation: ", std_devs)
print("Mean-squared error: ", mse)

# Show the plot graph.
plt.scatter(y_test, y_pred)
plt.title("y_test VS y_pred")
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()
