""" Cited source:
    - https://www.kaggle.com/code/sadafpj/insurance-prediction-using-regression-regulation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler  
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Dataset contains age, sex, BMI, how many children, if person smokes, region, and medical costs.
# We're attempting to predict the insurance costs.

# Read dataset using pandas.
data = pd.read_csv("..\datasets\insurance.csv")
# print(data.head)

data_frame_insurance = pd.DataFrame(data)
# print(data_frame_insurance)
# print(data_frame_insurance.info)
# print(data_frame_insurance.describe)

# Show histogram of data.
# data.hist(bins = 60, figsize=(20, 15))
# plt.show()

# Convert categorical features into numerical features, as ML models can't work with categorical figs
# (e.g., red, big, True, etc). We want to encode the columns with the categorical values using one-hot encoding.
data_frame_insurance_converted = pd.get_dummies(data_frame_insurance, 
                                                columns = ['sex', 'smoker', 'region'], 
                                                drop_first = True)

data_frame_insurance_converted = data_frame_insurance_converted.astype(int)
# print(data_frame_insurance_converted)

cols = ['age', 'bmi', 'children', 'sex_male', 'smoker_yes', 'region_northwest', 'region_southeast', 'region_southwest']

# Select X and Y.
X = pd.DataFrame(data_frame_insurance_converted, columns = cols)
Y = data_frame_insurance_converted['charges']

# Split data into train and test set.
x_train, x_test, y_train, y_test = train_test_split(X, Y, 
                                                    train_size = 0.7, 
                                                    test_size = 0.3)

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