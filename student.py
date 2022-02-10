import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model

"""
    In this program the objective is to predict the final grade based on some variables.
    Variables used:
     - first period and second period grades;
     - study time
     - number of past class failures
     - number of school absences
     Data from: https://archive.ics.uci.edu/ml/datasets/student+performance
"""

# Get and separate data
data = pd.read_csv("docs/student/student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# G1 and G2 are the first period and second period grades, respectively
predict = "G3"  # G3 - final grade

x = np.array(data.drop(labels=[predict], axis='columns'))
y = np.array(data[predict])

# Training data -> initial dataset you use to teach a machine learning application to recognize patterns
# Testing data  -> used to test the model
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# Linear Regression Algorithm
linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)

print("Accuracy: ", linear.score(x_test, y_test))
print('Coefficients: ', linear.coef_)  # Slope values for each variable
print('Intercept: ', linear.intercept_)

# Predict output
predictions = linear.predict(x_test)

print("\nPrediction".ljust(20), '|', "Variables".ljust(20), '|', "Actual".ljust(20))
print('-' * 62)
for i in range(len(predictions)):
    print(str(predictions[i]).ljust(20), '|', str(x_test[i]).ljust(20), '|', str(y_test[i]).ljust(20))
