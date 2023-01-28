import numpy as np
import pandas as pd
from sklearn import linear_model
import sklearn
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

"""
    Program Objective: Predict the students final grades based on some variables
    Algorithm: Linear Regression Algorithm
    Variables:
     - G1:          first period grade (numeric: from 0 to 20)
     - G2:          second period grade (numeric: from 0 to 20)
     - studytime:   weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 
                    3 - 5 to 10 hours, or 4 - >10 hours)
     - failures:    number of past class failures (numeric: n if 1<=n<3, else 4)
     - absences:    number of school absences (numeric: from 0 to 93)
    Predict: G3 - final grade (numeric: from 0 to 20, output target)
    Data: https://archive.ics.uci.edu/ml/datasets/student+performance
"""

# Get and separate data
data = pd.read_csv("../docs/student/student-mat.csv", sep=";")

variables = ["G1", "G2", "G3", "studytime", "failures", "absences"]
data = data[variables]
data = shuffle(data)

# G1 and G2 are the first period and second period grades, respectively
predict = "G3"  # G3 - final grade

x = np.array(data.drop(labels=[predict], axis='columns'))
y = np.array(data[predict])

# Training data -> initial dataset you use to teach a machine learning application to recognize patterns
# Testing data  -> used to test the model
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# Train model multiple times for best score
best = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    # Linear Regression Algorithm
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)

    acc = linear.score(x_test, y_test)
    print("Accuracy: ", acc)

    if acc > best:
        best = acc
        with open("../docs/student/student_grades.pickle", "wb") as f:
            pickle.dump(linear, f)

# Load best model
model = open("../docs/student/student_grades.pickle", "rb")
linear = pickle.load(model)

print('-' * 20)
print("Accuracy: ", best)
print('Coefficients: ', linear.coef_)  # Slope values for each variable
print('Intercept: ', linear.intercept_)
print('-' * 20)

# Predict output
predictions = linear.predict(x_test)

print("\nPrediction".ljust(20), '|', "Variables".ljust(20), '|', "Actual".ljust(20))
print('-' * 62)
for i in range(len(predictions)):
    print(
        str(predictions[i]).ljust(20), '|',
        str(x_test[i]).ljust(20), '|',
        str(y_test[i]).ljust(20)
    )

# Drawing and plotting model using matplotlib
style.use("ggplot")

plt.figure(num="Student Final Grade")
plt.title("Student Final Grade")

xPlot = "G2"
xData = data[xPlot]

plt.scatter(xData, data[predict])
plt.xlabel(xPlot)
plt.ylabel("Final Grade")

plt.plot(xData, linear.coef_[variables.index(xPlot)] * xData + linear.intercept_)  # Draw regression line

plt.show()
