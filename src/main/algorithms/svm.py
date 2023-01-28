import sklearn
from sklearn import svm, datasets, metrics

"""
    Program Objective: Classify each breast cancer case
    Algorithm: Support Vector Machines
    Variables: (alot)
    Predict: cancer class ('malignant', 'benign')
    Data: breast cancer data set from sklearn module
"""

# Load data
cancer = datasets.load_breast_cancer()
# print("Data Features: ", cancer.feature_names)
# print("Data Labels: ", cancer.target_names)
cancer_classes = ['malignant', 'benign']

# Split data
x = cancer.data  # Features
y = cancer.target  # Labels

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

# Training SVM
model = svm.SVC(kernel="linear")
model.fit(x_train, y_train)

# Testing model
predictions = model.predict(x_test)
acc = metrics.accuracy_score(y_test, predictions)
print("Accuracy: ", acc)

print("\nPrediction".ljust(10), '|', "Actual".ljust(10))
print('-' * 22)
for i in range(len(predictions)):
    print(
        cancer_classes[predictions[i]].ljust(10), '|',
        cancer_classes[y_test[i]].ljust(12)
    )
