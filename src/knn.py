import sklearn
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt

"""
    Program Objective: Classify each car
    Algorithm: K-Nearest Neighbors
    Variables:
        - buying:   buying price (vhigh, high, med, low)
        - maint:    price of the maintenance (vhigh, high, med, low)
        - doors:    number of doors (2, 3, 4, 5more)
        - persons:  capacity in terms of persons to carry (2, 4, more)
        - lug_boot: boot the size of luggage boot (small, med, big)
        - safety:   estimated safety of the car (low, med, high)
    Predict: class - car class (unacc, acc, good, vgood)
    Data: https://archive.ics.uci.edu/ml/datasets/car+evaluation
"""

# Load data
data = pd.read_csv("../docs/car/car.data")
columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]
predict = "class"
predict_values = ["unacc", "acc", "good", "vgood"]

# Converting data into numeric data
le = preprocessing.LabelEncoder()

buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
doors = le.fit_transform(list(data["doors"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))

x = list(zip(buying, maint, doors, persons, lug_boot, safety))
y = list(le.fit_transform(list(data[predict])))

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# Training a KNN Classifier
k = 7
model = KNeighborsClassifier(n_neighbors=k)
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print("Accuracy: ", acc)

# Testing model
predictions = model.predict(x_test)

print("\nPrediction".ljust(10), '|', "Data".ljust(18), '|', "Actual".ljust(12))
print('-' * 40)
for i in range(len(predictions)):
    print(
        predict_values[predictions[i]].ljust(10), '|',
        str(x_test[i]).ljust(16), '|',
        predict_values[y_test[i]].ljust(12)
    )

    n = model.kneighbors([x_test[i]], k, True)  # K-Neighbors of each point in the testing data
