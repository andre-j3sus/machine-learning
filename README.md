# Machine Learning 🧠

> Some Machine Learning algorithms and notes that I'm making while studying the field.

<p align="center">
  <img align="center" width="600" src="docs/imgs/machine_learning.jpeg" alt="Main Pic"/>
</p>

---

## Introduction 📈

Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial
intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human
intervention.

In machine learning, we create a model and try to predict the test data. There are two types of data:

- **Training data** - used to fit the model
- **Testing data** - used to test the model

Some machine learning algorithms:

- Linear Regression
- Logistic Regression
- Decision Tree
- K-Nearest Neighbors
- Support Vector Machine
- Naive Bayes Algorithm
- K-Means
- Random Forest Algorithm

---

Machine learning has several branches, which include; **supervised learning**, **unsupervised learning**, **deep
learning** and **reinforcement learning**:

### Supervised Learning

The algorithm is given a set of particular targets to aim for, using labeled data set, on contains matched sets of
observed inputs (X's), and the associated outputs (Y's), meaning that the algorithm is
"trained":

***<p align="center">Y = f(X)</p>***

Supervised learning can be categorized into two problems, depending on the nature of the target (Y) variable. These
include **classification** and **regression**:

- **Classification** focuses on sorting observations into distinct categories;
- **Regression** is based on making predictions of continuous target variables.

### Unsupervised Learning

The algorithm aims to produce a set of suitable labels (targets) under unsupervised learning. In other words, we have
inputs (X’s) that are used for analysis with no corresponding target (Y). It is useful for exploring new data sets that
are large and complex.

Two essential types of unsupervised learning are **dimension reduction** and **clustering**:

- **Dimension reduction** refers to reducing the number of inputs (features) while retaining variation across
  observations to maintain structure and usefulness of the information contained in the variation;
- **Clustering** seeks to sort observations into clusters (groups) such that the observations within a cluster are
  similar, while those in different clusters are dissimilar.

<p align="center">
  <img align="center" width="600" src="docs/imgs/supervised_vs_unsupervised.png" alt="Classification and Clustering"/>
</p>

### Deep Learning

Deep learning is a self-teaching system in which the existing data is used to train algorithms to establish patterns and
then use that to make predictions about new data. Highly complex tasks, such as image classification, face recognition,
speech recognition, and natural language processing, are addressed by sophisticated algorithms.

### Reinforcement Learning

In reinforcement learning, a computer learns from trial and error. It learns dynamically by adjusting actions based on
continuous feedback to maximize an outcome.

Deep learning and reinforcement learning are based on **neural networks** (NNs).

---

## Algorithms

### Linear Regression

Linear Regression is a machine learning algorithm based on supervised learning. Performs the task to predict a dependent
variable value (y) based on a given independent variable (x). So, this regression technique finds out a linear
relationship between x (input) and y(output).

Linear regression line equation: **y = a + bx**:

- x: explanatory variable (training data)
- y: dependent variable
- a: slope of the line (how much the y value increases for each x value)
- b: intercept (the value of y when x = 0)

<p align="center">
  <img align="center" width="600" src="docs/imgs/linear_regression.png" alt="Linear Regression"/>
</p>

---

### K-Nearest Neighbors (KNN)

KNN algorithm is a machine learning algorithm based on supervised learning, used for classification and regression.
Works by looking at the **K-closest points** to the given data point
(the one we want to classify) and picking the class that occurs the most to be the predicted value.

However, the algorithm is computationally heavy, because requires the entire data set to make a prediction, and has a
high memory usage, because requires that the entire data set be loaded into memory.

<p align="center">
  <img align="center" width="600" src="docs/imgs/knn.png" alt="KNN"/>
</p>

---

### Support Vector Machine (SVM)

Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers
detection. It's effective in high dimensional spaces (complicated data).

Divides data into multiple classes using a **hyper-plane**. A hyper plane is a fancy word for something that is
straight (e.g., line, plane) that can divide data points. To create a hyper-plane we need to pick two points known as
the **support vectors**; their distance to the plane must be identical, and they need to be the closest points to the
hyper-plane.

We choose the hyper-plane with the greatest possible **margin** (distance between points and plane). There are two types
of data:

- Hard margin: no points may exist inside the margin
- Soft margin: outliers may exist inside the margin
    - **hyper-parameter** is the amount of points allowed to exist inside the margin

**Kernels** provide a way for us to create a hyper-plane for "unorganized" data, bring it up to a higher dimension (in
this case from 2D→3D). There are several kernel options:

- Linear
- Polynomial
- Circular
- Hyperbolic Tangent (Sigmoid)

More info about SVMs [here](https://scikit-learn.org/stable/modules/svm.html#).

<p align="center">
  <img align="center" width="600" src="docs/imgs/svm.png" alt="SVMs"/>
</p>

---

### K-Means Clustering

K-Means clustering is an unsupervised learning algorithm that attempts to divide our training data into k unique
clusters to classify information. It is responsible for learning the differences between our data points and determine
what features determining what class. It attempts to separate each area of our high dimensional space into sections that
represent each class. When we are using it to predict it will simply find what section our point is in and assign it to
that class.

<p align="center">
  <img align="center" width="600" src="docs/imgs/clustering.png" alt="Clustering"/>
</p>

---

## Author Info

- LinkedIn - [André Jesus](https://www.linkedin.com/in/andre-jesus-engineering)
- Twitter - [@andre_j3sus](https://twitter.com/andre_j3sus)
- Website - [André Jesus](https://sites.google.com/view/andre-jesus/p%C3%A1gina-inicial)