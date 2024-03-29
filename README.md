# Machine Learning 🤖

> Some Machine Learning algorithms and notes that I'm making while studying the field.

<p align="center">
  <img align="center" width="600" src="docs/imgs/ml.jpeg" alt="Machine Learning"/>
</p>

---

## Introduction 📈

> Machine learning is a method of data analysis that automates analytical model building. It is a branch of **Artificial Intelligence** based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.

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

Machine learning has several branches, which include; **supervised learning**, **unsupervised learning**, **deep learning** and **reinforcement learning**:

### Supervised Learning

The algorithm is given a set of particular targets to aim for, using labeled data set, on contains matched sets of
observed inputs (X's), and the associated outputs (Y's), meaning that the algorithm is
"trained":

***<p align="center">`Y = f(X)`</p>***

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
  <img align="center" width="600" src="docs/imgs/supervised-and-unsupervised.png" alt="Classification and Clustering"/>
</p>

### Deep Learning

Deep learning is a self-teaching system in which the existing data is used to train algorithms to establish patterns and
then use that to make predictions about new data. Highly complex tasks, such as image classification, face recognition,
speech recognition, and natural language processing, are addressed by sophisticated algorithms.

### Reinforcement Learning

In reinforcement learning, a computer learns from trial and error. It learns dynamically by adjusting actions based on
continuous feedback to maximize an outcome.

### Neural Networks

Deep learning and reinforcement learning are based on **neural networks** (NNs). Artificial neural networks (ANNs),
usually simply called neural networks (NNs), are computing systems inspired by the biological neural networks that
constitute animal brains.

Similar to the human brain that has neurons interconnected to one another, artificial neural networks also have
**neurons** that are interconnected to one another in various layers of the networks. These neurons are known as
**nodes**.

When we feed information to the **input neurons** we get some information from the **output neurons**. Information
starts at the input neurons and travels to the next layers of neurons having what's called a weight and a **bias**
applied to it. These weight and biases start out randomly determined and are tweaked as the network learns and sees more
data.

<p align="center">
  <img align="center" width="600" src="docs/imgs/nn.png" alt="NNs"/>
</p>

---

## Algorithms

In this chapter you will find some notes about some basic machine learning algorithms.

### Linear Regression

Linear Regression is a machine learning algorithm based on supervised learning. Performs the task to predict a dependent
variable value (y) based on a given independent variable (x). So, this regression technique finds out a linear
relationship between x (input) and y(output).

Linear regression line equation: **`y = a + bx`**:

- x: explanatory variable (training data)
- y: dependent variable
- a: slope of the line (how much the y value increases for each x value)
- b: intercept (the value of y when x = 0)

<p align="center">
  <img align="center" width="600" src="docs/imgs/linear-and-logistic-regression.png" alt="Linear Regression"/>
</p>

---

### Logistic Regression

Logistic regression is a statistical model that in its basic form uses a logistic function to model a **binary**
dependent variable. In addition, as opposed to linear regression, logistic regression does not require a linear
relationship between inputs and output variables.

**Binary classification** refers to those classification tasks that have two class labels.

Logistic regression line equation (sigmoid function of linear regression equation):

**<p align="center">`ŷ = σ(wx + b)`</p>**
**<p align="center">`z = wx + b`</p>**
**<p align="center">`ŷ = σ(z) = 1/(1 + e^(-z))`</p>**

- x: explanatory variable (training data)
- ŷ: dependent variable (0 <= ŷ <= 1)
- w: dimensional vector
- b: real number

The **loss (error) function** is a value which is calculated at every instance that measures how well you're doing on the
single training example: **`L(ŷ, y) = -(y log(ŷ) + (1-y)log(1-ŷ))`** (we want to ŷ to be approximated to y).

The **cost function** is calculated as an average of loss functions that measures how well the parameters W and B are doing
on the entire training set: **`J(w, b) = 1/m * mΣ(i=1) (L(ŷ^i, y^i))`**

<p align="center">
  <img align="center" width="600" src="docs/imgs/linear-and-logistic-regression.png" alt="Logistic Regression"/>
</p>

#### Gradient Descent

To find the values of **`w`** and **`b`**, it's used the gradient descent (GD) algorithm. Gradient descent (GD) is an
iterative first-order optimisation algorithm used to find a local minimum/maximum of a given function. his method is
commonly used in machine learning (ML) and deep learning(DL) to minimise a cost/loss function.

<p align="center">
  <img align="center" width="400" src="docs/imgs/gradient-descent.jpg" alt="Gradient Descent"/>
</p>

---

### K-Nearest Neighbors (KNN)

KNN algorithm is a machine learning algorithm based on supervised learning, used for classification and regression.
Works by looking at the **K-closest points** to the given data point
(the one we want to classify) and picking the class that occurs the most to be the predicted value.

However, the algorithm is computationally heavy, because requires the entire data set to make a prediction, and has a
high memory usage, because requires that the entire data set be loaded into memory.

<p align="center">
  <img align="center" width="400" src="docs/imgs/knn.png" alt="KNN"/>
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
  <img align="center" width="400" src="docs/imgs/svm.png" alt="SVMs"/>
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
