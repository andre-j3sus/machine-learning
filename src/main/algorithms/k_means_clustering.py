import numpy as np
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics

"""
    Program Objective: Classify each digit
    Algorithm: K-Means Clustering
    Variables: -
    Predict: digit class
    Data: digits data set from sklearn module
"""

# Load data
digits = load_digits()
data = scale(digits.data)  # Scale down features with large values

y = digits.target
k = len(np.unique(y))
samples, features = data.shape


# Scoring
def bench_k_means(estimator, name, x):
    """
    Benchmark to evaluate the KMeans initialization methods.
    Function from sklearn: https://scikit-learn.org/0.18/auto_examples/cluster/plot_kmeans_digits.html
    """

    estimator.fit(x)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y, estimator.labels_),
             metrics.silhouette_score(x, estimator.labels_, metric='euclidean')
             )
          )


# Model
classifier = KMeans(n_clusters=k, init="random", n_init=10)
bench_k_means(classifier, "1", data)
