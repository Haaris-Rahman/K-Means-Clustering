import numpy as np


class KMeans:
    """
    K means clustering algorithm implemented for n-dimensional inputs
    Implements a plotting function to plot clusters in 1D, 2D and 3D
    """

    def __init__(self, clusters, max_iters=100):
        """
        Constructor for the KMeans class

        :param clusters: Number of clusters
        :type clusters: int
        :param max_iters: Maximum number of iterations for the algorithm
        :type max_iters: int
        """
        assert isinstance(clusters, int) and clusters > 0
        assert isinstance(max_iters, int) and max_iters > 0

        self.k = clusters
        self.max_iters = max_iters
        self.dimension = 0
        self.centroids = None

    def _calc_distance(self, X):
        """
        Calculates the distance of training point from each centroid
        :param X: The training set to cluster, a numpy array of shape (N, D) containing N examples with D dimensions
        :type X: np.ndarray
        :return: Distance of training point from each centroid, numpy array of shape(k, N)
        :rtype: np.ndarray
        """
        assert X is not None
        assert isinstance(X, np.ndarray)

        k = self.k
        centroids = self.centroids
        samples, dimension = X.shape
        distance = np.zeros((k, samples))
        for index in range(k):
            centroid = centroids[index]
            distance[index] = np.linalg.norm((X - centroid), axis=1)
        return distance

    def _get_centroid_space(self):
        """
        Generates the number of clusters for each iteration of mean splitting
        :return: Number of clusters for each iteration of mean splitting
        :rtype: np.ndarray
        """
        k = self.k
        cluster_space = np.logspace(1, np.floor(np.log2(k)), num=int(np.log2(k)), base=2, dtype=int)
        if k not in cluster_space:
            cluster_space = np.append(cluster_space, k)
        return cluster_space

    def _split_centroids(self, num_centroids):
        """
        Splits the each centroid into two as per mean splitting algorithm
        :param num_centroids:
        :type num_centroids:
        :return: 
        :rtype: 
        """
        assert num_centroids > 0

        dimension = self.dimension
        k = self.k
        centroids = np.empty((num_centroids, dimension))
        for index in range(k):
            if index % 2 == 0:
                centroids[index] = self.centroids[index // 2]
            else:
                centroids[index] = self.centroids[index // 2] + np.random.uniform(-1, 1, size=(1, dimension))
        self.centroids = centroids

    def fit(self, X, mean_splitting=False):
        """
        Initializes the centroids for KMeans.
        If mean_splitting is True, then it iteratively initializes centroids by
        taking their mean value as its own centroid and randomly initializes another centroid within +- 1 standard
        deviation from its own mean value.
        If mean_splitting is False, initializes centroids with random samples from the dataset.
        :param X: The training set to cluster, a numpy array of shape (N, D) containing N examples with D dimensions
        :type X: list or np.ndarray
        :param mean_splitting: Specify whether to use mean splitting algorithm
        :type mean_splitting: bool
        :return: Centroids of K clusters, numpy array of shape (K,D) containing K clusters with D dimensions
        :rtype: np.ndarray
        """

        assert X is not None
        assert isinstance(X, (np.ndarray, list))
        assert isinstance(mean_splitting, bool)

        if isinstance(X, list):
            X = np.array(X)

        samples, dimension = X.shape
        self.dimension = dimension
        k = self.k

        if mean_splitting:
            centroid_space = self._get_centroid_space()
            self.centroids = X.mean(axis=0).reshape(-1, dimension)
            for num_centroids in centroid_space:
                self.k = num_centroids
                self._split_centroids(num_centroids)
                self.centroids = self._kmeans(X)

        else:
            random_indices = np.random.choice(samples + 1, size=k, replace=False)
            self.centroids = X[random_indices]
            self.centroids = self._kmeans(X)
        return self.centroids

    def _kmeans(self, X):
        """
        Performs the k-means clustering algorithm.
        :param X: The training set to cluster, a numpy array of shape (N, D) containing N examples with D dimensions
        :type X: np.ndarray
        :return: Centroids of K clusters, numpy array of shape (K,D) containing K clusters with D dimensions
        :rtype: np.ndarray
        """
        assert X is not None
        assert isinstance(X, np.ndarray)

        k = self.k
        centroids = self.centroids

        for i in range(self.max_iters):
            old_mean = np.copy(centroids)
            distance = self._calc_distance(X)
            labels = distance.argmin(axis=0)
            for index in range(k):
                centroids[index] = X[labels == index].mean(axis=0)
            if np.array_equal(centroids, old_mean):
                break

        self.centroids = centroids
        return self.centroids

    def predict(self, X):
        """
        Predict the cluster of test samples

        :param X: The testing set to cluster, a numpy array of shape (N, D) containing N examples with D dimensions
        :type X: list or np.ndarray
        :return: The labels of the closest centroids of shape (N,)
        :rtype: np.ndarray
        """
        assert X is not None
        assert isinstance(X, (np.ndarray, list))
        if isinstance(X, list):
            X = np.array(X)

        assert self.dimension == X.shape[-1]
        distance = self._calc_distance(X)
        labels = distance.argmin(axis=0)
        return labels
