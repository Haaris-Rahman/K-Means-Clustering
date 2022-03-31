import numpy as np
from matplotlib import pyplot as plt


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

    def fit(self, X, mean_splitting=False):
        """
        Intializes the centroids for KMeans.
        If mean_splitting is True, then it iteratively initalizes centroids by taking their mean value as its own centroid
        and randomly initializes another centroid within +- 1 standard deviation from its own mean value
        If mean_splitting is False, intializes centroids with random values

        :param X: The training set to cluster, a numpy array of shape (N, D) containing N examples with D dimensions
        :type X: list or np.array
        :param mean_splitting: Specify whether to use mean splitting algorithm
        :type mean_splitting: bool
        :return: Centroids of K clusters, numpy array of shape (K,D) containing K clusters with D dimensions
        :rtype: np.array
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
            mean_cluster = np.logspace(
                1, np.floor(np.log2(k)), num=int(np.log2(k)), base=2, dtype=int
            )
            if k not in mean_cluster:
                mean_cluster = np.append(mean_cluster, k)
            self.centroids = X.mean(axis=0).reshape(-1, dimension)
            for i in mean_cluster:
                self.k = i
                centroids = np.empty((i, dimension))
                for index, centroid in enumerate(centroids):
                    if index % 2 == 0:
                        centroids[index] = self.centroids[index // 2]
                    else:
                        centroids[index] = self.centroids[
                            index // 2
                        ] + np.random.uniform(-1, 1, size=(1, dimension))

                self.centroids = centroids
                self.centroids = self._kmeans(X)

        else:
            self.centroids = np.random.standard_normal(size=(k, dimension))
            self.centroids = self._kmeans(X)
        return self.centroids

    def _kmeans(self, X):
        """
        Performs the k-means clustering algorithm. If a cluster has no samples, randomly reinitialize a new centroid

        :param X: The training set to cluster, a numpy array of shape (N, D) containing N examples with D dimensions
        :type X: np.array
        :return: Centroids of K clusters, numpy array of shape (K,D) containing K clusters with D dimensions
        :rtype: np.array
        """

        assert X is not None
        assert isinstance(X, (np.ndarray))

        k = self.k
        samples, dimension = X.shape
        self.dimension = dimension
        centroids = self.centroids

        distance = np.zeros((k, samples))

        for i in range(self.max_iters):
            old_mean = np.copy(centroids)
            for index, centroid in enumerate(centroids):
                centroid = centroid.reshape(-1, dimension)
                distance[index] = np.sqrt(np.sum(((X - centroid) ** 2), axis=1))

            labels = distance.argmin(axis=0)
            empty_cluster = []
            for index, _ in enumerate(centroids):
                if (labels == index).sum() == 0:
                    empty_cluster.append(index)
                else:
                    centroids[index] = X[labels == index].mean(axis=0)

            for index in empty_cluster:
                centroids[index] = centroids[index] + np.random.uniform(
                    -1, 1, size=(1, dimension)
                )

            if np.array_equal(centroids, old_mean):
                break

        self.centroids = centroids
        return self.centroids

    def predict(self, X):
        """
        Predict the cluster of test samples

        :param X: The testing set to cluster, a numpy array of shape (N, D) containing N examples with D dimensions
        :type X: list or np.array
        :return: The labels of the closest centroids of shape (N,)
        :rtype: np.array
        """
        """
        The predict function is used to predict the label of a new data point.
        The predict function takes in an array of data points, where each data point is a row vector.
        The predict function outputs an array containing the predicted labels for each example.

        :param self: Used to Access variables that belongs to the class.
        :param X: Used to Pass the data to be classified.
        :return: The labels of the closest centroids to the given data.

        :doc-author: Trelent
        """

        assert X is not None
        assert isinstance(X, (np.ndarray, list))
        if isinstance(X, list):
            X = np.array(X)

        assert self.dimension == X.shape[-1]

        k = self.k
        centroids = self.centroids
        samples, dimension = X.shape
        distance = np.zeros((k, samples))

        for index, centroid in enumerate(centroids):
            centroid = centroid.reshape(-1, dimension)
            distance[index] = np.sqrt(np.sum(((X - centroid) ** 2), axis=1))

        labels = distance.argmin(axis=0)
        return labels

    def plot(self, X_train, X_test=None, y_train=None, y_test=None, projection=False):
        """
        Plots the data points and centroids of the clusters.

        :param X_train: The training set, a numpy array of shape (N, D) containing N examples with D dimensions
        :type X_train: list or np.array
        :param X_test: (Optional) The testing set, a numpy array of shape (N, D) containing N examples with D dimensions
        :type X_test: list or np.array
        :param y_train: (Optional) Label of cluster for each training sample
        :type y_train: list or np.array
        :param y_test: (Optional) Label of cluster for each test sample
        :type y_test: list or np.array
        :param projection: (Optional) Plot in either 2D or 3D space (True = Plot in 3D)
        :type projection: bool
        :return: Matplotlib figure and axis object
        """

        assert X_train is not None
        assert isinstance(X_train, (np.ndarray, list))
        assert isinstance(projection, bool)

        if X_test is not None:
            assert isinstance(X_test, (np.ndarray, list))
        if y_train is not None:
            assert isinstance(y_train, (np.ndarray, list))
        if y_test is not None:
            assert isinstance(y_test, (np.ndarray, list))

        if isinstance(X_train, list):
            X_train = np.array(X_train)
        if isinstance(y_train, list):
            y_train = np.array(y_train)
        if isinstance(X_test, list):
            X_test = np.array(X_test)
        if isinstance(y_test, list):
            y_test = np.array(y_test)

        assert self.dimension == X_train.shape[1]
        assert 1 <= self.dimension <= 3

        if X_test is not None:
            assert self.dimension == X_test.shape[1]

        if y_train is None:
            y_train = self.predict(X_train)

        if y_test is None and X_test is not None:
            y_test = self.predict(X_test)

        dimension = self.dimension
        centroids = self.centroids
        fig = plt.figure()
        if dimension == 1:
            ax = fig.add_subplot(projection="3d" if projection else "rectilinear")
            for index, centroid in enumerate(centroids):
                cluster = X_train[y_train == index]
                train = ax.scatter(cluster[:, 0], np.zeros(cluster.shape[0]))
                ax.scatter(
                    centroid[0],
                    np.zeros(centroid.shape[0]),
                    s=200,
                    c=train.get_edgecolor(),
                )
                if X_test is not None:
                    cluster = X_test[y_test == index]
                    ax.scatter(
                        cluster[:, 0],
                        np.zeros(cluster.shape[0]),
                        marker="x",
                        c=train.get_edgecolor(),
                    )

        elif dimension == 2:
            ax = fig.add_subplot(projection="3d" if projection else "rectilinear")
            for index, centroid in enumerate(centroids):
                cluster = X_train[y_train == index]
                train = ax.scatter(cluster[:, 0], cluster[:, 1])
                ax.scatter(centroid[0], centroid[1], s=200, c=train.get_edgecolor())
                if X_test is not None:
                    cluster = X_test[y_test == index]
                    ax.scatter(
                        cluster[:, 0],
                        cluster[:, 1],
                        marker="x",
                        c=train.get_edgecolor(),
                    )
        else:
            ax = fig.add_subplot(projection="3d")
            for index, centroid in enumerate(centroids):
                cluster = X_train[y_train == index]
                train = ax.scatter(cluster[:, 0], cluster[:, 1], cluster[:, 2])
                ax.scatter(
                    centroid[0],
                    centroid[1],
                    centroid[2],
                    s=200,
                    c=train.get_edgecolor(),
                )
                if X_test is not None:
                    cluster = X_test[y_test == index]
                    ax.scatter(
                        cluster[:, 0],
                        cluster[:, 1],
                        cluster[:, 2],
                        marker="x",
                        c=train.get_edgecolor(),
                    )

        ax.legend(
            ["Train Samples", "Centroids", "Test Samples"]
        ) if X_test is not None else ax.legend([" Samples", "Centroids"])

        leg = ax.get_legend()
        for handle in leg.legendHandles:
            handle.set_color("brown")
        return fig, ax