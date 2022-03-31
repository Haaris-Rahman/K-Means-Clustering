import matplotlib.pyplot as plt
import numpy as np


def plot(centroids, X_train, X_test=None, y_train=None, y_test=None, projection=False):
    """
    Plots the data points and centroids of the clusters for the kmeans library
    :param centroids: The centroids obtained after kmeans training
    :type centroids: np.ndarray
    :param X_train: The training set, a numpy array of shape (N, D) containing N examples with D dimensions
    :type X_train: list or np.ndarray
    :param X_test: (Optional) The testing set, a numpy array of shape (N, D) containing N examples with D dimensions
    :type X_test: list or np.ndarray
    :param y_train: (Optional) Label of cluster for each training sample
    :type y_train: list or np.ndarray
    :param y_test: (Optional) Label of cluster for each test sample
    :type y_test: list or np.ndarray
    :param projection: (Optional) Plot in either 2D or 3D space (True = Plot in 3D)
    :type projection: bool
    :return: Matplotlib figure and axis object
    """

    assert X_train is not None
    assert isinstance(X_train, (np.ndarray, list))
    assert isinstance(projection, bool)

    if isinstance(X_train, list):
        X_train = np.array(X_train)

    if X_test is not None:
        assert isinstance(X_test, (np.ndarray, list))
        if isinstance(X_test, list):
            X_test = np.array(X_test)

    if y_train is not None:
        assert isinstance(y_train, (np.ndarray, list))
        if isinstance(y_train, list):
            y_train = np.array(y_train)

    if y_test is not None:
        assert isinstance(y_test, (np.ndarray, list))
        if isinstance(y_test, list):
            y_test = np.array(y_test)

    dimension = X_train.shape[1]
    if X_test is not None:
        assert dimension == X_test.shape[1]
    assert 1 <= dimension <= 3

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