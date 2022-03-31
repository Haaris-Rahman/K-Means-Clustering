# Example code to demonstrate use of the library

from kmeans import KMeans
import numpy as np
from matplotlib import pyplot as plt


def plot_1D(samples, test_split):

    train_samples = int(test_split * samples)
    shape = 1
    data1 = np.random.normal(loc=3, scale=1, size=(samples, shape))
    data2 = np.random.normal(loc=-3, scale=1, size=(samples, shape))
    data3 = np.random.normal(loc=-10, scale=1, size=(samples, shape))
    data4 = np.random.normal(loc=10, scale=1, size=(samples, shape))
    X_train = np.vstack(
        (
            data1[:train_samples],
            data2[:train_samples],
            data3[:train_samples],
            data4[:train_samples],
        )
    )
    X_test = np.vstack(
        (
            data1[train_samples:],
            data2[train_samples:],
            data3[train_samples:],
            data4[train_samples:],
        )
    )

    kmeans = KMeans(clusters=8)

    kmeans.fit(X_train)
    y_train = kmeans.predict(X_train)
    y_test = kmeans.predict(X_test)
    fig, ax = kmeans.plot(X_train, X_test, y_train, y_test)
    ax.set_title("1D Without Mean Splitting")

    kmeans.fit(X_train, mean_splitting = True)
    y_train = kmeans.predict(X_train)
    y_test = kmeans.predict(X_test)
    fig, ax = kmeans.plot(X_train, X_test, y_train, y_test)
    ax.set_title("1D With Mean Splitting")
    plt.show()


def plot_2D(samples, test_split):

    train_samples = int(test_split * samples)
    shape = 2
    data1 = np.random.normal(loc=-3, scale=1, size=(samples, shape))
    data2 = np.random.normal(loc=3, scale=1, size=(samples, shape))
    data3 = np.vstack((data1[:, 0], data2[:, 1])).T
    data4 = np.vstack((data2[:, 0], data1[:, 1])).T

    X_train = np.vstack(
        (
            data1[:train_samples],
            data2[:train_samples],
            data3[:train_samples],
            data4[:train_samples],
        )
    )
    X_test = np.vstack(
        (
            data1[train_samples:],
            data2[train_samples:],
            data3[train_samples:],
            data4[train_samples:],
        )
    )

    kmeans = KMeans(clusters=8)
    kmeans.fit(X_train)
    y_train = kmeans.predict(X_train)
    y_test = kmeans.predict(X_test)
    fig, ax = kmeans.plot(X_train, X_test, y_train, y_test)
    ax.set_title("2D Without Mean Splitting")

    kmeans.fit(X_train, mean_splitting = True)
    y_train = kmeans.predict(X_train)
    y_test = kmeans.predict(X_test)
    fig, ax = kmeans.plot(X_train, X_test, y_train, y_test)
    ax.set_title("2D With Mean Splitting")
    plt.show()


def plot_3D(samples, test_split):

    mean = np.array([10, -10, -10])
    cov = np.eye(3) * 5
    train_samples = int(test_split * samples)
    data1 = np.random.multivariate_normal(mean=mean, cov=cov, size=samples)
    data2 = np.random.multivariate_normal(mean=-mean, cov=cov, size=samples)
    data3 = np.vstack((data1[:, 0], data2[:, 1], data1[:, 2])).T
    data4 = np.vstack((data2[:, 0], data1[:, 1], data2[:, 2])).T

    X_train = np.vstack(
        (
            data1[:train_samples],
            data2[:train_samples],
            data3[:train_samples],
            data4[:train_samples],
        )
    )
    X_test = np.vstack(
        (
            data1[train_samples:],
            data2[train_samples:],
            data3[train_samples:],
            data4[train_samples:],
        )
    )

    kmeans = KMeans(clusters=8)
    kmeans.fit(X_train)
    y_train = kmeans.predict(X_train)
    y_test = kmeans.predict(X_test)
    fig, ax = kmeans.plot(X_train, X_test, y_train, y_test)
    ax.set_title("3D Without Mean Splitting")

    kmeans.fit(X_train, mean_splitting = True)
    y_train = kmeans.predict(X_train)
    y_test = kmeans.predict(X_test)
    fig, ax = kmeans.plot(X_train, X_test, y_train, y_test)
    ax.set_title("3D With Mean Splitting")
    plt.show()


if __name__ == "__main__":
    # For reproducibility, use random seed 1
    # np.random.seed(1)
    plot_1D(samples=100, test_split=0.5)
    plot_2D(samples=100, test_split=0.5)
    plot_3D(samples=200, test_split=0.5)
