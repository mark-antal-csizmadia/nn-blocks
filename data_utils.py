import pickle
import numpy as np


def load_cfar10_batch(path):
    """ Loads a batch of the CIFAR-10 dataset.

    Parameters
    ----------
    path : str
        Path to the data batch.

    Returns
    -------
    features : numpy.ndarray
        Shape is (number of data points, width of image, height of image, number of channels)
        For instance: (10000, 32, 32, 3)
        The width and height might be the other way around.

    labels : numpy.ndarray
        Shape is (number of data points, ).
        For instance: (10000, ).
        Includes between 0-9.

    Notes
    -----
    Based on: https://towardsdatascience.com/cifar-10-image-classification-in-tensorflow-5b501f7dc77c
    """
    with open(path, mode='rb') as file:
        # note the encoding type is 'latin1'
        batch = pickle.load(file, encoding='latin1')

    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = np.array(batch['labels'])

    return features, labels


def load_label_names():
    """ Loads the label names in the CIFAR-10 dataset.

    Parameters
    ----------
    None

    Returns
    -------
    list
        The labels as strings - 10 labels corresponding to 0-9.

    Notes
    -----
    None
    """
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def generate_linear_regression_dataset(m, b, std, n, seed=None):
    """ Generate one-dimensional data with linear trend.

    Parameters
    ----------
    m : float
        Slope of line.
    b : float
        Y-intercept of line.
    std : float
        Standard deviation of random error.
    n : int
        The number of data points.
    seed : int
        Random seed.

    Returns
    -------
    tuple
        Tuple of numpy.ndarrays of x and y.

    Notes
    -----
    None
    """
    if seed is not None:
        np.random.seed(seed)
    x = np.random.random_sample(n) * 50

    if seed is not None:
        np.random.seed(seed)
    e = np.random.randn(n) * std

    y = m*x + b + e

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    return x, y


def generate_non_linear_regression_dataset(b, std, n, seed=None):
    """ Generate one-dimensional data with linear trend.

    Parameters
    ----------
    b : float
        Y-intercept of curve.
    std : float
        Standard deviation of random error.
    n : int
        The number of data points.
    seed : int
        Random seed.

    Returns
    -------
    tuple
        Tuple of numpy.ndarrays of x and y.

    Notes
    -----
    None
    """
    if seed is not None:
        np.random.seed(seed)
    x = np.random.random_sample(n) * 10

    if seed is not None:
        np.random.seed(seed)
    e = np.random.randn(n) * std

    y = x ** 2 + b + e

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    return x, y
