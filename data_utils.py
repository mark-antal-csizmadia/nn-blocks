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
