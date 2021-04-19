import numpy as np


class Metrics():
    """ Metrics parent class.

    Attributes
    ----------
    None

    Methods
    -------
    __init__()
        Constuctor.
    """

    def __init__(self, ):
        """ Constructor.

        Parameters
        ----------
        None

        Notes
        -----
        None
        """
        pass


class AccuracyMetrics(Metrics):
    """ Accuracy metrics class.

    Attributes
    ----------
    name : str
        The name of the metric.

    Methods
    -------
    __init__()
        Constuctor.
    """

    def __init__(self, ):
        """ Constructor.

        Parameters
        ----------
        None

        Notes
        -----
        None
        """
        super().__init__()
        self.name = "accuracy"

    def compute(self, y, scores):
        """ Computes the accuracy of inferred numerical labels when compared to their true counterparts.

        Parameters
        ----------
        y : numpy.ndarray
            True labels.
            Shape is (number of data points, )
        scores : numpy.ndarray
            Activation of last layer of the model - the scores of the network.
            Shape is (batch_size, out_dim) where out_dim is the output
            dimension of the last layer of the model - usually same as
            the number of classes.

        Returns
        -------
        float
            The accuracy of inferred numerical labels when compared to their true counterparts.

        Notes
        -----
        None

        Raises
        ------
        AssertionError
            If y.shape is not the same as y_hat.shape
        """
        y_hat = np.argmax(scores, axis=1)
        assert y.shape == y_hat.shape
        n = y.shape[0]
        return np.where(y_hat == y)[0].size / n
