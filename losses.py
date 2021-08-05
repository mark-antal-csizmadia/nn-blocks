import numpy as np
from copy import deepcopy


class Loss():
    """ Loss parent class.

    Attributes
    ----------
    cache : dict
        Run-time cache of attributes such as gradients.

    Methods
    -------
    __init__()
        Constuctor.
    """

    def __init__(self, name, loss_smoother):
        """ Constructor.

        Parameters
        ----------
        None

        Notes
        -----
        None
        """
        self.cache = {}
        self.name = name
        self.loss_smoother = loss_smoother
        self.repr_str = self.name + " with " + self.loss_smoother.__repr__()

    def __repr__(self):
        return self.repr_str


class CategoricalHingeLoss(Loss):
    """ Categorical Hinge loss for realizing an SVM classifier.
    Usually preceeded by a linear activation.
    For multi-class classification.
    Inherits everything from class Loss.

    Attributes
    ----------
    cache : dict
        Run-time cache of attibutes such as gradients.

    Methods
    -------
    __init__()
        Constuctor.
    compute_loss(scores, y, layers_reg_loss)
        Computes loss of classifier - also includes the regularization losses from previous layers.
    grad()
        Computes the gradient of the loss function.
    """

    def __init__(self, loss_smoother):
        """ Constructor.

        Parameters
        ----------
        None

        Notes
        -----
        None
        """
        name = "categorical hinge loss"
        super().__init__(name, loss_smoother)

    def compute_loss(self, scores, y):
        """ Computes loss of classifier - also includes the regularization losses from previous layers.

        Parameters
        ----------
        scores : numpy.ndarray
            Scores. Usually from linear activation.
            Shape is (number of data points, number of classes)
        y : numpy.ndarray
            True labels.
            Shape is (number of data points, )

        Returns
        -------
        loss : float
            The overall loss of the classifier.

        Notes
        -----
        None
        """
        # number of classes
        c = scores.shape[1]
        # batch size
        n = y.shape[0]

        # correct_class_scores.shape = (batch size, 1)
        correct_class_scores = scores[range(n), y].reshape(n, 1)
        # margin.shape = (batch size, number of classes)
        margin = np.maximum(0, scores - correct_class_scores + 1)
        # do not consider correct class in loss
        margin[range(n), y] = 0
        loss = margin.sum() / n

        margin[margin > 0] = 1
        # valid_margin_count.shape = (batch size, )
        valid_margin_count = margin.sum(axis=1)
        # Subtract in correct class (-s_y)
        # margin.shape = (batch size, number of classes)
        margin[range(n), y] -= valid_margin_count
        margin /= n
        self.cache["g"] = deepcopy(margin)

        # smooth loss
        if self.loss_smoother is not None:
            loss = self.loss_smoother(loss)

        return loss

    def grad(self, ):
        """ Computes the gradient of the loss function.

        Parameters
        ----------
        None

        Returns
        -------
        numpy.ndarray or None
            None if gradient has not yet been computed.
            Shape of gradient is (batch size, number of classes)

        Notes
        -----
        None
        """
        if "g" in self.cache.keys():
            return deepcopy(self.cache["g"])
        else:
            return None


class CategoricalCrossEntropyLoss(Loss):
    """ Categorical cross-entropy loss.
    Usually preceeded by a linear activation.
    For multi-class classification.
    Inherits everything from class Loss.

    Attributes
    ----------
    cache : dict
        Run-time cache of attibutes such as gradients.

    Methods
    -------
    __init__()
        Constuctor.
    compute_loss(scores, y, layers_reg_loss)
        Computes loss of classifier - also includes the regularization losses from previous layers.
    grad()
        Computes the gradient of the loss function.
    """

    def __init__(self, loss_smoother):
        """ Constructor.

        Parameters
        ----------
        None


        Notes
        -----
        None
        """
        name = "categorical cross-entropy loss"
        super().__init__(name, loss_smoother)

    def compute_loss(self, scores, y):
        """ Computes loss of classifier - also includes the regularization losses from previous layers.

        Parameters
        ----------
        scores : numpy.ndarray
            Scores. Usually from softmax activation.
            Shape is (batch size, number of classes)
        y : numpy.ndarray
            True labels.
            Shape is (batch size, )

        Returns
        -------
        loss : float
            The overall loss of the classifier.

        Notes
        -----
        None
        """
        self.cache["g"] = deepcopy(y)
        n = y.shape[0]

        # correct_logprobs.shape = (batch_size, )
        correct_logprobs = -np.log(scores[range(n), y])

        # compute the loss: average cross-entropy loss and regularization
        loss = np.sum(correct_logprobs) / n

        # smooth loss
        if self.loss_smoother is not None:
            loss = self.loss_smoother(loss)

        return loss

    def grad(self, ):
        """ Computes the gradient of the loss function.

        Parameters
        ----------
        None

        Returns
        -------
        numpy.ndarray or None
            None if gradient has not yet been computed.
            Shape of gradient is (batch size, ). Note that the grad here is just y.

        Notes
        -----
        None
        """
        if "g" in self.cache.keys():
            return deepcopy(self.cache["g"])
        else:
            return None


class LossSmoother():
    def __init__(self, repr_str):
        self.first_call = True
        self.repr_str = "loss smoother " + repr_str

    def __call__(self, loss):
        raise NotImplementedError

    def __repr__(self):
        return self.repr_str


class LossSmootherConstant(LossSmoother):
    def __init__(self,):
        repr_str = "constant"
        super().__init__(repr_str)
        self.cache = {"loss_smooth": None}

    def __call__(self, loss):
        if self.first_call:
            self.first_call = False
            self.cache["loss_smooth"] = deepcopy(loss)
        else:
            self.cache["loss_smooth"] = deepcopy(loss)

        return deepcopy(self.cache["loss_smooth"])


class LossSmootherMovingAverage(LossSmoother):
    def __init__(self, alpha):
        repr_str = "exp ave"
        super().__init__(repr_str)
        self.alpha = alpha
        self.cache = {"loss_smooth": None}

    def __call__(self, loss):
        if self.first_call:
            self.first_call = False
            self.cache["loss_smooth"] = deepcopy(loss)
        else:
            loss_smooth = deepcopy(self.cache["loss_smooth"])
            self.cache["loss_smooth"] = self.alpha * loss_smooth + (1 - self.alpha) * loss

        return deepcopy(self.cache["loss_smooth"])
