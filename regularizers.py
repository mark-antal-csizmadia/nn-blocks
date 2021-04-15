import numpy as np


class Regularizer():
    """ Regularizer parent class.


    Attributes
    ----------
    reg_rate : float
        Regularization rate.

    Methods
    -------
    __init__(reg_rate)
        Constuctor.
    """

    def __init__(self, reg_rate):
        """ Constructor.

        Parameters
        ----------
        reg_rate : float
            Regularization rate.


        Notes
        -----
        None
        """
        self.reg_rate = reg_rate


class L2Regularizer(Regularizer):
    """ L2 regularizer.


    Attributes
    ----------
    reg_rate : float
        Regularization rate.

    Methods
    -------
    __init__(reg_rate)
        Constuctor.
    loss(param)
        Computes the regulariztion loss of the parameters such as
        layer loss in dense due to large weights.
    grad(param)
        Computes the gradient of the regularization term with respect
        to param.
    __repr__()
        Returns the string representation of class.
    """

    def __init__(self, reg_rate):
        """ Constructor.

        Parameters
        ----------
        reg_rate : float
            Regularization rate.


        Notes
        -----
        None
        """
        super().__init__(reg_rate)

    def loss(self, param):
        """ Computes the regulariztion loss of the parameters such as
        layer loss in dense due to large weights.

        Parameters
        ----------
        param : numpy.ndarray
            The parameter tensor.

        Returns
        -------
        float
            Regularization loss.

        Notes
        -----
        None
        """
        return 0.5 * self.reg_rate * np.sum(np.power(param, 2))

    def grad(self, param):
        """ Computes the gradient of the regularization term with respect
        to param.

        Parameters
        ----------
        param : numpy.ndarray
            The parameter tensor.

        Returns
        -------
        numpy.ndarray
            Regularization gradient with respect to param.

        Notes
        -----
        None
        """
        return self.reg_rate * param

    def __repr__(self, ):
        """ Returns the string representation of class.

        Parameters
        ----------
        None

        Returns
        -------
        repr_str : str
            The string representation of the class.

        Notes
        -----
        None
        """
        repr_str = "l2"
        return repr_str
