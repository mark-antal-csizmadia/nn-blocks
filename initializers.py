import numpy as np


class Initializer():
    """ Initializer parent class.


    Attributes
    ----------
    seed : int
        Seed of pseudo-random generators such as random parameter initialization.

    Methods
    -------
    __init__(seed=None)
        Constuctor.
    """

    def __init__(self, seed=None):
        """ Constructor.

        Parameters
        ----------
        seed : int
            Seed of pseudo-random generators such as random parameter initialization.


        Notes
        -----
        None
        """
        self.seed = seed


class NormalInitializer(Initializer):
    """ Normal, or Gaussian, parameter initializer.


    Attributes
    ----------
    coeff : float
        Multiplicative coefficient of Normal distribution.
    mean : float
        Mean of Normal distribution.
    std : float
        Standard deviation of Normal distribution.

    Methods
    -------
    __init__(seed=None, **params)
        Constuctor.
    initialize(size)
        Initializes parameters by drawing from a Normal distribution.
    __repr__
        Returns the string representation of class.
    """

    def __init__(self, seed, **params):
        """ Constructor.

        Parameters
        ----------
        seed : int
            Seed of pseudo-random generators such as random parameter
            initialization.
        params : dict
            Dictionary of initialization distribution parameters such as
            multiplicative coefficient, mean, and standard deviation.

        Notes
        -----
        None
        """
        super().__init__(seed)
        self.coeff = params["coeff"]
        self.mean = params["mean"]
        self.std = params["std"]

    def initialize(self, size):
        """ Initializes parameters by drawing from a Normal distribution.

        Parameters
        ----------
        size : tuple
            Tuple of dimensions of the parameter tensor.

        Returns
        -------
        numpy.ndarray
            Initialized parameters.

        Notes
        -----
        None
        """
        np.random.seed(self.seed)
        return self.coeff * np.random.normal(loc=self.mean, scale=self.std, size=size)

    def __repr__(self):
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
        repr_str = "normal ~ " + f"{self.coeff} x N({self.mean}, {self.std}^2)"
        return repr_str


class XavierInitializer(Initializer):
    """ Xavier initializer.
    From: Understanding the difficulty of training deep feedforward neural networks


    Attributes
    ----------
    coeff : float
        Multiplicative coefficient of Normal distribution.
    mean : float
        Mean of Normal distribution.
    std : None
        None as the Xavier initializer computes on its own the
        standard deviation of the Normal distribution.

    Methods
    -------
    __init__(seed=None, **params)
        Constuctor.
    initialize(size)
        Initializes parameters by drawing from a Normal distribution.
    __repr__()
        Returns the string representation of class.
    """

    def __init__(self, seed, **params):
        """ Constructor.

        Parameters
        ----------
        seed : int
            Seed of pseudo-random generators such as random parameter
            initialization.
        params : dict
            Dictionary of initialization distribution parameters such as
            multiplicative coefficient, mean, and the standard deviation
            is None and is computed in self.initialize(size).

        Notes
        -----
        None

        Raises
        ------
        AssertionError
            If the std in the params dict is not None.
        """
        super().__init__(seed)
        self.coeff = params["coeff"]
        self.mean = params["mean"]
        assert params["std"] is None, "Xavier init takes no std"

    def initialize(self, size):
        """ Initializes parameters by drawing from a Normal distribution with
        the Xavier strategy.

        Parameters
        ----------
        size : tuple
            Tuple of dimensions of the parameter tensor.

        Returns
        -------
        numpy.ndarray
            Initialized parameters.

        Notes
        -----
        None
        """
        # size=(in_dim, out_dim)
        np.random.seed(self.seed)
        in_dim = size[0]
        self.std = 1 / np.sqrt(in_dim)
        return self.coeff * np.random.normal(loc=self.mean, scale=self.std, size=size)

    def __repr__(self):
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
        repr_str = "Xavier ~ " + f"{self.coeff} x N({self.mean}, {self.std}^2)"
        return repr_str
