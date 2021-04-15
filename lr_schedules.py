import numpy as np


class LRSchedule():
    """ Learning rate schedule parent class.

    Attributes
    ----------
    lr_initial : float
        Initial, or base, learning rate.
    lr : float
        The latest learning rate.
    step : int
        Update step counter used for applying the learning rate schedule.

    Methods
    -------
    __init__()
        Constuctor.
    """

    def __init__(self, lr_initial):
        """ Constructor.

        Parameters
        ----------
        lr_initial : float
            Initial, or base, learning rate.

        Notes
        -----
        None
        """
        self.lr_initial = lr_initial
        self.lr = self.lr_initial
        self.step = 0


class LRConstantSchedule(LRSchedule):
    """ Constant learning rate schedule.

    Attributes
    ----------
    lr_initial : float
        Initial, or base, learning rate.
    lr : float
        The latest learning rate.
    step : int
        Update step counter used for applying the learning rate schedule.

    Methods
    -------
    __init__()
        Constuctor.
    apply_schedule()
        Applies the constant learning rate schedule.
    get_lr()
        Returns the latest learning rate.
    """

    def __init__(self, lr_initial):
        """ Constructor.
        Inherits everything from the LRSchedule class

        Parameters
        ----------
        lr_initial : float
            Initial, or base, learning rate.

        Notes
        -----
        None
        """
        super().__init__(lr_initial)

    def apply_schedule(self, ):
        """ Applies the constant learning rate schedule.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        None
        """
        pass

    def get_lr(self, ):
        """ Returns the latest learning rate.

        Parameters
        ----------
        None

        Returns
        -------
        float
            The latest learning rate.

        Notes
        -----
        None
        """
        return self.lr


class LRExponentialDecaySchedule(LRSchedule):
    """ Exponential decay learning rate schedule.

    Attributes
    ----------
    lr_initial : float
        Initial, or base, learning rate.
    lr : float
        The latest learning rate.
    step : int
        Update step counter used for applying the learning rate schedule.
    decay_steps : int
            The number of decay steps. The smaller, the faster the decay.
    decay_rate : float
        The rate of decay. The smaller, the faster the decay.? (weird, but looks like that)

    Methods
    -------
    __init__()
        Constuctor.
    apply_schedule()
        Applies the constant learning rate schedule.
    get_lr()
        Returns the latest learning rate.
    """

    def __init__(self, lr_initial, decay_steps, decay_rate):
        """ Constructor.
        Inherits everything from the LRSchedule class

        Parameters
        ----------
        lr_initial : float
            Initial, or base, learning rate.
        decay_steps : int
            The number of decay steps. The smaller, the faster the decay.
        decay_rate : float
            The rate of decay. The smaller, the faster the decay.? (weird, but looks like that)

        Notes
        -----
        None
        """
        super().__init__(lr_initial)
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate

    def apply_schedule(self, ):
        """ Applies the exponential decay learning rate schedule.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        Based on: https://keras.io/api/optimizers/learning_rate_schedules/exponential_decay/
        """
        self.lr = self.lr_initial * self.decay_rate ** (self.step / self.decay_steps)
        self.step += 1

    def get_lr(self, ):
        """ Returns the latest learning rate.

        Parameters
        ----------
        None

        Returns
        -------
        float
            The latest learning rate.

        Notes
        -----
        None
        """
        return self.lr


class LRCyclingSchedule(LRSchedule):
    """ Cyclical learning rate schedule.

    Attributes
    ----------
    lr_initial : float
        Initial, or base, learning rate.
    lr : float
        The latest learning rate.
    step : int
        Update step counter used for applying the learning rate schedule.
    lr_max : float
        The maximum learning rate.
    step_size : int
        The step size in number of update steps.
        A full cycle is 2 * step_size

    Methods
    -------
    __init__()
        Constuctor.
    apply_schedule()
        Applies the constant learning rate schedule.
    get_lr()
        Returns the latest learning rate.

    Notes
    -----
    Based on: Cyclical Learning Rates for Training Neural Networks
    Available at: https://arxiv.org/abs/1506.01186

    The schedule starts at lr_initial, goes to lr_max in step_size update steps,
    and then back to lr_initial in step_size update steps.
    A full cycle is 2*step_size update steps.
    """

    def __init__(self, lr_initial, lr_max, step_size):
        """ Constructor.
        Inherits everything from the LRSchedule class

        Parameters
        ----------
        lr_initial : float
            Initial, or base, learning rate.
        lr_max : float
            The maximum learning rate.
        step_size : int
            The step size in number of update steps.
            A full cycle is 2 * step_size

        Notes
        -----
        None
        """
        # self.lr_initial is lr_min, i.e.: the base lr
        super().__init__(lr_initial)
        self.lr_max = lr_max
        self.step_size = step_size

    def apply_schedule(self, ):
        """ Applies the cycling learning rate schedule.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        Based on: https://www.datacamp.com/community/tutorials/cyclical-learning-neural-nets
        """
        cycle = np.floor(1 + self.step / (2 * self.step_size))
        x = np.abs(self.step / self.step_size - 2 * cycle + 1)
        self.lr = self.lr_initial + (self.lr_max - self.lr_initial) * np.maximum(0, (1 - x))
        self.step += 1

    def get_lr(self, ):
        """ Returns the latest learning rate.

        Parameters
        ----------
        None

        Returns
        -------
        float
            The latest learning rate.

        Notes
        -----
        None
        """
        return self.lr
