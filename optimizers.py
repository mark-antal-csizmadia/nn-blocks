from copy import deepcopy


class Optimizer():
    """ Optimizer parent class.

    Attributes
    ----------
    lr_schedule : LRSchedule
        The learning rate schedule of the optimizer.
    lr : float
        The latest learning rate.

    Methods
    -------
    __init__()
        Constructor.
    apply_lr_schedule()
        Applies the learning rate schedule of the optimizer.
    get_lr()
        Returns the latest learning rate of the optimizer's learning rate schedule.
    """

    def __init__(self, lr_schedule):
        """ Constructor.

        Parameters
        ----------
        lr_schedule : LRSchedule
        The learning rate schedule of the optimizer.

        Notes
        -----
        None
        """
        self.lr_schedule = lr_schedule
        self.lr = self.lr_schedule.get_lr()

    def apply_lr_schedule(self, ):
        """ Applies the learning rate schedule of the optimizer.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        Updates self.lr
        """
        self.lr_schedule.apply_schedule()
        self.lr = self.lr_schedule.get_lr()

    def get_lr(self, ):
        """ Returns the latest learning rate of the optimizer's learning rate schedule.

        Parameters
        ----------
        None

        Returns
        -------
        lr : float
            The latest learning rate of the learning rate schedule of the optimizer.

        Notes
        -----
        Updates self.lr
        """
        return deepcopy(self.lr)


class SGDOptimizer(Optimizer):
    """ Stochastic gradient descent optimizer.

    Attributes
    ----------
    lr_schedule : LRSchedule
        The learning rate schedule of the optimizer.
    lr : float
        The latest learning rate.

    Methods
    -------
    __init__()
        Constructor.
    apply_lr_schedule()
        Applies the learning rate schedule of the optimizer.
    get_lr()
        Returns the latest learning rate of the optimizer's learning rate schedule.
    apply_grads(trainable_weights, grads)
        Applies the gradient update rule to trainable weights using gradients.
    """

    def __init__(self, lr_schedule):
        """ Constructor.
        Inherits everything from the Optimizer class.

        Parameters
        ----------
        lr_schedule : LRSchedule
            The learning rate schedule of the optimizer.

        Notes
        -----
        None
        """
        super().__init__(lr_schedule)

    def apply_grads(self, trainable_weights, grads):
        """ Applies the gradient update rule to trainable weights using gradients.

        Parameters
        ----------
        trainable_weights : list
            The list of dictionaries of the trainable parameters of all layers of a model.
            At idx is the dictionary of trainable parameters of layer idx in the Model.layers list.
            A list has two keys - w and b.

        grads : list
            The list of dictionaries of gradients of all parameters of all layers of a model.
            At idx is the dictionary of gradients of layer idx in the Model.layers list.
            A list has two keys - dw and db.

        Returns
        -------
        updated_trainable_weights : list
            The list of dictionaries of the updated trainable parameters of all layers of a model.
            At idx is the dictionary of the updated trainable parameters of layer idx
            in the Model.layers list.
            A list has two keys - w and b.

        Notes
        -----
        Iterates over layers in ascending order in the Model.layers list.

        Raises
        ------
        AssertionError
            If the lengths of trainable_weights and grads lists are not the same.
        """
        updated_trainable_weights = deepcopy(trainable_weights)

        assert len(trainable_weights) == len(grads)

        for idx in range(len(trainable_weights)):
            trainable_weight_dict = deepcopy(trainable_weights[idx])
            grad_dict = deepcopy(grads[idx])

            w = trainable_weight_dict["w"]
            b = trainable_weight_dict["b"]
            dw = grad_dict["dw"]
            db = grad_dict["db"]

            w -= self.lr * dw
            b -= self.lr * db

            trainable_weight_dict["w"] = deepcopy(w)
            trainable_weight_dict["b"] = deepcopy(b)

            updated_trainable_weights[idx] = deepcopy(trainable_weight_dict)

        return updated_trainable_weights
