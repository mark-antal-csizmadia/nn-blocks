from copy import deepcopy
import numpy as np


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

    def __init__(self, lr_schedule, repr_str):
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
        self.repr_str = repr_str

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

    def __repr__(self):
        return self.repr_str


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
    apply_grads(trainable_params, grads)
        Applies the gradient update rule to trainable params using gradients.
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
        repr_str = f"sgd with {lr_schedule.__repr__()}"
        super().__init__(lr_schedule, repr_str)

    def apply_grads(self, trainable_params, grads):
        """ Applies the gradient update rule to trainable params using gradients.

        Parameters
        ----------
        trainable_params : list
            The list of dictionaries of the trainable parameters of all layers of a model.
            At idx is the dictionary of trainable parameters of layer idx in the Model.layers list.

        grads : list
            The list of dictionaries of gradients of all parameters of all layers of a model.
            At idx is the dictionary of gradients of layer idx in the Model.layers list.

        Returns
        -------
        updated_trainable_params : list
            The list of dictionaries of the updated trainable parameters of all layers of a model.
            At idx is the dictionary of the updated trainable parameters of layer idx
            in the Model.layers list.

        Notes
        -----
        Iterates over layers in ascending order in the Model.layers list.

        Raises
        ------
        AssertionError
            If the lengths of trainable_weights and grads lists are not the same.
        """
        updated_trainable_params = deepcopy(trainable_params)

        assert len(trainable_params) == len(grads)

        for idx in range(len(trainable_params)):
            param_dict = deepcopy(trainable_params[idx])
            grad_dict = deepcopy(grads[idx])

            for p, g in zip(param_dict, grad_dict):
                updated_trainable_params[idx][p] = param_dict[p] - self.lr * grad_dict[g]

        return deepcopy(updated_trainable_params)


class AdaGradOptimizer(Optimizer):
    """ AdaGrad gradient descent optimizer.

    Attributes
    ----------
    lr_schedule : LRSchedule
        The learning rate schedule of the optimizer.
    lr : float
        The latest learning rate.
    first_call : bool
        If first call.
    epsilon : float
        Numerical stability constant.
    cache : list
        A list of dicts for cache of feature of param grads such
        as running mean of squared grads, m.

    Methods
    -------
    __init__()
        Constructor.
    apply_lr_schedule()
        Applies the learning rate schedule of the optimizer.
    get_lr()
        Returns the latest learning rate of the optimizer's learning rate schedule.
    apply_grads(trainable_params, grads)
        Applies the gradient update rule to trainable params using gradients.
    build_cache(trainable_params, grads):
        Build cache on first call.
    update_cache(trainable_params, grads):
        Update cache on call.
    get_opt_grad(trainable_params, grads):
        Get the optimizer specific grads computed using the cache.
    """

    def __init__(self, lr_schedule, epsilon=1e-6):
        """ Constructor.
        Inherits everything from the Optimizer class.

        Parameters
        ----------
        lr_schedule : LRSchedule
            The learning rate schedule of the optimizer.
        epsilon : float
            Numerical stability constant.

        Notes
        -----
        None
        """
        repr_str = f"adagrad with {lr_schedule.__repr__()}"
        super().__init__(lr_schedule, repr_str)
        self.first_call = True
        self.epsilon = epsilon
        self.cache = []

    def build_cache(self, trainable_params, grads):
        """ Build cache on first call.

        Parameters
        ----------
        trainable_params : list
            The list of dictionaries of the trainable parameters of all layers of a model.
            At idx is the dictionary of trainable parameters of layer idx in the Model.layers list.

        grads : list
            The list of dictionaries of gradients of all parameters of all layers of a model.
            At idx is the dictionary of gradients of layer idx in the Model.layers list.

        Returns
        -------
        None

        Notes
        -----
        self.cache is built after calling.
        """
        for idx in range(len(trainable_params)):
            param_dict = deepcopy(trainable_params[idx])
            grad_dict = deepcopy(grads[idx])
            m_dict = {}
            for p, g in zip(param_dict, grad_dict):
                m_dict[p] = np.zeros(param_dict[p].shape)
            self.cache.append(m_dict)

    def update_cache(self, trainable_params, grads):
        """ Update cache on call.

        Parameters
        ----------
        trainable_params : list
            The list of dictionaries of the trainable parameters of all layers of a model.
            At idx is the dictionary of trainable parameters of layer idx in the Model.layers list.

        grads : list
            The list of dictionaries of gradients of all parameters of all layers of a model.
            At idx is the dictionary of gradients of layer idx in the Model.layers list.

        Returns
        -------
        None

        Notes
        -----
        self.cache is updated after calling.
        """
        # asset not empty
        assert self.cache

        for idx in range(len(trainable_params)):
            param_dict = deepcopy(trainable_params[idx])
            grad_dict = deepcopy(grads[idx])
            m_dict = deepcopy(self.cache[idx])

            for p, g in zip(param_dict, grad_dict):
                m_dict[p] += np.power(grad_dict[g], 2)

            self.cache[idx] = deepcopy(m_dict)

    def get_opt_grad(self, trainable_params, grads):
        """ Get the optimizer specific grads computed using the cache.

        Parameters
        ----------
        trainable_params : list
            The list of dictionaries of the trainable parameters of all layers of a model.
            At idx is the dictionary of trainable parameters of layer idx in the Model.layers list.

        grads : list
            The list of dictionaries of gradients of all parameters of all layers of a model.
            At idx is the dictionary of gradients of layer idx in the Model.layers list.

        Returns
        -------
        opt_grads : list
            The list of dictionaries of the optimizer specifc gradients of all parameters of
            all layers of a model. At idx is the dictionary of gradients of layer idx in
            the Model.layers list.

        Notes
        -----
        None
        """
        # asset not empty
        assert self.cache

        opt_grads = deepcopy(grads)

        for idx in range(len(trainable_params)):
            param_dict = deepcopy(trainable_params[idx])
            grad_dict = deepcopy(grads[idx])
            m_dict = deepcopy(self.cache[idx])

            for p, g in zip(param_dict, grad_dict):
                opt_grads[idx][g] = grad_dict[g] / np.sqrt(m_dict[p] + self.epsilon)

        return deepcopy(opt_grads)

    def apply_grads(self, trainable_params, grads):
        """ Applies the gradient update rule to trainable params using gradients.

        Parameters
        ----------
        trainable_params : list
            The list of dictionaries of the trainable parameters of all layers of a model.
            At idx is the dictionary of trainable parameters of layer idx in the Model.layers list.

        grads : list
            The list of dictionaries of gradients of all parameters of all layers of a model.
            At idx is the dictionary of gradients of layer idx in the Model.layers list.

        Returns
        -------
        updated_trainable_params : list
            The list of dictionaries of the updated trainable parameters of all layers of a model.
            At idx is the dictionary of the updated trainable parameters of layer idx
            in the Model.layers list.

        Notes
        -----
        Iterates over layers in ascending order in the Model.layers list.

        Raises
        ------
        AssertionError
            If the lengths of trainable_weights and grads lists are not the same.
        """
        updated_trainable_params = deepcopy(trainable_params)

        assert len(trainable_params) == len(grads)

        if self.first_call:
            self.first_call = False
            self.build_cache(trainable_params, grads)

        self.update_cache(trainable_params, grads)
        opt_grads = self.get_opt_grad(trainable_params, grads)

        for idx in range(len(trainable_params)):
            param_dict = deepcopy(trainable_params[idx])
            grad_dict = deepcopy(grads[idx])
            opt_grad_dict = deepcopy(opt_grads[idx])

            for p, g in zip(param_dict, grad_dict):
                updated_trainable_params[idx][p] = param_dict[p] - self.lr * opt_grad_dict[g]

        return deepcopy(updated_trainable_params)

