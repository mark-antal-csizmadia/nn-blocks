import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pickle
from copy import deepcopy
import datetime
import sys
from itertools import product
import pandas as pd


# Data utils
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


# Grad check utils
def loss_func(model, loss, x, y, **param_dict):

    layer_idx = deepcopy(param_dict["layer_idx"])
    # w or b
    param_str = deepcopy(param_dict["param_str"])
    # np matrix
    param_val = deepcopy(param_dict["param_val"])

    trainable_weights = model.get_trainable_weights()
    trainable_weights[layer_idx][param_str] = deepcopy(param_val)
    model.set_trainable_weights(trainable_weights)

    scores = model.forward(x)
    #layers_reg_loss = model.get_reg_loss()
    l = loss.compute_loss(scores, y)

    return l


def get_num_gradient(model, loss, x, y, verbose, **param_dict):
    """
    a naive implementation of numerical gradient of f at x
    - f should be a function that takes a single argument
    - x is the point (numpy array) to evaluate the gradient at
    """

    # fx = f(x) # evaluate function value at original point
    l = loss_func(model, loss, x, y, **param_dict)

    param_val = deepcopy(param_dict["param_val"])
    grad = np.zeros(param_val.shape)
    h = 0.00001

    # iterate over all indexes in x

    if verbose:
        pbar = tqdm(total=param_val.size, file=sys.stdout)

    it = np.nditer(param_val, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:

        # evaluate function at x+h
        ix = it.multi_index
        # print(ix)

        old_value = param_val[ix]
        param_val[ix] = old_value + h # increment by h

        param_dict["param_val"] = deepcopy(param_val)

        # fxph = f(x) # evalute f(x + h)
        lxph = loss_func(model, loss, x, y, **param_dict)

        param_val[ix] = old_value - h # decrease by h

        param_dict["param_val"] = deepcopy(param_val)

        # fxmh = f(x) # evalute f(x - h)

        lxmh = loss_func(model, loss, x, y, **param_dict)

        param_val[ix] = old_value # restore to previous value (very important!)

        # compute the partial derivative
        # the slope
        grad[ix] = (lxph - lxmh) / ( 2 *h)
        it.iternext() # step to next dimension

        if verbose:
            pbar.update(1)

    if verbose:
        pbar.close()

    return grad


def grad_check_without_reg(model, loss, x, y, verbose, seed=None):
    # does not include w regularization in numerical grad computation
    assert x.shape[0] == y.shape[0], "x and y have different number of data points"
    print(f"starting grad check with {x.shape[0]} data points \n")
    print(model)
    print("\n")

    layer_n = len(model.layers)

    for layer_idx, trainable_weights_in_layer in enumerate(model.get_trainable_weights()):

        for param_str, param_val in trainable_weights_in_layer.items():
            model_new = deepcopy(model)

            trainable_weights = model_new.get_trainable_weights()

            np.random.seed(seed)
            new_param_val = np.random.normal(loc=0, scale=0.01, size=param_val.shape)

            param_dict = {
                "layer_idx": layer_idx,
                "param_str": param_str,
                "param_val": new_param_val
            }

            trainable_weights[layer_idx][param_str] = deepcopy(new_param_val)
            model_new.set_trainable_weights(trainable_weights)

            print(f"--layer: {layer_idx + 1}/{layer_n}, "
                  f"{param_str}.shape={param_val.shape} ({param_val.size} params)")

            grad_numerical = get_num_gradient(deepcopy(model_new), loss, x, y, verbose, **param_dict)

            scores = model_new.forward(x)
            assert model_new.get_reg_loss() == 0.0

            l = loss.compute_loss(scores, y)

            model_new.backward(loss.grad())

            grads_analytic = model_new.get_gradients()

            grad_analytic = deepcopy(grads_analytic[layer_idx]["d" + param_str])

            rel_error = np.abs(grad_analytic - grad_numerical) \
                        / (np.maximum(np.abs(grad_analytic), np.abs(grad_numerical)) + 10e-20)

            decimal = 6
            np.testing.assert_array_almost_equal(grad_numerical, grad_analytic, decimal=decimal)
            print(f"analytic and numerical grads are equal up to {decimal} decimals")
            print(f"max rel error={np.max(rel_error):.6e}")
            print(f"passed\n")

    print(f"completed grad check\n")


# All classes
# Activations
class Activation():
    """ Activation parent class.

    Attributes
    ----------
    cache : dict
        Run-time cache of attibutes such as gradients.

    Methods
    -------
    __init__()
        Constuctor.
    """

    def __init__(self, ):
        self.cache = {}


class ReLUActivation(Activation):
    """ ReLU activation.
    Can be followed by virtually anything.
    Inherits everything from class Activation.

    Attributes
    ----------
    cache : dict
        Run-time cache of attibutes such as gradients.

    Methods
    -------
    __init__()
        Constuctor.
    forward(z)
        Activates the linear transformation of the layer, and
        forward propagates activation. Activation is rectified linear.
    backward(g)
        Backpropagates incoming gradient into the layer, based on the rectified linear activation.
    __repr__()
        Returns the string representation of class.
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

    def forward(self, z):
        """ Activates the linear transformation of the layer, and
        forward propagates activation. Activation is rectified linear.

        Parameters
        ----------
        z : numpy.ndarray
            Linear transformation of layer.
            Shape is unknown here, but will usually be
            (batch size, this layer output dim = next layer input dim)

        Returns
        -------
        numpy.ndarray
            ReLU activation.

        Notes
        -----
        None
        """
        a = np.maximum(0, z)
        self.cache["a"] = deepcopy(a)
        return a

    def backward(self, g_in):
        """ Backpropagates incoming gradient into the layer, based on the rectified linear activation.

        Parameters
        ----------
        g_in : numpy.ndarray
            Incoming gradient to the activation.
            Shape is unknown here, but will usually be
            (batch size, this layer output dim = next layer input dim)

        Returns
        -------
        numpy.ndarray
            Gradient of activation.
            Shape is unknown here, but will usually be
            (batch size, this layer output dim = next layer input dim)

        Notes
        -----
        None
        """
        a = self.cache["a"]
        g_out = deepcopy(g_in)
        g_out[a <= 0] = 0.0
        return g_out

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
        repr_str = "relu"
        return repr_str


class SoftmaxActivation(Activation):
    """ Softmax activation.
    Usually activation of last layer and forward propagates into a CategoricalCrossEntropyLoss.
    Inherits everything from class Activation.

    Attributes
    ----------
    cache : dict
        Run-time cache of attibutes such as gradients.

    Methods
    -------
    __init__()
        Constuctor.
    forward(z)
        Activates the linear transformation of the layer, and
        forward propagates activation. Activation is softmax.
    backward(g)
        Backpropagates incoming gradient into the layer, based on the softmax activation.
    __repr__()
        Returns the string representation of class.
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

    def forward(self, z):
        """ Activates the linear transformation of the layer, and
        forward propagates activation. Activation is softmax.

        Parameters
        ----------
        z : numpy.ndarray
            Linear transformation of layer.
            Shape is unknown here, but will usually be
            (batch size, this layer output dim = number of classes)

        Returns
        -------
        numpy.ndarray
            Softmax activation.
            Shape is (batch size, this layer output dim = number of classes)

        Notes
        -----
        None
        """
        # avoid numeric instability
        z -= np.max(z, axis=1, keepdims=True)

        # get unnormalized probabilities
        # exp_scores.shape = (batch_size, K)
        exp_z = np.exp(z)

        # normalize them for each example
        # probs.shape = (batch_size, K)
        a = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        self.cache["a"] = deepcopy(a)

        return a

    def backward(self, g_in):
        """ Backpropagates incoming gradient into the layer, based on the softmax activation.

        Parameters
        ----------
        g_in : numpy.ndarray
            Incoming gradient to the activation.
            Shape is unknown here, but will usually be
            (batch size, )

        Returns
        -------
        numpy.ndarray
            Gradient of activation.
            Shape is unknown here, but will usually be
            (batch size, )

        Notes
        -----
        None
        """
        # g_in is y, y.shape = (batch_size,)
        # g_out.shape = (batch_size, K)
        n = g_in.shape[0]
        a = self.cache["a"]
        g_out = deepcopy(a)
        g_out[range(n), g_in] -= 1
        g_out /= n

        return g_out

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
        repr_str = "softmax"
        return repr_str


# Initializers
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
        repr_str = "Xavier ~ " + f"{self.coeff:.6f} x N({self.mean:.6f}, {self.std:.6f}^2)"
        return repr_str


# Dense layer

class Dense():
    """ Dense (fully-connected) layer class.


    Attributes
    ----------
    in_dim : int
        Input dimension.
    out_dim : int
        Output dimension.
    kernel_initializer : Initializer
        The weight parameter initializer.
    bias_initializer : Initializer
        The bias parameter initializer.
    kernel_regularizer : Regularizer
        The weight parameter regularizer.
    activation : Activation
        Layer activation.
    w : numpy.ndarray
        The weight parameters, of shape (in_dim, out_dim)
    b : numpy.ndarray
        The bias parameters, of shape (1, out_dim)
    cache : dict
        The run-time cache for storing activations, etc.
    grads : dict
        The run-time cache for storing gradients.

    Methods
    -------
    __init__(in_dim, out_dim, kernel_initializer, bias_initializer, kernel_regularizer, activation)
        Constuctor.
    get_w()
        Returns the weight parameters.
    get_b()
        Returns the bias parameters.
    set_w()
        Sets the weight parameters.
    set_b()
        Sets the bias parameters.
    get_dw()
        Returns the gradients of weight parameters.
    get_db()
        Returns the gradients bias parameters.
    get_reg_loss_w()
        Returns the regularization loss of the weight parameters.
    get_reg_grad_w()
        Returns the regularization gradient of the weight parameters.
    forward(x)
        Forward-propagates signals through the layer and its activation.
    backward(g_in)
        Back-propagates gradients through the the activation of the layer and then the layer.
    __repr__()
        Returns the string representation of class.
    """

    def __init__(self, in_dim, out_dim, kernel_initializer, bias_initializer, kernel_regularizer, activation):
        """ Constructor.

        Parameters
        ----------
        in_dim : int
            Input dimension.
        out_dim : int
            Output dimension.
        kernel_initializer : Initializer
            The weight parameter initializer.
        bias_initializer : Initializer
            The bias parameter initializer.
        kernel_regularizer : Regularizer
            The weight parameter regularizer.
        activation : Activation
            Layer activation.

        Notes
        -----
        None
        """
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

        self.w = kernel_initializer.initialize(size=(in_dim, out_dim))
        self.b = bias_initializer.initialize(size=(1, out_dim))

        self.kernel_regularizer = kernel_regularizer

        self.activation = activation

        self.cache = {}
        self.grads = {}

    def get_w(self, ):
        """ Returns the weight parameters.

        Parameters
        ----------
        None

        Returns
        -------
        numpy.ndarray
            The weight parameters.

        Notes
        -----
        None
        """
        return deepcopy(self.w)

    def get_b(self, ):
        """ Returns the bias parameters.

        Parameters
        ----------
        None

        Returns
        -------
        numpy.ndarray
            The bias parameters.

        Notes
        -----
        None
        """
        return deepcopy(self.b)

    def set_w(self, w):
        """ Sets the weight parameters.

        Parameters
        ----------
        w : numpy.ndarray
            The weight parameters.

        Returns
        -------
        None

        Notes
        -----
        None
        """
        self.w = deepcopy(w)

    def set_b(self, b):
        """ Sets the bias parameters.

        Parameters
        ----------
        b : numpy.ndarray
            The bias parameters.

        Returns
        -------
        None

        Notes
        -----
        None
        """
        self.b = deepcopy(b)

    def get_dw(self, ):
        """ Returns the gradients of weight parameters.

        Parameters
        ----------
        None

        Returns
        -------
        ret : None or numpy.ndarray
            The gradients of weight parameters, or None if does not exist yet.

        Notes
        -----
        None
        """
        if "dw" in self.grads.keys():
            dw = self.grads["dw"]
            ret = deepcopy(dw)
        else:
            ret = None

        return ret

    def get_db(self, ):
        """ Returns the gradients of bias parameters.

        Parameters
        ----------
        None

        Returns
        -------
        ret : None or numpy.ndarray
            The gradients of bias parameters, or None if does not exist yet.

        Notes
        -----
        None
        """
        if "db" in self.grads.keys():
            db = self.grads["db"]
            ret = deepcopy(db)
        else:
            ret = None

        return ret

    def get_reg_loss_w(self, ):
        """ Returns the regularization loss of the weight parameters.

        Parameters
        ----------
        None

        Returns
        -------
        float
            The regularization loss of the weight parameters

        Notes
        -----
        None
        """
        if self.kernel_regularizer is None:
            return 0.0
        else:
            return self.kernel_regularizer.loss(self.w)

    def get_reg_grad_w(self, ):
        """ Returns the regularization gradient of the weight parameters.

        Parameters
        ----------
        None

        Returns
        -------
        float or numpy.ndarray
            Returns the regularization gradient of the weight parameters.
            Float 0.0 if does not exist yet - since added later doesn't matter if
            0.0 is float or matrix.

        Notes
        -----
        None
        """
        if self.kernel_regularizer is None:
            return 0.0
        else:
            return self.kernel_regularizer.grad(self.w)

    def forward(self, x):
        """ Forward-propagates signals through the layer and its activation.

        Parameters
        ----------
        x : numpy.ndarray
            Input data to layer of shape (batch_size, in_dim).

        Returns
        -------
        a : numpy.ndarray
            Activation of linear transformation, of shape (batch_size, out_dim).

        Notes
        -----
        x.shape = (batch_size, in_dim)
        self.w.shape=(in_dim, out_dim)
        self.b.shape=(1, out_dim)
        z.shape = (batch_size, out_dim)
        a.shape = (batch_size, out_dim)
        """
        self.cache["x"] = deepcopy(x)
        z = np.dot(x, self.w) + self.b
        a = self.activation.forward(z)

        return a

    def backward(self, g_in):
        """ Back-propagates gradients through the the activation of the layer and then the layer.

        Parameters
        ----------
        g_in : numpy.ndarray
            Incoming (from later layers or losses) gradients, of shape (batch_size, out_dim).

        Returns
        -------
        g_out : numpy.ndarray
            Outgoing (to previous layers, or input data) gradients, of shape (batch_size, in_dim).

        Notes
        -----
        g_in.shape = (batch_size, out_dim)
        self.cache["x"].shape = (batch_size, in_dim)
        self.w.shape=(in_dim, out_dim)
        self.b.shape=(1, out_dim)
        dw.shape=(in_dim, out_dim)
        db.shape=(1, out_dim)
        g_out.shape = (batch_size, in_dim)
        """
        x = self.cache["x"]
        g_a = self.activation.backward(g_in)

        dw = np.dot(deepcopy(x).T, g_a)
        dw += self.get_reg_grad_w()

        db = np.sum(g_a, axis=0, keepdims=True)

        self.grads["dw"] = deepcopy(dw)
        self.grads["db"] = deepcopy(db)

        g_out = np.dot(g_a, self.w.T)

        return g_out

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
        repr_str = "dense: \n" \
                   + "\t w -- init:" + self.kernel_initializer.__repr__() \
                   + ", reg: " + self.kernel_regularizer.__repr__() + "\n" \
                   + "\t b -- init: " + self.bias_initializer.__repr__() + "\n" \
                   + "\t activation: " + self.activation.__repr__() + "\n"
        return repr_str


# Losses
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

    def __init__(self, ):
        """ Constructor.

        Parameters
        ----------
        None

        Notes
        -----
        None
        """
        self.cache = {}


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


# Learning rate schedules
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
        Constructor.
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


# Perfromance metrics
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

    def get_metrics(self, y, y_hat):
        """ Computes the accuracy of inferred numerical labels when compared to their true counterparts.

        Parameters
        ----------
        y : numpy.ndarray
            True labels.
            Shape is (number of data points, )
        y : numpy.ndarray
            True labels.
            Shape is (number of data points, )

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
        assert y.shape == y_hat.shape

        n = y.shape[0]
        return np.where(y_hat == y)[0].size / n


# Model class

class Model():
    """ Model class.

    Attributes
    ----------
    layers : list
        List of layers of model.
    reg_loss : float
        The sum of the regularization losses of all layers of the model.
    compiled : bool
        Flag showing if the model is compiled.
    optimizer : None or Optimizer
        The optimizer used for fitting the model to the data.
    loss : None or Loss
        The loss function of the optimization.
    metrics_dict : None or dict
        The dictionary of the training and validation metric values over training.
    loss_dict : None or dict
        The dictionary of the training and validation loss values over training.
    cost_dict : None or dict
        The dictionary of the training and validation cost values over training.
        Note that cost = data loss + regularization loss
    metrics : None or list
        The list of metrics for evaluating the model during training and validation over training.
    lr_dict : None or dict
        The dictionary of the learning rate values over update steps.

    Methods
    -------
    __init__(layers)
        Constuctor.
    forward(x)
        Forward propagates signal through the model.
    backward(y)
        Back-propagates signal through the model.
    get_reg_loss()
        Returns the overall regularization loss of the layers in the model.
    get_gradients()
        Returns the gradients of all parameters of all layers.
    get_trainable_weights()
        Returns all trainable parameters of all layers.
    set_trainable_weights(trainable_weights)
        Sets all trainable parameters of all layers.
    compile_model(optimizer, loss, metrics)
        Compiles the model.
    fit(x_train, y_train, x_val, y_val, n_epochs, batch_size)
        Fits the model to the data.
    __repr__()
        Returns the string representation of class.


    """

    def __init__(self, layers):
        """ Constructor.

        Parameters
        ----------
        layers : list
            List of layers of model.

        Notes
        -----
        None
        """
        self.layers = layers
        self.reg_loss = 0.0
        self.compiled = False
        self.optimizer = None
        self.loss = None
        self.metrics_dict = None
        self.loss_dict = None
        self.cost_dict = None
        self.metrics = None
        self.lr_dict = None

    def forward(self, x):
        """ Forward propagates signal through the model.

        Parameters
        ----------
        x : numpy.ndarray
            Input data to model, shape is (batch_size, in_dim)
            where in_dim is the input dimension of the first layer of the model.

        Returns
        -------
        scores : numpy.ndarray
            Activation of last layer of the model - the scores of the network.
            Shape is (batch_size, out_dim) where out_dim is the output
            dimension of the last layer of the model - usually same as
            the number of classes.

        Notes
        -----
        Iterates over layers in ascending order in the self.layers list.
        """
        scores = deepcopy(x)

        self.reg_loss = 0.0

        for layer in self.layers:
            scores_temp = layer.forward(scores)
            scores = deepcopy(scores_temp)
            self.reg_loss += layer.get_reg_loss_w()

        return scores

    def backward(self, y):
        """ Back-propagates signal through the model.

        Parameters
        ----------
        y : numpy.ndarray
            Labels of the input data to model, shape is (batch_size, ).

        Returns
        -------
        None

        Notes
        -----
        Iterates over layers in descending order in the self.layers list.
        """
        g = deepcopy(y)

        for layer in list(reversed(self.layers)):
            g_temp = layer.backward(g)
            g = deepcopy(g_temp)

    def get_reg_loss(self, ):
        """ Returns the overall regularization loss of the layers in the model.

        Parameters
        ----------
        None

        Returns
        -------
        float
            The sum of the regularization losses of all layers of the model.

        Notes
        -----
        None
        """
        return self.reg_loss

    def get_gradients(self, ):
        """ Returns the gradients of all parameters of all layers.

        Parameters
        ----------
        None

        Returns
        -------
        grads : list
            The list of dictionaries of gradients of all parameters of all layers of the model.
            At idx is the dictionary of gradients of layer idx in the self.layers list.
            A list has two keys - dw and db.

        Notes
        -----
        Iterates over layers in ascending order in the self.layers list.
        """
        grads = []
        for idx, layer in enumerate(self.layers):
            dw = layer.get_dw()
            db = layer.get_db()
            grads.append({"dw": deepcopy(dw), "db": deepcopy(db)})

        return deepcopy(grads)

    def get_trainable_weights(self, ):
        """ Returns all trainable parameters of all layers.

        Parameters
        ----------
        None

        Returns
        -------
        trainable_weights : list
            The list of dictionaries of the trainable parameters of all layers of the model.
            At idx is the dictionary of trainable parameters of layer idx in the self.layers list.
            A list has two keys - w and b.

        Notes
        -----
        Iterates over layers in ascending order in the self.layers list.
        """
        trainable_weights = []
        for idx, layer in enumerate(self.layers):
            w = layer.get_w()
            b = layer.get_b()
            trainable_weights.append({"w": deepcopy(w), "b": deepcopy(b)})

        return deepcopy(trainable_weights)

    def set_trainable_weights(self, trainable_weights):
        """ Sets all trainable parameters of all layers.

        Parameters
        ----------
        trainable_weights : list
            The list of dictionaries of the trainable parameters of all layers of the model.
            At idx is the dictionary of trainable parameters of layer idx in the self.layers list.
            A list has two keys - w and b.

        Returns
        -------
        None

        Notes
        -----
        Iterates over layers in ascending order in the self.layers list.
        """
        for idx, layer in enumerate(self.layers):
            trainable_weight_dict = deepcopy(trainable_weights[idx])
            w = trainable_weight_dict["w"]
            b = trainable_weight_dict["b"]
            layer.set_w(deepcopy(w))
            layer.set_b(deepcopy(b))

    def compile_model(self, optimizer, loss, metrics):
        """ Compiles the model.

        Parameters
        ----------
        optimizer : None or Optimizer
            The optimizer used for fitting the model to the data.
        loss : None or Loss
            The loss function of the optimization.
        metrics : None or list
            The list of metrics for evaluating the model during training and validation over training.

        Returns
        -------
        None

        Notes
        -----
        Sets self.compiled to True. If self.compiled is not called, self.fit will raise AssertionError.
        """
        self.optimizer = optimizer
        self.loss = loss

        metrics_train = {metric.name + "_train": [] for metric in metrics}
        metrics_val = {metric.name + "_val": [] for metric in metrics}
        self.metrics_dict = {**metrics_train, **metrics_val}
        self.loss_dict = {"loss_train": [], "loss_val": []}
        self.cost_dict = {"cost_train": [], "cost_val": []}
        self.metrics = metrics

        self.lr_dict = {"lr": []}

        self.compiled = True

    def fit(self, x_train, y_train, x_val, y_val, n_epochs, batch_size):
        """ Fits the model to the data.

        Parameters
        ----------
        x_train : numpy.ndarray
            Training data to model of shape (batch_size, in_dim) where in_dim is
            the input dimension of the first layer of the first layer.
        y_train : numpy.ndarray
            True labels of training data.
            Shape is (batch_size, )
        x_val : numpy.ndarray
            Validation data to model of shape (batch_size, in_dim) where in_dim is
            the input dimension of the first layer of the first layer.
        y_val : numpy.ndarray
            True labels of validation data.
            Shape is (batch_size, )
        n_epochs : int
            The number of epochs to train for.
        batch_size : int
            The batch size of the mini-batch gradient descent algorithm.
            x_train.shape[0] has to be divisible by batch_size

        Returns
        -------
        dict
            The history of training and validation loss, metrics, and learning rates.
            dict is {**self.metrics_dict, **self.loss_dict, **self.lr_dict}

        Notes
        -----
        None

        Raises
        ------
        AssertionError
            If the model has not yet been complied with the self.compiled method.
        """
        assert self.compiled, "Model has to be compiled before fitting."

        for n_epoch in range(n_epochs):
            print(f"starting epoch: {n_epoch + 1} ...")

            # Shuffle data
            indices = np.arange(x_train.shape[0])
            np.random.shuffle(indices)
            x_train = x_train[indices]
            y_train = y_train[indices]

            n_batch = int(x_train.shape[0] / batch_size)

            batches = tqdm(range(n_batch), file=sys.stdout)

            for b in batches:
                batches.set_description(f"batch {b + 1}/{n_batch}")
                x_batch = x_train[b * batch_size:(b + 1) * batch_size]
                y_batch = y_train[b * batch_size:(b + 1) * batch_size]

                scores = self.forward(x_batch)

                layers_reg_loss = self.get_reg_loss()
                data_loss = self.loss.compute_loss(scores, y_batch)

                cost = data_loss + layers_reg_loss

                self.backward(self.loss.grad())

                trainable_weights = \
                    self.optimizer.apply_grads(trainable_weights=self.get_trainable_weights(),
                                               grads=self.get_gradients())

                self.set_trainable_weights(trainable_weights)

                # should I do it here?
                self.optimizer.apply_lr_schedule()

            scores_train = self.forward(x_train)

            layers_reg_loss_train = self.get_reg_loss()
            data_loss_train = self.loss.compute_loss(scores_train, y_train)
            cost_train = data_loss_train + layers_reg_loss_train

            y_hat_train = np.argmax(scores_train, axis=1)

            # acc_train = self.metrics.accuracy.get_accuracy(y_train, y_hat_train)

            scores_val = self.forward(x_val)

            layers_reg_loss_val = self.get_reg_loss()
            data_loss_val = self.loss.compute_loss(scores_val, y_val)
            cost_val = data_loss_val + layers_reg_loss_val

            # n_val = y_val.shape[0]
            y_hat_val = np.argmax(scores_val, axis=1)

            self.loss_dict["loss_train"].append(data_loss_train)
            self.loss_dict["loss_val"].append(data_loss_val)
            self.cost_dict["cost_train"].append(cost_train)
            self.cost_dict["cost_val"].append(cost_val)

            train_str = f"train loss = {data_loss_train} / train cost = {cost_train}"
            val_str = f"val loss = {data_loss_val} / val cost = {cost_val}"

            for metrics in self.metrics:
                metrics_value_train = metrics.get_metrics(y_train, y_hat_train)
                self.metrics_dict[metrics.name + "_train"].append(metrics_value_train)
                train_str += f", train {metrics.name} = {metrics_value_train}"

                metrics_value_val = metrics.get_metrics(y_val, y_hat_val)
                self.metrics_dict[metrics.name + "_val"].append(metrics_value_val)
                val_str += f", val {metrics.name} = {metrics_value_val}"

            self.lr_dict["lr"].append(self.optimizer.get_lr())
            # acc_val = self.metrics.accuracy.get_accuracy(y_val, y_hat_val)

            print(f"epoch {n_epoch + 1}/{n_epochs} \n "
                  f"\t -- {train_str} \n"
                  f"\t -- {val_str} \n\n")

            # self.optimizer.apply_lr_schedule()

        return {**self.metrics_dict, **self.loss_dict, **self.cost_dict, **self.lr_dict}

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
        repr_str = "model summary: \n"
        for idx, layer in enumerate(self.layers):
            repr_str = repr_str + f"layer {idx}: " + layer.__repr__() + "\n"

        return repr_str


# Optimizers

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


# Regularizers

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


# Hyper-param searching tuner
class Tuner():
    def __init__(self, build_model, objective, iterations=1, **params):
        # objective is of Metrics for now
        self.build_model = build_model
        self.objective = objective
        self.iterations = iterations
        self.params = params
        self.params_product = list(product(*params.values()))
        self.params_names = list(params.keys())

    def search(self, x_train, y_train, x_val, y_val, n_epochs, batch_size):
        # list of tuples = list(product([1,2,3],[3,4]))
        # for tuple in list:
        # rows in final df
        rows = []

        # params_product = tqdm(self.params_product, file=sys.stdout)

        n_prod = len(self.params_product)

        for idx_prod, prod in enumerate(self.params_product):

            params = {}
            for idx, param_name in enumerate(self.params_names):
                params[param_name] = prod[idx]
            # print(params)
            # print(n_prod)

            # if more than 1 iterations
            objective_list = []

            for it in range(self.iterations):
                print("*" * 5)
                print(f"tuner: {idx_prod + 1}/{n_prod} config (iter: {it + 1}/{self.iterations})")
                # build_model with tuple params
                model = self.build_model(seed=200, **params)
                # fit model
                history = model.fit(x_train, y_train, x_val, y_val, n_epochs, batch_size)
                # meaasure objective on model
                scores_val = model.forward(x_val)
                y_hat_val = np.argmax(scores_val, axis=1)
                objective_val = self.objective.get_metrics(y_val, y_hat_val)
                # save objective in list
                objective_list.append(objective_val)

            # average objective in list
            objective_mean = np.array(objective_list).mean()
            # save tuple of params and objective as dict
            objective_dict = {self.objective.name: objective_mean}
            row_dict = {**params, **objective_dict}
            rows.append(row_dict)
            print("*" * 5 + "\n")

        # df from list of dicts of params and objective val
        df = pd.DataFrame(data=rows)

        # save to csv
        date_string = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")
        path = os.path.join("tuner_results", date_string + ".csv")

        df.to_csv(path, encoding='utf-8', index=False)

        # argmax across rows and return best params as dict (~**params)
        best_params = dict(df.loc[df[self.objective.name].idxmax()])
        best_objective = best_params.pop(self.objective.name)

        return best_objective, best_params


# Grad check func
def test_grad_check(x, y, seed=np.random.randint(low=1, high=300)):
    params = {"coeff": 1.0, "mean": 0.0, "std": None}

    in_dim = x.shape[1]
    out_dim = 10
    mid_dim = 50

    dense_1 = \
        Dense(in_dim=in_dim, out_dim=mid_dim,
              kernel_initializer=XavierInitializer(seed=seed, **params),
              bias_initializer=XavierInitializer(seed=seed + 1, **params),
              kernel_regularizer=None,
              activation=ReLUActivation()
              )

    dense_2 = \
        Dense(in_dim=mid_dim, out_dim=out_dim,
              kernel_initializer=XavierInitializer(seed=seed + 2, **params),
              bias_initializer=XavierInitializer(seed=seed + 3, **params),
              kernel_regularizer=None,
              activation=SoftmaxActivation()
              )

    layers = [
        dense_1,
        dense_2
    ]

    model = Model(layers)
    print(model)

    loss = CategoricalCrossEntropyLoss()

    verbose = True
    grad_check_without_reg(model, loss, x, y, verbose, seed=seed + 1)

    print("test_grad_check passed")


# Model build for Tuner
def build_model_func(seed=200, **params):
    assert "reg_rate_l2" in params.keys()
    reg_rate_l2 = params["reg_rate_l2"]

    params = {"coeff": 1.0, "mean": 0.0, "std": None}

    # reg_rate_l2 = 0.025

    in_dim = 3072
    out_dim = 10
    mid_dim = 50

    # seed = 200

    dense_1 = \
        Dense(in_dim=in_dim, out_dim=mid_dim,
              kernel_initializer=XavierInitializer(seed=seed, **params),
              bias_initializer=XavierInitializer(seed=seed + 1, **params),
              kernel_regularizer=L2Regularizer(reg_rate=reg_rate_l2),
              activation=ReLUActivation()
              )

    dense_2 = \
        Dense(in_dim=mid_dim, out_dim=out_dim,
              kernel_initializer=XavierInitializer(seed=seed + 2, **params),
              bias_initializer=XavierInitializer(seed=seed + 3, **params),
              kernel_regularizer=L2Regularizer(reg_rate=reg_rate_l2),
              activation=SoftmaxActivation()
              )

    layers = [
        dense_1,
        dense_2
    ]

    model = Model(layers)
    print(model)

    loss = CategoricalCrossEntropyLoss()

    # assignment:
    # n_epochs = 4
    # batch_size = 100

    lr_initial = 1e-5
    lr_max = 1e-1
    step_size = 900
    lr_schedule = LRCyclingSchedule(lr_initial, lr_max, step_size)
    optimizer = SGDOptimizer(lr_schedule=lr_schedule)

    metrics = [AccuracyMetrics()]

    model.compile_model(optimizer, loss, metrics)
    # history = model.fit(x_train, y_train, x_val, y_val, n_epochs, batch_size)

    return model


# Sampling lambda
def coarse_custom(n):
    l_min = -5
    l_max = -1
    # np.random.seed(seed)

    return [10 ** (l_min + (l_max - l_min) * np.random.uniform(low=0, high=1)) for i in range(n)]


def coarse_to_fine_custom(best_via_coarse, n):
    half_interval = 0.2
    low = best_via_coarse * (1 - half_interval)
    high = best_via_coarse * (1 + half_interval)

    return [np.random.uniform(low=low, high=high) for i in range(n)]


# Plotting stuff
def plot_losses(history):
    plt.plot(history["loss_train"], label="train")
    plt.plot(history["loss_val"], label="val")
    plt.grid()
    plt.title("Loss vs. epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    path = "losses.png"
    plt.savefig(path)
    plt.show()


def plot_costs(history):
    plt.plot(history["cost_train"], label="train")
    plt.plot(history["cost_val"], label="val")
    plt.grid()
    plt.title("Cost vs. epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Cost")
    plt.legend()
    path = "costs.png"
    plt.savefig(path)
    plt.show()


def plot_accuracies(history):
    plt.plot(history["accuracy_train"], label="train")
    plt.plot(history["accuracy_val"], label="val")
    plt.grid()
    plt.title("Accuracy vs. epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    path = "accuracies.png"
    plt.savefig(path)
    plt.show()


def plot_lr(history):
    plt.plot(history["lr"], label="lr")
    plt.grid()
    plt.title("Learning rate vs. epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Learning rate")
    plt.legend()
    path = "lrs.png"
    plt.savefig(path)
    plt.show()


def part_1_grad_check():
    # train set is batch 1, val set is batch 2, test set is test
    path = os.path.join("data", "data_batch_1")
    x_train_img, y_train = load_cfar10_batch(path)

    path = os.path.join("data", "data_batch_2")
    x_val_img, y_val = load_cfar10_batch(path)

    path = os.path.join("data", "test_batch")
    x_test_img, y_test = load_cfar10_batch(path)

    # check counts in datasets
    print(f"train set shape: {x_train_img.shape}, "
          f"val set shape: {x_val_img.shape}, test set shape: {x_test_img.shape}")
    print(f"train labels shape: {y_train.shape},"
          f" val labels shape: {y_val.shape}, test labels shape: {y_test.shape}")

    # assert balanced dataset
    train_counts = np.unique(y_train, return_counts=True)[1]
    train_ratios = train_counts / train_counts.sum()

    val_counts = np.unique(y_val, return_counts=True)[1]
    val_ratios = val_counts / val_counts.sum()

    test_counts = np.unique(y_test, return_counts=True)[1]
    test_ratios = test_counts / test_counts.sum()

    # np.testing.assert_array_equal(train_ratios, val_ratios)
    # np.testing.assert_array_equal(val_ratios, test_ratios)

    np.testing.assert_allclose(train_ratios, val_ratios, rtol=1e-1, atol=0)
    np.testing.assert_allclose(val_ratios, test_ratios, rtol=1e-1, atol=0)

    x_train_un = x_train_img.reshape(x_train_img.shape[0], -1)
    x_val_un = x_val_img.reshape(x_val_img.shape[0], -1)
    x_test_un = x_test_img.reshape(x_test_img.shape[0], -1)

    x_train = x_train_un / 255.
    x_val = x_val_un / 255.
    x_test = x_test_un / 255.

    mean = np.mean(x_train, axis=0).reshape(1, x_train.shape[1])
    std = np.std(x_train, axis=0).reshape(1, x_train.shape[1])

    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std

    # grad check
    test_grad_check(x_train[:20, :10], y_train[:20])


def part_2_replicate_fig_3():
    # train set is batch 1, val set is batch 2, test set is test

    path = os.path.join("data", "data_batch_1")
    x_train_img, y_train = load_cfar10_batch(path)

    path = os.path.join("data", "data_batch_2")
    x_val_img, y_val = load_cfar10_batch(path)

    path = os.path.join("data", "test_batch")
    x_test_img, y_test = load_cfar10_batch(path)

    # check counts in datasets
    print(f"train set shape: {x_train_img.shape}, "
          f"val set shape: {x_val_img.shape}, test set shape: {x_test_img.shape}")
    print(f"train labels shape: {y_train.shape},"
          f" val labels shape: {y_val.shape}, test labels shape: {y_test.shape}")

    # assert balanced dataset
    train_counts = np.unique(y_train, return_counts=True)[1]
    train_ratios = train_counts / train_counts.sum()

    val_counts = np.unique(y_val, return_counts=True)[1]
    val_ratios = val_counts / val_counts.sum()

    test_counts = np.unique(y_test, return_counts=True)[1]
    test_ratios = test_counts / test_counts.sum()

    # np.testing.assert_array_equal(train_ratios, val_ratios)
    # np.testing.assert_array_equal(val_ratios, test_ratios)

    np.testing.assert_allclose(train_ratios, val_ratios, rtol=1e-1, atol=0)
    np.testing.assert_allclose(val_ratios, test_ratios, rtol=1e-1, atol=0)

    x_train_un = x_train_img.reshape(x_train_img.shape[0], -1)
    x_val_un = x_val_img.reshape(x_val_img.shape[0], -1)
    x_test_un = x_test_img.reshape(x_test_img.shape[0], -1)

    x_train = x_train_un / 255.
    x_val = x_val_un / 255.
    x_test = x_test_un / 255.

    mean = np.mean(x_train, axis=0).reshape(1, x_train.shape[1])
    std = np.std(x_train, axis=0).reshape(1, x_train.shape[1])

    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std

    coeff = 1.0
    mean = 0.0
    std = 0.01
    params = {"coeff": coeff, "mean": mean, "std": None}

    reg_rate_l2 = 0.01

    in_dim = x_train.shape[1]
    out_dim = 10
    mid_dim = 50

    seed = 200

    dense_1 = \
        Dense(in_dim=in_dim, out_dim=mid_dim,
              kernel_initializer=XavierInitializer(seed=seed, **params),
              bias_initializer=XavierInitializer(seed=seed + 1, **params),
              kernel_regularizer=L2Regularizer(reg_rate=reg_rate_l2),
              activation=ReLUActivation()
              )

    dense_2 = \
        Dense(in_dim=mid_dim, out_dim=out_dim,
              kernel_initializer=XavierInitializer(seed=seed + 2, **params),
              bias_initializer=XavierInitializer(seed=seed + 3, **params),
              kernel_regularizer=L2Regularizer(reg_rate=reg_rate_l2),
              activation=SoftmaxActivation()
              )

    layers = [
        dense_1,
        dense_2
    ]

    model = Model(layers)
    print(model)

    loss = CategoricalCrossEntropyLoss()

    n_epochs = 10
    batch_size = 100

    lr_initial = 1e-5
    lr_max = 1e-1
    step_size = 500
    lr_schedule = LRCyclingSchedule(lr_initial, lr_max, step_size)
    optimizer = SGDOptimizer(lr_schedule=lr_schedule)

    metrics = [AccuracyMetrics()]

    model.compile_model(optimizer, loss, metrics)
    history = model.fit(x_train, y_train, x_val, y_val, n_epochs, batch_size)

    plot_losses(history)
    plot_costs(history)
    plot_accuracies(history)
    plot_lr(history)

    scores_test = model.forward(x_test)
    y_hat_test = np.argmax(scores_test, axis=1)
    acc_test = AccuracyMetrics().get_metrics(y_test, y_hat_test)

    print(f"test acc: {acc_test}")


def part_3_replicate_fig_4():
    # train set is batch 1, val set is batch 2, test set is test

    path = os.path.join("data", "data_batch_1")
    x_train_img, y_train = load_cfar10_batch(path)

    path = os.path.join("data", "data_batch_2")
    x_val_img, y_val = load_cfar10_batch(path)

    path = os.path.join("data", "test_batch")
    x_test_img, y_test = load_cfar10_batch(path)

    # check counts in datasets
    print(f"train set shape: {x_train_img.shape}, "
          f"val set shape: {x_val_img.shape}, test set shape: {x_test_img.shape}")
    print(f"train labels shape: {y_train.shape},"
          f" val labels shape: {y_val.shape}, test labels shape: {y_test.shape}")

    # assert balanced dataset
    train_counts = np.unique(y_train, return_counts=True)[1]
    train_ratios = train_counts / train_counts.sum()

    val_counts = np.unique(y_val, return_counts=True)[1]
    val_ratios = val_counts / val_counts.sum()

    test_counts = np.unique(y_test, return_counts=True)[1]
    test_ratios = test_counts / test_counts.sum()

    # np.testing.assert_array_equal(train_ratios, val_ratios)
    # np.testing.assert_array_equal(val_ratios, test_ratios)

    np.testing.assert_allclose(train_ratios, val_ratios, rtol=1e-1, atol=0)
    np.testing.assert_allclose(val_ratios, test_ratios, rtol=1e-1, atol=0)

    x_train_un = x_train_img.reshape(x_train_img.shape[0], -1)
    x_val_un = x_val_img.reshape(x_val_img.shape[0], -1)
    x_test_un = x_test_img.reshape(x_test_img.shape[0], -1)

    x_train = x_train_un / 255.
    x_val = x_val_un / 255.
    x_test = x_test_un / 255.

    mean = np.mean(x_train, axis=0).reshape(1, x_train.shape[1])
    std = np.std(x_train, axis=0).reshape(1, x_train.shape[1])

    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std

    coeff = 1.0
    mean = 0.0
    std = 0.01
    params = {"coeff": coeff, "mean": mean, "std": None}

    # reg_rate_l2 = 0.1
    reg_rate_l2 = 0.025

    in_dim = x_train.shape[1]
    out_dim = 10
    mid_dim = 50

    seed = 200

    dense_1 = \
        Dense(in_dim=in_dim, out_dim=mid_dim,
              kernel_initializer=XavierInitializer(seed=seed, **params),
              bias_initializer=XavierInitializer(seed=seed + 1, **params),
              kernel_regularizer=L2Regularizer(reg_rate=reg_rate_l2),
              activation=ReLUActivation()
              )

    dense_2 = \
        Dense(in_dim=mid_dim, out_dim=out_dim,
              kernel_initializer=XavierInitializer(seed=seed + 2, **params),
              bias_initializer=XavierInitializer(seed=seed + 3, **params),
              kernel_regularizer=L2Regularizer(reg_rate=reg_rate_l2),
              activation=SoftmaxActivation()
              )

    layers = [
        dense_1,
        dense_2
    ]

    model = Model(layers)
    print(model)

    loss = CategoricalCrossEntropyLoss()

    n_epochs = 50
    batch_size = 100

    # lr_initial = 0.01
    # lr_schedule = LRConstantSchedule(lr_initial)
    # decay_steps = n_epochs * 2
    # decay_rate = 0.9
    # lr_schedule = LRExponentialDecaySchedule(lr_initial, decay_steps, decay_rate)

    lr_initial = 1e-5
    lr_max = 1e-1
    step_size = 800
    lr_schedule = LRCyclingSchedule(lr_initial, lr_max, step_size)
    optimizer = SGDOptimizer(lr_schedule=lr_schedule)

    metrics = [AccuracyMetrics()]

    model.compile_model(optimizer, loss, metrics)
    history = model.fit(x_train, y_train, x_val, y_val, n_epochs, batch_size)

    plot_losses(history)
    plot_costs(history)
    plot_accuracies(history)
    plot_lr(history)

    scores_test = model.forward(x_test)
    y_hat_test = np.argmax(scores_test, axis=1)
    acc_test = AccuracyMetrics().get_metrics(y_test, y_hat_test)

    print(f"test acc: {acc_test}")


def part_4_coarse_to_fine_search():
    # Load data
    # train and val set are batch 1, 2, 3, 4, and 5, test set is test
    path = os.path.join("data", "data_batch_1")
    x_train_img_1, y_train_1 = load_cfar10_batch(path)

    path = os.path.join("data", "data_batch_2")
    x_train_img_2, y_train_2 = load_cfar10_batch(path)

    path = os.path.join("data", "data_batch_3")
    x_train_img_3, y_train_3 = load_cfar10_batch(path)

    path = os.path.join("data", "data_batch_4")
    x_train_img_4, y_train_4 = load_cfar10_batch(path)

    path = os.path.join("data", "data_batch_5")
    x_train_img_5, y_train_5 = load_cfar10_batch(path)

    x_train_val_img = np.vstack([x_train_img_1, x_train_img_2, x_train_img_3, x_train_img_4, x_train_img_5])
    y_train_val = np.hstack([y_train_1, y_train_2, y_train_3, y_train_4, y_train_5])

    x_train_img, x_val_img, y_train, y_val = train_test_split(x_train_val_img, y_train_val,
                                                              test_size=0.1, random_state=42)

    path = os.path.join("data", "test_batch")
    x_test_img, y_test = load_cfar10_batch(path)

    # check counts in datasets
    print(f"train set shape: {x_train_img.shape}, "
          f"val set shape: {x_val_img.shape}, test set shape: {x_test_img.shape}")
    print(f"train labels shape: {y_train.shape},"
          f" val labels shape: {y_val.shape}, test labels shape: {y_test.shape}")

    # assert balanced dataset
    train_counts = np.unique(y_train, return_counts=True)[1]
    train_ratios = train_counts / train_counts.sum()

    val_counts = np.unique(y_val, return_counts=True)[1]
    val_ratios = val_counts / val_counts.sum()

    test_counts = np.unique(y_test, return_counts=True)[1]
    test_ratios = test_counts / test_counts.sum()

    # np.testing.assert_array_equal(train_ratios, val_ratios)
    # np.testing.assert_array_equal(val_ratios, test_ratios)

    # np.testing.assert_allclose(train_ratios, val_ratios, rtol=1e-1, atol=0)
    # np.testing.assert_allclose(val_ratios, test_ratios, rtol=1e-1, atol=0)

    # Pre-process data
    x_train_un = x_train_img.reshape(x_train_img.shape[0], -1)
    x_val_un = x_val_img.reshape(x_val_img.shape[0], -1)
    x_test_un = x_test_img.reshape(x_test_img.shape[0], -1)

    x_train = x_train_un / 255.
    x_val = x_val_un / 255.
    x_test = x_test_un / 255.

    mean = np.mean(x_train, axis=0).reshape(1, x_train.shape[1])
    std = np.std(x_train, axis=0).reshape(1, x_train.shape[1])

    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std

    objective = AccuracyMetrics()
    build_model = build_model_func

    # coarse
    n = 10
    n_epochs = 8
    batch_size = 100

    n_s = int(2 * np.floor(x_train.shape[0] / batch_size))
    print(f"step size of cyc. lr: {n_s} update steps")

    cycle_steps = 2 * n_s
    print(f"full cycle of cyc.lr : {cycle_steps} update steps")

    # print(cycle * batch_size)

    epochs_one_full_cycle = (cycle_steps * batch_size) / x_train.shape[0]
    print(f"{epochs_one_full_cycle} epochs = 1 full cycle = {cycle_steps} update steps")

    n_cycle = 2
    print(f"{n_cycle} cycle = {n_cycle * epochs_one_full_cycle} epochs = {n_cycle * cycle_steps} update steps")

    params = {"reg_rate_l2": coarse_custom(n=n)}
    tuner = Tuner(build_model, objective, iterations=1, **params)
    best_objective, best_params = tuner.search(x_train, y_train, x_val, y_val, n_epochs, batch_size)

    print(f"best obj:{best_objective:.4f}, with {best_params}")

    # coarse to fine
    n = 10
    n_epochs = 8
    batch_size = 100

    params = {"reg_rate_l2": coarse_to_fine_custom(best_params["reg_rate_l2"], n=n)}
    tuner = Tuner(build_model, objective, iterations=1, **params)
    best_objective, best_params = tuner.search(x_train, y_train, x_val, y_val, n_epochs, batch_size)

    print(f"best obj:{best_objective:.4f}, with {best_params}")

    return best_params


def part_5_final_model(best_params):
    # Load data
    # train set is batch 1, val set is batch 2, test set is test
    path = os.path.join("data", "data_batch_1")
    x_train_img_1, y_train_1 = load_cfar10_batch(path)

    path = os.path.join("data", "data_batch_2")
    x_train_img_2, y_train_2 = load_cfar10_batch(path)

    path = os.path.join("data", "data_batch_3")
    x_train_img_3, y_train_3 = load_cfar10_batch(path)

    path = os.path.join("data", "data_batch_4")
    x_train_img_4, y_train_4 = load_cfar10_batch(path)

    path = os.path.join("data", "data_batch_5")
    x_train_img_5, y_train_5 = load_cfar10_batch(path)

    x_train_val_img = np.vstack([x_train_img_1, x_train_img_2, x_train_img_3, x_train_img_4, x_train_img_5])
    y_train_val = np.hstack([y_train_1, y_train_2, y_train_3, y_train_4, y_train_5])

    x_train_img, x_val_img, y_train, y_val = train_test_split(x_train_val_img, y_train_val,
                                                              test_size=0.02, random_state=42)

    path = os.path.join("data", "test_batch")
    x_test_img, y_test = load_cfar10_batch(path)

    # check counts in datasets
    print(f"train set shape: {x_train_img.shape}, "
          f"val set shape: {x_val_img.shape}, test set shape: {x_test_img.shape}")
    print(f"train labels shape: {y_train.shape},"
          f" val labels shape: {y_val.shape}, test labels shape: {y_test.shape}")

    # assert balanced dataset
    train_counts = np.unique(y_train, return_counts=True)[1]
    train_ratios = train_counts / train_counts.sum()

    val_counts = np.unique(y_val, return_counts=True)[1]
    val_ratios = val_counts / val_counts.sum()

    test_counts = np.unique(y_test, return_counts=True)[1]
    test_ratios = test_counts / test_counts.sum()

    # np.testing.assert_array_equal(train_ratios, val_ratios)
    # np.testing.assert_array_equal(val_ratios, test_ratios)

    # np.testing.assert_allclose(train_ratios, val_ratios, rtol=1e-1, atol=0)
    # np.testing.assert_allclose(val_ratios, test_ratios, rtol=1e-1, atol=0)

    # Pre-process data
    x_train_un = x_train_img.reshape(x_train_img.shape[0], -1)
    x_val_un = x_val_img.reshape(x_val_img.shape[0], -1)
    x_test_un = x_test_img.reshape(x_test_img.shape[0], -1)

    x_train = x_train_un / 255.
    x_val = x_val_un / 255.
    x_test = x_test_un / 255.

    mean = np.mean(x_train, axis=0).reshape(1, x_train.shape[1])
    std = np.std(x_train, axis=0).reshape(1, x_train.shape[1])

    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std

    coeff = 1.0
    mean = 0.0
    std = 0.01
    params = {"coeff": coeff, "mean": mean, "std": None}

    # reg_rate_l2 = 0.1
    # best obj:0.5134, with {'reg_rate_l2': 0.00036537637001811185}
    reg_rate_l2 = best_params["reg_rate_l2"]
    # print(reg_rate_l2)
    # raise

    in_dim = x_train.shape[1]
    out_dim = 10
    mid_dim = 50

    seed = 200

    dense_1 = \
        Dense(in_dim=in_dim, out_dim=mid_dim,
              kernel_initializer=XavierInitializer(seed=seed, **params),
              bias_initializer=XavierInitializer(seed=seed + 1, **params),
              kernel_regularizer=L2Regularizer(reg_rate=reg_rate_l2),
              activation=ReLUActivation()
              )

    dense_2 = \
        Dense(in_dim=mid_dim, out_dim=out_dim,
              kernel_initializer=XavierInitializer(seed=seed + 2, **params),
              bias_initializer=XavierInitializer(seed=seed + 3, **params),
              kernel_regularizer=L2Regularizer(reg_rate=reg_rate_l2),
              activation=SoftmaxActivation()
              )

    layers = [
        dense_1,
        dense_2
    ]

    model = Model(layers)
    print(model)

    loss = CategoricalCrossEntropyLoss()

    n_epochs = 12
    batch_size = 100

    n_s = int(2 * np.floor(x_train.shape[0] / batch_size))
    print(f"step size of cyc. lr: {n_s} update steps")

    cycle_steps = 2 * n_s
    print(f"full cycle of cyc.lr : {cycle_steps} update steps")

    # print(cycle * batch_size)

    epochs_one_full_cycle = (cycle_steps * batch_size) / x_train.shape[0]
    print(f"{epochs_one_full_cycle} epochs = 1 full cycle = {cycle_steps} update steps")

    n_cycle = 3
    print(f"{n_cycle} cycle = {n_cycle * epochs_one_full_cycle} epochs = {n_cycle * cycle_steps} update steps")

    # lr_initial = 0.01
    # lr_schedule = LRConstantSchedule(lr_initial)
    # decay_steps = n_epochs * 2
    # decay_rate = 0.9
    # lr_schedule = LRExponentialDecaySchedule(lr_initial, decay_steps, decay_rate)

    lr_initial = 1e-5
    lr_max = 1e-1
    step_size = 980
    lr_schedule = LRCyclingSchedule(lr_initial, lr_max, step_size)
    optimizer = SGDOptimizer(lr_schedule=lr_schedule)

    metrics = [AccuracyMetrics()]

    model.compile_model(optimizer, loss, metrics)
    history = model.fit(x_train, y_train, x_val, y_val, n_epochs, batch_size)

    plot_losses(history)
    plot_costs(history)
    plot_accuracies(history)
    plot_lr(history)

    scores_test = model.forward(x_test)
    y_hat_test = np.argmax(scores_test, axis=1)
    acc_test = AccuracyMetrics().get_metrics(y_test, y_hat_test)

    print(f"test acc: {acc_test}")


def main():
    # grad check
    part_1_grad_check()
    part_2_replicate_fig_3()
    part_3_replicate_fig_4()
    best_params = part_4_coarse_to_fine_search()
    part_5_final_model(best_params)


if __name__ == "__main__":
    main()
