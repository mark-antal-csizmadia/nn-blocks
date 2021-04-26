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
import json



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


class LinearActivation(Activation):
    """ Linear activation.
    Usually followed by CategoricalHingeLoss.
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
        forward propagates activation. Activation is linear.
    backward(g)
        Backpropagates incoming gradient into the layer, based on the linear activation.
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
        forward propagates activation. Activation is linear.

        Parameters
        ----------
        z : numpy.ndarray
            Linear transformation of layer.
            Shape is unknown here, but will usually be
            (batch size, this layer output dim = next layer input dim)

        Returns
        -------
        numpy.ndarray
            Linear activation.

        Notes
        -----
        None
        """
        return deepcopy(z)

    def backward(self, g):
        """ Backpropagates incoming gradient into the layer, based on the linear activation.

        Parameters
        ----------
        g : numpy.ndarray
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
        return deepcopy(g)

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
        repr_str = "linear"
        return repr_str


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
        z_stable = z - np.max(z, axis=1, keepdims=True)

        # get unnormalized probabilities
        # exp_scores.shape = (batch_size, K)
        #exp_z = np.exp(z_stable)

        # normalize them for each example
        # probs.shape = (batch_size, K)
        #a = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        Z = np.sum(np.exp(z_stable), axis=1, keepdims=True)
        log_probs = z_stable - np.log(Z)
        a = np.exp(log_probs)

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


def numerical_gradient_check_model(x, y, model, loss):
    in_dim = x.shape[1]
    params = {"mode": "train"}
    scores = model.forward(x, **params)
    layers_reg_loss = model.get_reg_loss()
    data_loss = loss.compute_loss(scores, y)
    cost = data_loss + layers_reg_loss
    model.backward(loss.grad(), **params)
    grads_analytic = model.get_gradients()

    def get_loss(model, x, y):
        params = {"mode": "train"}
        scores = model.forward(x, **params)
        layers_reg_loss = model.get_reg_loss()
        data_loss = loss.compute_loss(scores, y)
        cost = data_loss + layers_reg_loss
        return cost

    def set_param(model, layer_idx, param_name, param_val, x, y):
        trainable_params = model.get_trainable_params()
        trainable_params[layer_idx][param_name] = param_val
        model.set_trainable_params(trainable_params)

        return get_loss(model, x, y)

    trainable_params = model.get_trainable_params()
    for layer_idx in range(len(model.layers)):
        for param_name in trainable_params[layer_idx].keys():
            print(f"layer={layer_idx}, param_name={param_name}")
            f = lambda param_val: set_param(model, layer_idx, param_name, param_val, x, y)

            o_param_val = trainable_params[layer_idx][param_name]
            o_param_val_2 = deepcopy(o_param_val)
            grad_numerical = eval_numerical_gradient(f, o_param_val, verbose=False)

            grad_analytic = grads_analytic[layer_idx]["d" + param_name]

            rel_error = np.abs(grad_analytic - grad_numerical) \
                        / (np.maximum(np.abs(grad_analytic), np.abs(grad_numerical)) + 1e-9)
            np.testing.assert_array_almost_equal(grad_analytic, grad_numerical, decimal=4)
            print(f"max rel error={np.max(rel_error)}")
            trainable_params[layer_idx][param_name] = o_param_val_2

    print("test_grad_check passed")


def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
    """
    a naive implementation of numerical gradient of f at x
    - f should be a function that takes a single argument
    - x is the point (numpy array) to evaluate the gradient at

    From: https://cs231n.github.io/assignments2021/assignment2/
    """

    fx = f(x)  # evaluate function value at original point
    grad = np.zeros_like(x)
    # iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:

        # evaluate function at x+h
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h  # increment by h
        fxph = f(x)  # evalute f(x + h)
        x[ix] = oldval - h
        fxmh = f(x)  # evaluate f(x - h)
        x[ix] = oldval  # restore

        # compute the partial derivative with centered formula
        grad[ix] = (fxph - fxmh) / (2 * h)  # the slope
        if verbose:
            print(ix, grad[ix])
        it.iternext()  # step to next dimension

    return grad


def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """
    Evaluate a numeric gradient for a function that accepts a numpy
    array and returns a numpy array.

    From: https://cs231n.github.io/assignments2021/assignment2/
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad



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
        repr_str = "normal ~ " + f"{self.coeff:.6f} x N({self.mean:.6f}, {self.std:.6f}^2)"
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
        repr_str = "Xavier ~ " + f"{self.coeff:.6f} x N({self.mean:.6f}, {self.std:.6f}^2)"
        return repr_str



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
        Constructor.
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

        self.has_learnable_params = True

    def if_has_learnable_params(self, ):
        """ Returns if the layer has learnable params. Dense layer does have learnable params.

        Parameters
        ----------
        None

        Returns
        -------
        has_learnable_params
            True if the layer has learnable params.

        Notes
        -----
        None
        """
        return self.has_learnable_params

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

    def get_learnable_params(self):
        return {"w": self.get_w(), "b": self.get_b()}

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

    def set_learnable_params(self, **learnable_params):
        self.set_w(learnable_params["w"])
        self.set_b(learnable_params["b"])

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

    def get_learnable_params_grads(self):
        return {"dw": self.get_dw(), "db": self.get_db()}

    def get_reg_loss(self, ):
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

    def forward(self, x, **params):
        """ Forward-propagates signals through the layer and its activation.

        Parameters
        ----------
        x : numpy.ndarray
            Input data to layer of shape (batch_size, in_dim).
        params : dict
            Dict of params for forward pass such as train or test mode, seed, etc. Unused in Dense layer.

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

    def backward(self, g_in, **params):
        """ Back-propagates gradients through the the activation of the layer and then the layer.

        Parameters
        ----------
        g_in : numpy.ndarray
            Incoming (from later layers or losses) gradients, of shape (batch_size, out_dim).
        params : dict
            Dict of params for forward pass such as train or test mode, seed, etc. Unused in Dense layer.

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
                   + f"\t shape -- in: {self.in_dim}, out: {self.out_dim}\n" \
                   + "\t w -- init: " + self.kernel_initializer.__repr__() \
                   + ", reg: " + self.kernel_regularizer.__repr__() + "\n" \
                   + "\t b -- init: " + self.bias_initializer.__repr__() + "\n" \
                   + "\t activation: " + self.activation.__repr__() + "\n"
        return repr_str


class Dropout():
    """ Inv dropout - scaling at train time"""

    def __init__(self, p):
        self.p = p
        self.cache = {}
        self.has_learnable_params = False

    def if_has_learnable_params(self, ):
        return self.has_learnable_params

    def forward(self, x, **params):
        mode = params["mode"]
        seed = params["seed"]
        assert mode in ["train", "test"]

        if mode == "train":
            np.random.seed(seed)
            mask = (np.random.rand(*x.shape) < self.p) / self.p
            self.cache["mask"] = deepcopy(mask)
            # drop it boi!
            out = x * mask
        else:
            out = x

        return deepcopy(out)

    def backward(self, g_in, **params):
        mode = params["mode"]
        assert mode in ["train", "test"]

        if mode == "train":
            mask = deepcopy(self.cache["mask"])
            g_out = g_in * mask
        else:
            g_out = deepcopy(g_in)

        return g_out

    def __repr__(self, ):
        repr_str = f"dropout with p={self.p}"
        return repr_str


class BatchNormalization():
    def __init__(self, momentum, epsilon):
        self.momentum = momentum
        self.epsilon = epsilon
        # will be init at first computation
        self.beta = None
        self.gamma = None
        self.moving_mean = None
        self.moving_variance = None
        self.cache = {}
        self.grads = {}

        self.has_learnable_params = True

    def if_has_learnable_params(self, ):
        """ Returns if the layer has learnable params. Dense layer does have learnable params.

        Parameters
        ----------
        None

        Returns
        -------
        has_learnable_params
            True if the layer has learnable params.

        Notes
        -----
        None
        """
        return self.has_learnable_params

    def get_gamma(self, ):
        """ Returns the gamma parameters.

        Parameters
        ----------
        None

        Returns
        -------
        numpy.ndarray
            The gamma parameters.

        Notes
        -----
        None
        """
        return deepcopy(self.gamma)

    def get_beta(self, ):
        """ Returns the beta parameters.

        Parameters
        ----------
        None

        Returns
        -------
        numpy.ndarray
            The beta parameters.

        Notes
        -----
        None
        """
        return deepcopy(self.beta)

    def get_learnable_params(self):
        return {"gamma": self.get_gamma(), "beta": self.get_beta()}

    def set_gamma(self, gamma):
        """ Sets the gamma parameters.

        Parameters
        ----------
        gamma : numpy.ndarray
            The gamma parameters.

        Returns
        -------
        None

        Notes
        -----
        None
        """
        self.gamma = deepcopy(gamma)

    def set_beta(self, beta):
        """ Sets the beta parameters.

        Parameters
        ----------
        beta : numpy.ndarray
            The beta parameters.

        Returns
        -------
        None

        Notes
        -----
        None
        """
        self.beta = deepcopy(beta)

    def set_learnable_params(self, **learnable_params):
        self.set_gamma(learnable_params["gamma"])
        self.set_beta(learnable_params["beta"])

    def get_dgamma(self, ):
        """ Returns the gradients of gamma parameters.

        Parameters
        ----------
        None

        Returns
        -------
        ret : None or numpy.ndarray
            The gradients of gamma parameters, or None if does not exist yet.

        Notes
        -----
        None
        """
        if "dgamma" in self.grads.keys():
            dgamma = self.grads["dgamma"]
            ret = deepcopy(dgamma)
        else:
            ret = None

        return ret

    def get_dbeta(self, ):
        """ Returns the gradients of beta parameters.

        Parameters
        ----------
        None

        Returns
        -------
        ret : None or numpy.ndarray
            The gradients of beta parameters, or None if does not exist yet.

        Notes
        -----
        None
        """
        if "dbeta" in self.grads.keys():
            dbeta = self.grads["dbeta"]
            ret = deepcopy(dbeta)
        else:
            ret = None

        return ret

    def get_learnable_params_grads(self):
        return {"dgamma": self.get_dgamma(), "dbeta": self.get_dbeta()}

    def get_reg_loss(self, ):
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
        return 0.0

    def forward(self, x, **params):
        """
        x.shape = (batch_size, in_dim)
        """
        mode = params["mode"]
        assert mode in ["train", "test"]

        in_dim = x.shape[1]
        size = (in_dim,)

        if self.moving_mean is None:
            self.moving_mean = np.zeros(size)
        if self.moving_variance is None:
            self.moving_variance = np.zeros(size)
        if self.beta is None:
            # zeros based on Keras
            self.beta = np.zeros(size)
        if self.gamma is None:
            # ones based on Keras
            self.gamma = np.ones(size)

        if mode == "train":
            # batch mean and var and std, all of shape (in_dim, )?
            mean_batch = np.mean(x, axis=0)
            var_batch = np.var(x, axis=0)
            std_batch = np.sqrt(var_batch + self.epsilon)

            # z transform (normalize) batch
            z = (x - mean_batch) / std_batch
            # scale and shift batch with learnable params
            a = self.gamma * z + self.beta

            # moving averages of mean and variance
            self.moving_mean = self.momentum * self.moving_mean + (1 - self.momentum) * mean_batch
            self.moving_variance = self.momentum * self.moving_variance + (1 - self.momentum) * var_batch

            self.cache["x"] = deepcopy(x)
            self.cache["mean_batch"] = deepcopy(mean_batch)
            self.cache["std_batch"] = deepcopy(std_batch)
            self.cache["z"] = deepcopy(z)

        else:
            std_batch = np.sqrt(self.moving_variance + self.epsilon)
            z = (x - self.moving_mean) / std_batch
            a = self.gamma * z + self.beta

        return a

    def backward(self, g_in, **params):
        """
        g_in.shape = (batch_size, out_dim)
        in_dim = out_dim of batch norm
        """
        mode = params["mode"]
        assert mode == "train"

        x = deepcopy(self.cache["x"])
        mean_batch = deepcopy(self.cache["mean_batch"])
        std_batch = deepcopy(self.cache["std_batch"])
        z = deepcopy(self.cache["z"])

        # grads of learnable params of batch norm layer (only in train mode)
        # a = self.gamma*z  + self.beta
        # so dJ/dgamma = dJ/da * da/dgamma = g_in * da/dgamma
        # where da/dgamma = z
        # dgamma.shape = (in_dim,)
        dgamma = np.sum(g_in * z, axis=0)
        # so dJ/dbeta = dJ/da * da/dbeta = g_in * da/dbeta
        # where da/dbeta = 1
        # dbeta.shape = (in_dim,)
        dbeta = np.sum(g_in, axis=0)

        # cache grads
        self.grads["dgamma"] = deepcopy(dgamma)
        self.grads["dbeta"] = deepcopy(dbeta)

        # downstream grads
        # a = gamma*z  + beta
        # where z = (x-mean_batch) / std_batch

        # dJ/dz = dJ/da * da/dz = g_in * da/dz
        # where da/dz = gamma
        # dz.shape = (n_batch, in_dim)
        dz = g_in * self.gamma

        # call (x - mean_batch) -> var_1
        # so dJ/dvar_1 = dJ/dz * dz/dvar_1
        # where dz/dvar_1 = 1/std_batch
        # dvar_1.shape = (n_batch, in_dim)
        # only partial derivative
        dvar_1_1 = dz * 1 / std_batch

        # call 1/std_batch -> var_2
        # dJ/dvar_2 = dJ/dz * dz/dvar_2
        # where dz/dvar_2 = var_1 = z * std_batch
        # dvar_2.shape = (in_dim, )
        dvar_2 = np.sum(dz * z * std_batch, axis=0)

        # call std_batch -> var_3
        # dJ/var_3 = dJ/dvar_2 * dvar_2/var_3
        # where dvar_2/var_3 = -1/var_3**2
        # dvar_3.shape = (in_dim, )
        dvar_3 = -dvar_2 / std_batch ** 2

        # call var_batch + epsilon -> var_4
        # note var_3 -> std_batch = sqrt(var_batch + epsilon)
        # dJ/dvar_4 = dJ/dvar_3 * dvar_3/dvar_4
        # where dvar_3/dvar_4 = 0.5 * 1/var_3
        # dvar_4.shape = (in_dim, )
        dvar_4 = 0.5 * dvar_3 / std_batch

        # call (x-mean_batch)**2 -> var_5
        # note var_batch = 1/n * sum((x-mean_batch)**2, axis=0) -> var_4
        # dJ/dvar_5 = dJ/dvar_4 * dvar_4/dvar_5
        # where dvar_4/dvar_5 = 1/n_batch * 1
        # dvar_5.shape = (n_batch, in_dim)
        n_batch = x.shape[0]
        dvar_5 = 1 / n_batch * np.ones(x.shape) * dvar_4

        # called (x-mean_batch) = z * std_batch -> var_1 (above)
        # dJ/dvar_1 = dJ/dvar_1_1 + dJ/dvar_5 * dvar_5/dvar_1
        # where dvar_5/dvar_1 = 2*var_1
        # dvar_1.shape = (n_batch, in_dim)
        dvar_1_2 = 2 * z * std_batch * dvar_5

        # sum partial derivatives wrt var_1
        # dvar_1.shape = (n_batch, in_dim)
        dvar_1 = dvar_1_1 + dvar_1_2

        # dJ/dx = dJ/dvar_1 * dvar_1/dx + dJ/dmean_batch * dmean_batch/dx
        # where dvar_1/dx = 1
        # dx.shape = (n_batch, in_dim)
        dx_1 = deepcopy(dvar_1)
        # dJ/dmean_batch = dJ/dvar_1 * dvar_1/dmean_batch
        # where dvar_1/dmean_batch = -1
        # dmean_batch.shape = (in_dim, )
        dmean_batch = -np.sum(dvar_1, axis=0)
        # dJ/dx = dJ/dmean_batch * dmean_batch/dx
        # where dmean_batch/dx = 1/n_batch
        # dx.shape = (n_batch, in_dim)
        n_batch = x.shape[0]
        dx_2 = 1 / n_batch * np.ones(x.shape) * dmean_batch

        # finally, downstream gradient is
        # dx.shape = (n_batch, in_dim)
        dx = dx_1 + dx_2

        return dx

    def __repr__(self):
        repr_str = f"batch norm with momentum {self.momentum}"
        return repr_str



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

    def __init__(self, name):
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

    def __repr__(self):
        return self.name


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

    def __init__(self, ):
        """ Constructor.

        Parameters
        ----------
        None

        Notes
        -----
        None
        """
        name = "categorical hinge loss"
        super().__init__(name)

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

    def __init__(self, ):
        """ Constructor.

        Parameters
        ----------
        None


        Notes
        -----
        None
        """
        name = "categorical cross-entropy loss"
        super().__init__(name)

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

    def __init__(self, lr_initial, repr_str):
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
        self.repr_str = repr_str

    def __repr__(self):
        return self.repr_str


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
        repr_str = "constant lr schedule"
        super().__init__(lr_initial, repr_str)

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
        repr_str = "exp. decay lr schedule"
        super().__init__(lr_initial, repr_str)
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
        repr_str = "cycling lr schedule"
        super().__init__(lr_initial, repr_str)
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
    get_trainable_params()
        Returns all trainable parameters of all layers.
    set_trainable_params(trainable_params)
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

    def forward(self, x, **params):
        """ Forward propagates signal through the model.

        Parameters
        ----------
        x : numpy.ndarray
            Input data to model, shape is (batch_size, in_dim)
            where in_dim is the input dimension of the first layer of the model.
        params : dict
            Dict of params for forward pass such as train or test mode, seed, etc.

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
            scores_temp = layer.forward(scores, **params)
            scores = deepcopy(scores_temp)
            if layer.if_has_learnable_params():
                self.reg_loss += layer.get_reg_loss()

        return scores

    def backward(self, y, **params):
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
            g_temp = layer.backward(g, **params)
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
            if layer.if_has_learnable_params():
                #dw = layer.get_dw()
                #db = layer.get_db()
                learnable_params_grads = layer.get_learnable_params_grads()
            else:
                raise Exception("no grads yet")
            grads.append(learnable_params_grads)

        return deepcopy(grads)

    def get_trainable_params(self, ):
        """ Returns all trainable parameters of all layers.

        Parameters
        ----------
        None

        Returns
        -------
        trainable_params : list
            The list of dictionaries of the trainable parameters of all layers of the model.
            At idx is the dictionary of trainable parameters of layer idx in the self.layers list.
            A list has two keys - w and b.

        Notes
        -----
        Iterates over layers in ascending order in the self.layers list.
        """
        trainable_params = []
        for idx, layer in enumerate(self.layers):
            if layer.if_has_learnable_params():
                #w = layer.get_w()
                #b = layer.get_b()
                learnable_params = layer.get_learnable_params()
            else:
                raise Exception("no trainable params")
            trainable_params.append(learnable_params)

        return deepcopy(trainable_params)

    def set_trainable_params(self, trainable_params):
        """ Sets all trainable parameters of all layers.

        Parameters
        ----------
        trainable_params : list
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
            trainable_param_dict = deepcopy(trainable_params[idx])
            #w = trainable_weight_dict["w"]
            #b = trainable_weight_dict["b"]
            if layer.if_has_learnable_params():
                #layer.set_w(deepcopy(w))
                #layer.set_b(deepcopy(b))
                layer.set_learnable_params(**trainable_param_dict)
            else:
                pass

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

    def compute_metrics(self, y, scores, postfix=None):
        assert postfix in ["train", "val"] or postfix is None
        metrics_dict = {}

        for metrics in self.metrics:
            metrics_value = metrics.compute(y, scores)
            if postfix is not None:
                key = metrics.name + "_" + postfix
                self.metrics_dict[key].append(metrics_value)
            else:
                key = metrics.name
            metrics_dict[key] = metrics_value

        return metrics_dict

    def fit(self, x_train, y_train, x_val, y_val, n_epochs, batch_size, verbose):
        """ Fits the model to the data.

        Parameters
        ----------
        x_train : numpy.ndarray
            Training data to model of shape (batch_size, in_dim) where in_dim is
            the input dimension of the first layer of the Model.
        y_train : numpy.ndarray
            True labels of training data.
            Shape is (batch_size, )
        x_val : numpy.ndarray
            Validation data to model of shape (batch_size, in_dim) where in_dim is
            the input dimension of the first layer of the Model.
        y_val : numpy.ndarray
            True labels of validation data.
            Shape is (batch_size, )
        n_epochs : int
            The number of epochs to train for.
        batch_size : int
            The batch size of the mini-batch gradient descent algorithm.
            x_train.shape[0] has to be divisible by batch_size
        verbose : int
            The degree to which training progress is printed in the console.
            2: print all, 1: print some, 0: do not print

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
        assert isinstance(verbose, int) and verbose in [0, 1, 2], \
            f"verbose has to be an integer and in [0,1,2], but got {verbose} (type: {type(verbose)})"

        for n_epoch in range(n_epochs):
            if verbose in [1, 2]:
                print(f"starting epoch: {n_epoch + 1} ...")

            # Shuffle data
            indices = np.arange(x_train.shape[0])
            np.random.shuffle(indices)
            x_train = x_train[indices]
            y_train = y_train[indices]

            n_batch = int(x_train.shape[0] / batch_size)

            if verbose in [2]:
                batches = tqdm(range(n_batch), file=sys.stdout)
            else:
                batches = range(n_batch)

            params_train = {"mode": "train", "seed": None}

            for b in batches:
                if verbose in [2]:
                    batches.set_description(f"batch {b + 1}/{n_batch}")

                x_batch = x_train[b * batch_size:(b + 1) * batch_size]
                y_batch = y_train[b * batch_size:(b + 1) * batch_size]

                scores = self.forward(x_batch, **params_train)

                layers_reg_loss = self.get_reg_loss()
                data_loss = self.loss.compute_loss(scores, y_batch)
                cost = data_loss + layers_reg_loss

                self.backward(self.loss.grad(), **params_train)

                trainable_params = \
                    self.optimizer.apply_grads(trainable_params=self.get_trainable_params(),
                                               grads=self.get_gradients())

                self.set_trainable_params(trainable_params)

                # should I do it here? yes
                self.optimizer.apply_lr_schedule()

            self.lr_dict["lr"].append(self.optimizer.get_lr())

            params_test = {"mode": "test", "seed": None}
            scores_train = self.forward(x_train, **params_test)
            layers_reg_loss_train = self.get_reg_loss()
            data_loss_train = self.loss.compute_loss(scores_train, y_train)
            cost_train = data_loss_train + layers_reg_loss_train

            scores_val = self.forward(x_val, **params_test)
            layers_reg_loss_val = self.get_reg_loss()
            data_loss_val = self.loss.compute_loss(scores_val, y_val)
            cost_val = data_loss_val + layers_reg_loss_val

            self.loss_dict["loss_train"].append(data_loss_train)
            self.loss_dict["loss_val"].append(data_loss_val)
            self.cost_dict["cost_train"].append(cost_train)
            self.cost_dict["cost_val"].append(cost_val)
            train_str = f"train loss = {data_loss_train} / train cost = {cost_train}"
            val_str = f"val loss = {data_loss_val} / val cost = {cost_val}"

            metrics_dict_train = self.compute_metrics(y_train, scores_train, postfix="train")
            metrics_dict_val = self.compute_metrics(y_val, scores_val, postfix="val")
            train_str += "\n\t -- " + json.dumps(metrics_dict_train)
            val_str += "\n\t -- " + json.dumps(metrics_dict_val)

            if verbose in [1, 2]:
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
        assert self.compiled
        repr_str = "model summary: \n"
        for idx, layer in enumerate(self.layers):
            repr_str = repr_str + f"layer {idx}: " + layer.__repr__() + "\n"
        repr_str += self.loss.__repr__() + "\n"
        repr_str += self.optimizer.__repr__() + "\n"

        return repr_str



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
            A list has two keys - w and b.

        grads : list
            The list of dictionaries of gradients of all parameters of all layers of a model.
            At idx is the dictionary of gradients of layer idx in the Model.layers list.
            A list has two keys - dw and db.

        Returns
        -------
        updated_trainable_params : list
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
        updated_trainable_params = deepcopy(trainable_params)

        assert len(trainable_params) == len(grads)

        for idx in range(len(trainable_params)):
            param_dict = deepcopy(trainable_params[idx])
            grad_dict = deepcopy(grads[idx])

            for p, g in zip(param_dict, grad_dict):
                updated_trainable_params[idx][p] = param_dict[p] - self.lr * grad_dict[g]

        return deepcopy(updated_trainable_params)



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
        repr_str = f"l2 with {self.reg_rate:.4e}"
        return repr_str


def build_model_2_layer_with_loss_cross_entropy(reg_rate, in_dim, seed):
    params = {"coeff": 1.0, "mean": 0.0, "std": None}
    out_dim = 10
    mid_dim = 50

    if reg_rate != 0.0:
        kernel_regularizer = L2Regularizer(reg_rate)
    else:
        kernel_regularizer = None

    dense_1 = \
        Dense(in_dim=in_dim, out_dim=mid_dim,
              kernel_initializer=XavierInitializer(seed=seed, **params),
              bias_initializer=XavierInitializer(seed=seed + 1, **params),
              kernel_regularizer=kernel_regularizer,
              activation=ReLUActivation()
              )

    dense_2 = \
        Dense(in_dim=mid_dim, out_dim=out_dim,
              kernel_initializer=XavierInitializer(seed=seed + 2, **params),
              bias_initializer=XavierInitializer(seed=seed + 3, **params),
              kernel_regularizer=kernel_regularizer,
              activation=SoftmaxActivation()
              )

    layers = [
        dense_1,
        dense_2
    ]

    model = Model(layers)
    loss = CategoricalCrossEntropyLoss()

    return model, loss


def build_model_3_layer_with_loss_cross_entropy(reg_rate, in_dim, seed):
    params = {"coeff": 1.0, "mean": 0.0, "std": None}
    mid_dim_1 = 50
    mid_dim_2 = 50
    out_dim = 10

    if reg_rate != 0.0:
        kernel_regularizer = L2Regularizer(reg_rate)
    else:
        kernel_regularizer = None

    dense_1 = \
        Dense(in_dim=in_dim, out_dim=mid_dim_1,
              kernel_initializer=XavierInitializer(seed=seed, **params),
              bias_initializer=XavierInitializer(seed=seed + 1, **params),
              kernel_regularizer=kernel_regularizer,
              activation=ReLUActivation()
              )

    dense_2 = \
        Dense(in_dim=mid_dim_1, out_dim=mid_dim_2,
              kernel_initializer=XavierInitializer(seed=seed + 2, **params),
              bias_initializer=XavierInitializer(seed=seed + 3, **params),
              kernel_regularizer=kernel_regularizer,
              activation=ReLUActivation()
              )

    dense_3 = \
        Dense(in_dim=mid_dim_2, out_dim=out_dim,
              kernel_initializer=XavierInitializer(seed=seed + 4, **params),
              bias_initializer=XavierInitializer(seed=seed + 5, **params),
              kernel_regularizer=kernel_regularizer,
              activation=SoftmaxActivation()
              )

    layers = [
        dense_1,
        dense_2,
        dense_3
    ]

    model = Model(layers)
    loss = CategoricalCrossEntropyLoss()

    return model, loss


def build_model_2_layer_with_bn_with_loss_cross_entropy(reg_rate, in_dim, seed):
    params = {"coeff": 1.0, "mean": 0.0, "std": None}
    out_dim = 10
    mid_dim = 50

    if reg_rate != 0.0:
        kernel_regularizer = L2Regularizer(reg_rate)
    else:
        kernel_regularizer = None

    dense_1 = \
        Dense(in_dim=in_dim, out_dim=mid_dim,
              kernel_initializer=XavierInitializer(seed=seed, **params),
              bias_initializer=XavierInitializer(seed=seed + 1, **params),
              kernel_regularizer=kernel_regularizer,
              activation=ReLUActivation()
              )

    bn_1 = BatchNormalization(momentum=0.9, epsilon=1e-5)

    dense_2 = \
        Dense(in_dim=mid_dim, out_dim=out_dim,
              kernel_initializer=XavierInitializer(seed=seed + 2, **params),
              bias_initializer=XavierInitializer(seed=seed + 3, **params),
              kernel_regularizer=kernel_regularizer,
              activation=SoftmaxActivation()
              )

    layers = [
        dense_1,
        bn_1,
        dense_2
    ]

    model = Model(layers)
    loss = CategoricalCrossEntropyLoss()

    return model, loss


def build_model_3_layer_with_bn_with_loss_cross_entropy(reg_rate, in_dim, seed):
    params = {"coeff": 1.0, "mean": 0.0, "std": None}
    mid_dim_1 = 50
    mid_dim_2 = 50
    out_dim = 10

    if reg_rate != 0.0:
        kernel_regularizer = L2Regularizer(reg_rate)
    else:
        kernel_regularizer = None

    dense_1 = \
        Dense(in_dim=in_dim, out_dim=mid_dim_1,
              kernel_initializer=XavierInitializer(seed=seed, **params),
              bias_initializer=XavierInitializer(seed=seed + 1, **params),
              kernel_regularizer=kernel_regularizer,
              activation=ReLUActivation()
              )

    bn_1 = BatchNormalization(momentum=0.9, epsilon=1e-5)

    dense_2 = \
        Dense(in_dim=mid_dim_1, out_dim=mid_dim_2,
              kernel_initializer=XavierInitializer(seed=seed + 2, **params),
              bias_initializer=XavierInitializer(seed=seed + 3, **params),
              kernel_regularizer=kernel_regularizer,
              activation=ReLUActivation()
              )

    bn_2 = BatchNormalization(momentum=0.9, epsilon=1e-5)

    dense_3 = \
        Dense(in_dim=mid_dim_2, out_dim=out_dim,
              kernel_initializer=XavierInitializer(seed=seed + 4, **params),
              bias_initializer=XavierInitializer(seed=seed + 5, **params),
              kernel_regularizer=kernel_regularizer,
              activation=SoftmaxActivation()
              )

    layers = [
        dense_1,
        bn_1,
        dense_2,
        bn_2,
        dense_3
    ]

    model = Model(layers)
    loss = CategoricalCrossEntropyLoss()

    return model, loss


def test_model(x, y, seed=6):
    size = (2, 20)
    c = 10
    np.random.seed(seed + 1)
    x = np.random.normal(loc=0, scale=1, size=size)
    np.random.seed(seed + 2)
    y = np.random.randint(c, size=size[0])

    in_dim = x.shape[1]

    build_model_loss_func_list = [
        build_model_2_layer_with_loss_cross_entropy,
        build_model_3_layer_with_loss_cross_entropy,
        build_model_2_layer_with_bn_with_loss_cross_entropy,
        build_model_3_layer_with_bn_with_loss_cross_entropy
    ]

    np.random.seed(seed + 3)
    reg_rates = 10e-1 * np.random.randint(low=1, high=10, size=5)

    for reg_rate in reg_rates:
        for build_model_func in build_model_loss_func_list:
            model, loss = build_model_func(reg_rate, in_dim, seed)
            numerical_gradient_check_model(x, y, model, loss)


def plot_losses(history, filename):
    plt.plot(history["loss_train"], label="train")
    plt.plot(history["loss_val"], label="val")
    plt.grid()
    plt.title("Loss vs. epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    path = f"{filename}_losses.png"
    plt.savefig(path)
    plt.show()


def plot_costs(history, filename):
    plt.plot(history["cost_train"], label="train")
    plt.plot(history["cost_val"], label="val")
    plt.grid()
    plt.title("Cost vs. epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Cost")
    plt.legend()
    path = f"{filename}_costs.png"
    plt.savefig(path)
    plt.show()


def plot_accuracies(history, filename):
    plt.plot(history["accuracy_train"], label="train")
    plt.plot(history["accuracy_val"], label="val")
    plt.grid()
    plt.title("Accuracy vs. epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    path = f"{filename}_accuracies.png"
    plt.savefig(path)
    plt.show()


def plot_lr(history, filename):
    plt.plot(history["lr"], label="lr")
    plt.grid()
    plt.title("Learning rate vs. epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Learning rate")
    plt.legend()
    path = f"{filename}_lrs.png"
    plt.savefig(path)
    plt.show()


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
                print(f"searching: {params}")
                # build_model with tuple params
                model = self.build_model(seed=200, **params)
                # fit model
                history = model.fit(x_train, y_train, x_val, y_val, n_epochs, batch_size, verbose=2)
                # meaasure objective on model
                params_test = {"mode": "test"}
                scores_val = model.forward(x_val, **params_test)
                objective_val = self.objective.compute(y_val, scores_val)
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
        date_string = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S:%f")
        path = os.path.join(date_string + ".csv")

        df.to_csv(path, encoding='utf-8', index=False)

        # argmax across rows and return best params as dict (~**params)
        best_params = dict(df.loc[df[self.objective.name].idxmax()])
        best_objective = best_params.pop(self.objective.name)

        return best_objective, best_params


def network_bn_sigma(dims, if_bn, sigma):
    # filename = f"9_layer_with_bn_normal_init_sigma_{sigma}"
    seed = 10

    params = {"coeff": 1.0, "mean": 0.0, "std": sigma}

    reg_rate = 0.005

    layers = []

    for n_layer in range(1, len(dims)):
        # print(n_layer)
        # print(dims[n_layer])
        if n_layer == len(dims) - 1:
            dense = \
                Dense(in_dim=dims[n_layer - 1], out_dim=dims[n_layer],
                      kernel_initializer=NormalInitializer(seed=seed + n_layer, **params),
                      bias_initializer=NormalInitializer(seed=seed + n_layer, **params),
                      kernel_regularizer=L2Regularizer(reg_rate),
                      activation=SoftmaxActivation()
                      )
        else:
            dense = \
                Dense(in_dim=dims[n_layer - 1], out_dim=dims[n_layer],
                      kernel_initializer=NormalInitializer(seed=seed + n_layer, **params),
                      bias_initializer=NormalInitializer(seed=seed + n_layer, **params),
                      kernel_regularizer=L2Regularizer(reg_rate),
                      activation=ReLUActivation()
                      )
            if if_bn:
                bn = BatchNormalization(momentum=0.9, epsilon=1e-5)

        layers.append(dense)

        if n_layer == len(dims) - 1:
            pass
        else:
            if if_bn:
                layers.append(bn)

    model = Model(layers)
    loss = CategoricalCrossEntropyLoss()

    lr_initial = 1e-5
    lr_max = 1e-1
    step_size = 2250
    lr_schedule = LRCyclingSchedule(lr_initial, lr_max, step_size)

    optimizer = SGDOptimizer(lr_schedule=lr_schedule)

    metrics = [AccuracyMetrics()]

    model.compile_model(optimizer, loss, metrics)
    print(f"len(layers) = {len(layers)}")
    print(model)

    return model


def load_data_for_assignment():
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

    return x_train, y_train, x_val, y_val, x_test, y_test


def part_1_grad_check(x_train, y_train):
    test_model(x_train[:2, :10], y_train[:2])


def part_2_3_layer_no_bn(x_train, y_train, x_val, y_val, x_test, y_test):
    seed = 10

    params = {"coeff": 1.0, "mean": 0.0, "std": None}

    in_dim = x_train.shape[1]
    mid_dim_1 = 50
    mid_dim_2 = 50
    out_dim = 10

    reg_rate = 0.005

    dense_1 = \
        Dense(in_dim=in_dim, out_dim=mid_dim_1,
              kernel_initializer=XavierInitializer(seed=seed, **params),
              bias_initializer=XavierInitializer(seed=seed + 1, **params),
              kernel_regularizer=L2Regularizer(reg_rate),
              activation=ReLUActivation()
              )

    dense_2 = \
        Dense(in_dim=mid_dim_1, out_dim=mid_dim_2,
              kernel_initializer=XavierInitializer(seed=seed + 2, **params),
              bias_initializer=XavierInitializer(seed=seed + 3, **params),
              kernel_regularizer=L2Regularizer(reg_rate),
              activation=ReLUActivation()
              )

    dense_3 = \
        Dense(in_dim=mid_dim_2, out_dim=out_dim,
              kernel_initializer=XavierInitializer(seed=seed + 4, **params),
              bias_initializer=XavierInitializer(seed=seed + 5, **params),
              kernel_regularizer=L2Regularizer(reg_rate),
              activation=SoftmaxActivation()
              )

    layers = [
        dense_1,
        dense_2,
        dense_3
    ]

    model = Model(layers)
    loss = CategoricalCrossEntropyLoss()

    lr_initial = 1e-5
    lr_max = 1e-1
    step_size = 2250
    lr_schedule = LRCyclingSchedule(lr_initial, lr_max, step_size)

    optimizer = SGDOptimizer(lr_schedule=lr_schedule)

    metrics = [AccuracyMetrics()]

    model.compile_model(optimizer, loss, metrics)
    print(model)

    batch_size = 100
    n_epochs = 20
    verbose = 2
    history = model.fit(x_train, y_train, x_val, y_val, n_epochs, batch_size, verbose)

    params_test = {"mode": "test"}
    scores_test = model.forward(x_test, **params_test)
    y_hat_test = np.argmax(scores_test, axis=1)
    metrics_test = model.compute_metrics(y_test, scores_test)

    filename = "3_layer_no_bn_xavier_init"
    plot_losses(history, filename)
    plot_costs(history, filename)
    plot_accuracies(history, filename)
    plot_lr(history, filename)

    print(f"test acc: {metrics_test}")


def part_3_3_layer_with_bn(x_train, y_train, x_val, y_val, x_test, y_test):
    seed = 10

    params = {"coeff": 1.0, "mean": 0.0, "std": None}

    in_dim = x_train.shape[1]
    mid_dim_1 = 50
    mid_dim_2 = 50
    out_dim = 10

    reg_rate = 0.005

    dense_1 = \
        Dense(in_dim=in_dim, out_dim=mid_dim_1,
              kernel_initializer=XavierInitializer(seed=seed, **params),
              bias_initializer=XavierInitializer(seed=seed + 1, **params),
              kernel_regularizer=L2Regularizer(reg_rate),
              activation=ReLUActivation()
              )
    bn_1 = BatchNormalization(momentum=0.9, epsilon=1e-5)

    dense_2 = \
        Dense(in_dim=mid_dim_1, out_dim=mid_dim_2,
              kernel_initializer=XavierInitializer(seed=seed + 2, **params),
              bias_initializer=XavierInitializer(seed=seed + 3, **params),
              kernel_regularizer=L2Regularizer(reg_rate),
              activation=ReLUActivation()
              )

    bn_2 = BatchNormalization(momentum=0.9, epsilon=1e-5)

    dense_3 = \
        Dense(in_dim=mid_dim_2, out_dim=out_dim,
              kernel_initializer=XavierInitializer(seed=seed + 4, **params),
              bias_initializer=XavierInitializer(seed=seed + 5, **params),
              kernel_regularizer=L2Regularizer(reg_rate),
              activation=SoftmaxActivation()
              )

    layers = [
        dense_1,
        bn_1,
        dense_2,
        bn_2,
        dense_3
    ]

    model = Model(layers)
    loss = CategoricalCrossEntropyLoss()

    lr_initial = 1e-5
    lr_max = 1e-1
    step_size = 2250
    lr_schedule = LRCyclingSchedule(lr_initial, lr_max, step_size)

    optimizer = SGDOptimizer(lr_schedule=lr_schedule)

    metrics = [AccuracyMetrics()]

    model.compile_model(optimizer, loss, metrics)
    print(model)

    batch_size = 100
    n_epochs = 20
    verbose = 2
    history = model.fit(x_train, y_train, x_val, y_val, n_epochs, batch_size, verbose)

    params_test = {"mode": "test"}
    scores_test = model.forward(x_test, **params_test)
    y_hat_test = np.argmax(scores_test, axis=1)
    metrics_test = model.compute_metrics(y_test, scores_test)

    filename = "3_layer_with_bn_xavier_init"
    plot_losses(history, filename)
    plot_costs(history, filename)
    plot_accuracies(history, filename)
    plot_lr(history, filename)

    print(f"test acc: {metrics_test}")


def part_4_9_layer_no_bn(x_train, y_train, x_val, y_val, x_test, y_test):
    seed = 10

    params = {"coeff": 1.0, "mean": 0.0, "std": None}

    in_dim = x_train.shape[1]
    out_dim = 10

    reg_rate = 0.005

    dims = [50, 30, 20, 20, 10, 10, 10, 10]

    dims = list(reversed(list(reversed(dims)) + [in_dim])) + [out_dim]

    layers = []

    for n_layer in range(1, len(dims)):
        # print(n_layer)
        # print(dims[n_layer])
        if n_layer == len(dims) - 1:
            dense = \
                Dense(in_dim=dims[n_layer - 1], out_dim=dims[n_layer],
                      kernel_initializer=XavierInitializer(seed=seed + n_layer, **params),
                      bias_initializer=XavierInitializer(seed=seed + n_layer, **params),
                      kernel_regularizer=L2Regularizer(reg_rate),
                      activation=SoftmaxActivation()
                      )
        else:
            dense = \
                Dense(in_dim=dims[n_layer - 1], out_dim=dims[n_layer],
                      kernel_initializer=XavierInitializer(seed=seed + n_layer, **params),
                      bias_initializer=XavierInitializer(seed=seed + n_layer, **params),
                      kernel_regularizer=L2Regularizer(reg_rate),
                      activation=ReLUActivation()
                      )

        layers.append(dense)

    model = Model(layers)
    loss = CategoricalCrossEntropyLoss()

    lr_initial = 1e-5
    lr_max = 1e-1
    step_size = 2250
    lr_schedule = LRCyclingSchedule(lr_initial, lr_max, step_size)

    optimizer = SGDOptimizer(lr_schedule=lr_schedule)

    metrics = [AccuracyMetrics()]

    model.compile_model(optimizer, loss, metrics)
    print(model)

    batch_size = 100
    n_epochs = 20
    verbose = 2
    history = model.fit(x_train, y_train, x_val, y_val, n_epochs, batch_size, verbose)

    params_test = {"mode": "test"}
    scores_test = model.forward(x_test, **params_test)
    y_hat_test = np.argmax(scores_test, axis=1)
    metrics_test = model.compute_metrics(y_test, scores_test)

    filename = "9_layer_no_bn_xavier_init"
    plot_losses(history, filename)
    plot_costs(history, filename)
    plot_accuracies(history, filename)
    plot_lr(history, filename)

    print(f"test acc: {metrics_test}")


def part_4_9_layer_with_bn(x_train, y_train, x_val, y_val, x_test, y_test):
    seed = 12

    params = {"coeff": 1.0, "mean": 0.0, "std": None}

    in_dim = x_train.shape[1]
    out_dim = 10

    reg_rate = 0.005

    dims = [50, 30, 20, 20, 10, 10, 10, 10]

    dims = list(reversed(list(reversed(dims)) + [in_dim])) + [out_dim]

    layers = []

    for n_layer in range(1, len(dims)):
        # print(n_layer)
        # print(dims[n_layer])
        if n_layer == len(dims) - 1:
            dense = \
                Dense(in_dim=dims[n_layer - 1], out_dim=dims[n_layer],
                      kernel_initializer=XavierInitializer(seed=seed + n_layer, **params),
                      bias_initializer=XavierInitializer(seed=seed + n_layer, **params),
                      kernel_regularizer=L2Regularizer(reg_rate),
                      activation=SoftmaxActivation()
                      )
        else:
            dense = \
                Dense(in_dim=dims[n_layer - 1], out_dim=dims[n_layer],
                      kernel_initializer=XavierInitializer(seed=seed + n_layer, **params),
                      bias_initializer=XavierInitializer(seed=seed + n_layer, **params),
                      kernel_regularizer=L2Regularizer(reg_rate),
                      activation=ReLUActivation()
                      )
            bn = BatchNormalization(momentum=0.9, epsilon=1e-5)

        layers.append(dense)

        if n_layer == len(dims) - 1:
            pass
        else:
            layers.append(bn)

    model = Model(layers)
    loss = CategoricalCrossEntropyLoss()

    lr_initial = 1e-5
    lr_max = 1e-1
    step_size = 2250
    lr_schedule = LRCyclingSchedule(lr_initial, lr_max, step_size)

    optimizer = SGDOptimizer(lr_schedule=lr_schedule)

    metrics = [AccuracyMetrics()]

    model.compile_model(optimizer, loss, metrics)
    print(f"len(layers) = {len(layers)}")
    print(model)

    batch_size = 100
    n_epochs = 20
    verbose = 2
    history = model.fit(x_train, y_train, x_val, y_val, n_epochs, batch_size, verbose)

    params_test = {"mode": "test"}
    scores_test = model.forward(x_test, **params_test)
    y_hat_test = np.argmax(scores_test, axis=1)
    metrics_test = model.compute_metrics(y_test, scores_test)

    filename = "9_layer_with_bn_xavier_init"
    plot_losses(history, filename)
    plot_costs(history, filename)
    plot_accuracies(history, filename)
    plot_lr(history, filename)

    print(f"test acc: {metrics_test}")


def part_5_coarse_to_fine_search(x_train, y_train, x_val, y_val, x_test, y_test):
    def build_model_func(seed=200, **params):
        assert "reg_rate_l2" in params.keys()
        reg_rate_l2 = params["reg_rate_l2"]

        params = {"coeff": 1.0, "mean": 0.0, "std": None}

        in_dim = x_train.shape[1]
        mid_dim_1 = 50
        mid_dim_2 = 50
        out_dim = 10

        # reg_rate = 0.005

        dense_1 = \
            Dense(in_dim=in_dim, out_dim=mid_dim_1,
                  kernel_initializer=XavierInitializer(seed=seed, **params),
                  bias_initializer=XavierInitializer(seed=seed + 1, **params),
                  kernel_regularizer=L2Regularizer(reg_rate_l2),
                  activation=ReLUActivation()
                  )
        bn_1 = BatchNormalization(momentum=0.9, epsilon=1e-5)

        dense_2 = \
            Dense(in_dim=mid_dim_1, out_dim=mid_dim_2,
                  kernel_initializer=XavierInitializer(seed=seed + 2, **params),
                  bias_initializer=XavierInitializer(seed=seed + 3, **params),
                  kernel_regularizer=L2Regularizer(reg_rate_l2),
                  activation=ReLUActivation()
                  )

        bn_2 = BatchNormalization(momentum=0.9, epsilon=1e-5)

        dense_3 = \
            Dense(in_dim=mid_dim_2, out_dim=out_dim,
                  kernel_initializer=XavierInitializer(seed=seed + 4, **params),
                  bias_initializer=XavierInitializer(seed=seed + 5, **params),
                  kernel_regularizer=L2Regularizer(reg_rate_l2),
                  activation=SoftmaxActivation()
                  )

        layers = [
            dense_1,
            bn_1,
            dense_2,
            bn_2,
            dense_3
        ]

        model = Model(layers)
        loss = CategoricalCrossEntropyLoss()

        lr_initial = 1e-5
        lr_max = 1e-1
        step_size = 2250
        lr_schedule = LRCyclingSchedule(lr_initial, lr_max, step_size)

        optimizer = SGDOptimizer(lr_schedule=lr_schedule)

        metrics = [AccuracyMetrics()]

        model.compile_model(optimizer, loss, metrics)
        print(model)

        return model

    def coarse_custom(n, seed):
        l_min = -5
        l_max = -1
        np.random.seed(seed)

        return [10 ** (l_min + (l_max - l_min) * np.random.uniform(low=0, high=1)) for i in range(n)]

    def coarse_to_fine_custom(best_via_coarse, n, seed):
        half_interval = 0.2
        low = best_via_coarse * (1 - half_interval)
        high = best_via_coarse * (1 + half_interval)
        np.random.seed(seed)

        return [np.random.uniform(low=low, high=high) for i in range(n)]

    def coarse_to_fine_loop(best_params, best_objective, coarse_to_fine_custom, build_model, objective,
                            x_train, y_train, x_val, y_val, n_epochs, batch_size, n, n_levels, seed):
        winner_params = deepcopy(best_params)
        winner_objective = deepcopy(best_objective)

        for n_level in range(n_levels):
            print("-" * 10)
            print(f"level {n_level + 1}/{n_levels}")
            # n = 10
            # batch_size = 100
            # n_epochs = 20

            params = {"reg_rate_l2": coarse_to_fine_custom(best_params["reg_rate_l2"], n=n, seed=seed)}
            tuner = Tuner(build_model, objective, iterations=1, **params)
            best_objective, best_params = tuner.search(x_train, y_train, x_val, y_val, n_epochs, batch_size)

            if winner_objective < best_objective:
                winner_params = deepcopy(best_params)
                winner_objective = deepcopy(best_objective)

            print("-" * 10)
            print(f"level {n_level + 1}/{n_levels}")
            print(f"best obj:{winner_objective:.4f}, with {winner_params}")
            print("-" * 10)

        return winner_objective, winner_params

    objective = AccuracyMetrics()
    build_model = build_model_func

    # coarse
    n = 5
    batch_size = 100
    n_epochs = 20
    seed = 12

    params = {"reg_rate_l2": coarse_custom(10, seed)}
    tuner = Tuner(build_model, objective, iterations=1, **params)
    best_objective, best_params = tuner.search(x_train, y_train, x_val, y_val, n_epochs, batch_size)

    print(f"best obj:{best_objective:.4f}, with {best_params}")

    # coarse to fine
    n_levels = 2
    best_objective, best_params = \
        coarse_to_fine_loop(best_params, best_objective, coarse_to_fine_custom, build_model, objective,
                            x_train, y_train, x_val, y_val, n_epochs, batch_size, n, n_levels, seed)

    print(f"best obj:{best_objective:.4f}, with {best_params}")

    # train with best lambda
    model = build_model_func(seed=200, **best_params)
    batch_size = 100
    n_epochs = 30
    verbose = 2
    history = model.fit(x_train, y_train, x_val, y_val, n_epochs, batch_size, verbose)

    params_test = {"mode": "test"}
    scores_test = model.forward(x_test, **params_test)
    y_hat_test = np.argmax(scores_test, axis=1)
    metrics_test = model.compute_metrics(y_test, scores_test)

    filename = "3_layer_with_bn_xavier_init_best_l2"
    plot_losses(history, filename)
    plot_costs(history, filename)
    plot_accuracies(history, filename)
    plot_lr(history, filename)

    print(f"test acc: {metrics_test}")


def part_6_sensitivity_to_init(x_train, y_train, x_val, y_val, x_test, y_test):
    in_dim = x_train.shape[1]
    out_dim = 10

    dims_3 = [50, 50]
    dims_3 = list(reversed(list(reversed(dims_3)) + [in_dim])) + [out_dim]

    dims_9 = [50, 30, 20, 20, 10, 10, 10, 10]
    dims_9 = list(reversed(list(reversed(dims_9)) + [in_dim])) + [out_dim]

    dims_all = [dims_3, dims_9]

    batch_size = 100
    n_epochs = 20

    sigmas = [1e-1, 1e-3, 1e-4]

    results_dict = {"n_layers": [], "if_bn": [], "sigma": [], "test_accuracy": []}

    if_bns = [False, True]

    for dims in dims_all:

        for if_bn in if_bns:

            for sigma in sigmas:
                model = network_bn_sigma(dims, if_bn, sigma)

                history = model.fit(x_train, y_train, x_val, y_val, n_epochs, batch_size)

                params_test = {"mode": "test"}
                scores_test = model.forward(x_test, **params_test)
                y_hat_test = np.argmax(scores_test, axis=1)
                metrics_test = model.compute_metrics(y_test, scores_test)

                filename = f"{len(dims) - 1}_layer_{'with' if if_bn else 'no'}_bn_normal_init_sigma_{sigma}"
                plot_losses(history, filename)
                plot_costs(history, filename)
                plot_accuracies(history, filename)
                plot_lr(history, filename)

                print(f"test acc: {metrics_test}")

                results_dict["n_layers"].append(len(dims) - 1)
                results_dict["if_bn"].append(if_bn)
                results_dict["sigma"].append(sigma)
                results_dict["test_accuracy"].append(metrics_test["accuracy"])

    # df from list of dicts of params and objective val
    results_df = pd.DataFrame(data=results_dict)

    # save to csv
    path = os.path.join("sensitivity_to_init.csv")
    results_df.to_csv(path, encoding='utf-8', index=False)
    print(results_df)


def main():
    x_train, y_train, x_val, y_val, x_test, y_test = load_data_for_assignment()
    part_1_grad_check(x_train, y_train)
    part_2_3_layer_no_bn(x_train, y_train, x_val, y_val, x_test, y_test)
    part_3_3_layer_with_bn(x_train, y_train, x_val, y_val, x_test, y_test)
    part_4_9_layer_no_bn(x_train, y_train, x_val, y_val, x_test, y_test)
    part_4_9_layer_with_bn(x_train, y_train, x_val, y_val, x_test, y_test)
    part_5_coarse_to_fine_search(x_train, y_train, x_val, y_val, x_test, y_test)
    part_6_sensitivity_to_init(x_train, y_train, x_val, y_val, x_test, y_test)


if __name__ == "__main__":
    main()
