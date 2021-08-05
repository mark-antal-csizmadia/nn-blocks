import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pickle
from copy import deepcopy
from math import sqrt, ceil
import datetime
import sys
from itertools import product
import pandas as pd
import json
from random import shuffle
import emoji


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


class TanhActivation(Activation):
    """ Tanh activation.
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
        forward propagates activation. Activation is tanh.
    backward(g)
        Backpropagates incoming gradient into the layer, based on the tanh activation.
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
        forward propagates activation. Activation is tanh.

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
        a = np.tanh(z)
        self.cache["a"] = deepcopy(a)
        return a

    def backward(self, g_in):
        """ Backpropagates incoming gradient into the layer, based on the tanh activation.

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
        a = deepcopy(self.cache["a"])
        g_out = (1 - np.power(a, 2)) * g_in
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
        repr_str = "tanh"
        return repr_str



class CharByCharSynhthetizer():
    """ Synthetize text (char-by-char) from a trained RNN using a one-hot encoder."""

    def __init__(self, rnn, char_init, encode_lambda, onehot_encoder, decode_lambda, ts, n_step, path_out):
        self.rnn = rnn
        self.char_init = char_init
        self.encode_lambda = encode_lambda
        self.onehot_encoder = onehot_encoder
        self.decode_lambda = decode_lambda
        self.ts = ts
        self.n_step = n_step
        self.path_out = path_out

        try:
            os.remove(path_out)
        except OSError:
            pass

        self.idx_init = encode_lambda([char_init])[0]
        self.onehot_init = onehot_encoder(onehot_encoder(np.array([self.idx_init]).T, encode=True))

    def sample(self, lenght, p):
        """ Weighted sampling of next character based on RNN predicitons."""
        # select character from softmax weighted dist over all chars
        return np.random.choice(range(lenght), size=1, replace=True, p=p.flatten())

    def __call__(self, step):
        if step % self.n_step == 0:
            assert self.onehot_init.shape == (1, self.rnn.out_dim)
            x = deepcopy(self.onehot_init)
            sequence = []

            for t in range(self.ts):
                p = self.rnn.forward(x)
                x_idx = self.sample(lenght=self.rnn.out_dim, p=p)
                x_char = self.decode_lambda(x_idx)[0]
                sequence.append(x_char)
                x = self.onehot_encoder(np.array([x_idx]).T, encode=True)

            sequence_str = "".join(sequence)

            with open(self.path_out, "a") as f:
                f.write(f"step={step}\n\n")
                f.write(sequence_str)
                f.write("\n\n")

            tqdm.write(f"step={step}\n\n")
            tqdm.write(sequence_str)
            tqdm.write("\n\n")

        else:
            pass



class OneHotEncoder():
    """ One-hot encoder class.

    Attributes
    ----------
    length : int
        The length of the one-hot encoding.

    Methods
    -------
    __init__(layers)
        Constuctor.
    __call__(x, encode=True)
        Encode a sequence of integers into a one-hot encoded vectors,
        or decode a sequence of one-hot encoded vectors into a
        sequence of integers.
    __repr__()
        Returns the string representation of class.
    """

    def __init__(self, length):
        """ Constructor.

        Parameters
        ----------
        length : int
            The length of the one-hot encoding.

        Notes
        -----
        None
        """
        # length of one-hot encoding
        self.length = length

    def __call__(self, x, encode=True):
        """ Encode a sequence of integers into a one-hot encoded vectors,
        or decode a sequence of one-hot encoded vectors into a
        sequence of integers..

        Parameters
        ----------
        x : np.ndarray
            The sequence of index representation of chars, of shape (n_chars,)

        Returns
        -------
        e or d: np.ndarray
            The sequence of one-hot encoded vectors of chars, of shape (n_chars, length)

        Notes
        -----
        None
        """
        if encode:
            e = np.zeros((x.shape[0], self.length))
            e[np.arange(x.shape[0]), x] = 1
            return e.astype(int)
        else:
            d = np.argwhere(x == 1)[:, 1]
            return d.astype(int)

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
        repr_str = "one-hot encoder"
        return repr_str


def unique_characters(data):
    """ Get the list of unique characters in a data.

    Parameters
    ----------
    data : list
        A list of strings. The strings may be of different lenghts.

    Returns
    -------
    np.ndarray
        The list of unique characters in all of the strings in data.

    Notes
    -----
    None
    """
    chars = []

    for text in data:
        chars_current = list(dict.fromkeys(text))
        chars = list(dict.fromkeys(chars + chars_current))

    return np.array(chars)


def char_to_idx(char, chars):
    """ Convert a char to an index from the encoder np array.

    Parameters
    ----------
    char : str
        A char.
    chars : np.ndarray
        All chars.

    Returns
    -------
    np.ndarray
        The index repre of char, of shape (,).

    Notes
    -----
    None
    """
    return np.argwhere(char == chars).flatten()[0]


def idx_to_char(idx, chars):
    """ Convert an index to char in the encoder np array.

    Parameters
    ----------
    idx : int
        The index repr of a char.
    chars : np.ndarray
        All chars.

    Returns
    -------
    str
        The char.

    Notes
    -----
    None
    """
    return chars[idx]


def encode(decoding, chars):
    """ Encode a sequence of chars into a sequence of indices based on the encoder.

    Parameters
    ----------
    decoding : np.ndarray
        The sequence of chars, of shape (n_chars,)
    chars : np.ndarray
        All chars.

    Returns
    -------
    encoding : np.ndarray
        The sequence of index representation of the chars, of shape (n_chars,)

    Notes
    -----
    None
    """
    encoding = []

    for d in decoding:
        encoding.append(char_to_idx(d, chars))

    encoding = np.array(encoding)

    return encoding


def decode(encoding, chars):
    """ Decode a sequence of indices into a sequence of chars based on the encoder.

    Parameters
    ----------
    encoding : np.ndarray
        The sequence of index representation of the chars, of shape (n_chars,)
    chars : np.ndarray
        All chars.

    Returns
    -------
    decoding : np.ndarray
        The sequence of chars, of shape (n_chars,)

    Notes
    -----
    None
    """
    decoding = []

    for e in encoding:
        decoding.append(idx_to_char(e, chars))

    decoding = np.array(decoding)

    return decoding


def make_decoded_dataset(dataset):
    """ Decode a dataset of strings into a list of characters.

    Parameters
    ----------
    dataset : list
        A list of strings (contexts) maybe of varying size.

    Returns
    -------
    decoded_dataset : list
        A list of lists (contexts) where a context is a list of characters.

    Notes
    -----
    None
    """
    decoded_dataset = []
    for context in dataset:
        context_elements = list(context)
        decoded_dataset.append(context_elements)
    return decoded_dataset


def make_encoded_dataset(decoded_dataset, chars):
    """ Encode a dataset of list of charcters into a list of integers.

    Parameters
    ----------
    decoded_dataset : list
        A list of lists (contexts) where a context is a list of characters.
    chars : np.ndarray
        All chars.

    Returns
    -------
    encoded_dataset : list
        A list of lists (contexts) where a context is a list of integers.
        An integer corresponds to its index in chars.

    Notes
    -----
    None
    """
    encoded_dataset = []
    for decoded_context in decoded_dataset:
        encoded_context = encode(decoded_context, chars)
        encoded_dataset.append(encoded_context)
    return encoded_dataset


def make_one_hot_encoded_dataset(encoded_dataset, onehot_encoder):
    """ One-hot encode a dataset of list of integers into a list of one-hot encoded vectors.

    Parameters
    ----------
    encoded_dataset : list
        A list of lists (contexts) where a context is a list of integers.
        An integer corresponds to its index in chars.
    onehot_encoder : OneHotEncoder
        A one-hot encoder initilaized with chars (all unique characters in the dataset).

    Returns
    -------
    onehot_encoded_dataset : list
        A list of one-hot encoded vectors (contexts).
        The index of 1s in the vectors corresponds to the index of the character in chars.

    Notes
    -----
    None
    """
    onehot_encoded_dataset = []
    for encoded_context in encoded_dataset:
        onehot_encoded_context = onehot_encoder(encoded_context, encode=True)
        onehot_encoded_dataset.append(onehot_encoded_context)

    return onehot_encoded_dataset


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



class RNN():
    """ Many-to-many RNN layer for character-to-character sequence modelling.

    Attributes
    ----------
    in_dim : int
        Input dimension.
    out_dim : int
        Output dimension.
    hidden_dim : int
        Hidden dimension.
    kernel_h_initializer : Initializer
        The weight parameter initializer of the hidden neurons.
    bias_h_initializer : Initializer
        The bias parameter initializer of the hidden neurons.
    kernel_o_initializer : Initializer
        The weight parameter initializer of the output neurons.
    bias_o_initializer : Initializer
        The bias parameter initializer of the output neurons.
    kernel_regularizer : Regularizer
        The weight parameter regularizer for all parameters.
        Separate for h and o neurons. Not used yet.
    activation_h : Activation
        Layer activation of hidden neurons.
    activation_o : Activation
        Layer activation of output neurons.
    u : numpy.ndarray
        The weight parameters dotted with the input vector,
        of shape (in_dim, hidden_dim)
    w : numpy.ndarray
        The weight parameters dotted with the pre-activation hidden vector,
        of shape (hidden_dim, hidden_dim)
    b : numpy.ndarray
        The bias parameters added to the input-previous hidden vector
        linear combination, of shape (1, hidden_dim)
    v : numpy.ndarray
        The weight parameters dotted with the activated hidden vector,
        of shape (hidden_dim, out_dim)
    c : numpy.ndarray
        The bias parameters added to the dotted activated hidden vector,
        of shape (1, out_dim)
    cache : dict
        The run-time cache for storing activations, etc.
    grads : dict
        The run-time cache for storing gradients.
    h_shape : tuple
        Hidden vector shape.
    has_learnable_params : bool
        If layer has learnable/trainable params.

    Methods
    -------
    __init__(in_dim, out_dim, kernel_initializer, bias_initializer, kernel_regularizer, activation)
        Constructor.
    get_u()
        Returns the u parameters.
    get_w()
        Returns the w parameters.
    get_b()
        Returns the b parameters.
    get_v()
        Returns the v parameters.
    get_c()
        Returns the c parameters.
    set_u()
        Sets the u parameters.
    set_w()
        Sets the w parameters.
    set_b()
        Sets the b parameters.
    set_v()
        Sets the v parameters.
    set_c()
        Sets the c parameters.
    get_du()
        Returns the gradients of u parameters.
    get_dw()
        Returns the gradients of w parameters.
    get_db()
        Returns the gradients b parameters.
    get_dv()
        Returns the gradients of v parameters.
    get_dc()
        Returns the gradients c parameters.
    get_learnable_params()
        Get all learnable params.
    set_learnable_params(**learnable_params)
        Set all learnable params.
    get_learnable_params_grads()
        Get the gradients of the learnable params.
    get_reg_loss()
        Returns the regularization loss of the weight parameters.
    if_has_learnable_params()
        Returns if layer has learnable params.
    forward(x, **params)
        Forward-propagates signals through the layer and its activation.
    backward(g_in, **params)
        Back-propagates gradients through the the activation of the layer and then the layer.
        Note that the RNN layer implements backpropagation through time (BPTT).
    __repr__()
        Returns the string representation of class.
    """

    def __init__(self, in_dim, out_dim, hidden_dim,
                 kernel_h_initializer, bias_h_initializer,
                 kernel_o_initializer, bias_o_initializer,
                 kernel_regularizer,
                 activation_h, activation_o):
        """ Constructor.

        Parameters
        ----------
        in_dim : int
            Input dimension.
        out_dim : int
            Output dimension.
        hidden_dim : int
            Hidden dimension.
        kernel_h_initializer : Initializer
            The weight parameter initializer of the hidden neurons.
        bias_h_initializer : Initializer
            The bias parameter initializer of the hidden neurons.
        kernel_o_initializer : Initializer
            The weight parameter initializer of the output neurons.
        bias_o_initializer : Initializer
            The bias parameter initializer of the output neurons.
        kernel_regularizer : Regularizer
            The weight parameter regularizer for all parameters.
            Separate for h and o neurons. Not used yet.
        activation_h : Activation
            Layer activation of hidden neurons.
        activation_o : Activation
            Layer activation of output neurons.
        kwargs : dict
            Utils such as one-hot encoder.

        Notes
        -----
        None
        """
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim

        self.kernel_h_initializer = kernel_h_initializer
        self.bias_h_initializer = bias_h_initializer
        self.kernel_o_initializer = kernel_o_initializer
        self.bias_o_initializer = bias_o_initializer

        self.u = kernel_h_initializer.initialize(size=(in_dim, hidden_dim))
        self.w = kernel_h_initializer.initialize(size=(hidden_dim, hidden_dim))
        self.b = bias_h_initializer.initialize(size=(1, hidden_dim))

        self.v = kernel_o_initializer.initialize(size=(hidden_dim, out_dim))
        self.c = bias_o_initializer.initialize(size=(1, out_dim))

        self.kernel_regularizer = kernel_regularizer

        self.activation_h = activation_h
        self.activation_o = activation_o

        self.cache = {}
        self.grads = {}

        self.h_shape = (1, hidden_dim)
        self.cache["h"] = np.zeros(self.h_shape)

        self.has_learnable_params = True

    def forward(self, x, **params):
        """ Forward-propagates signals through the layer and its activation.

        Parameters
        ----------
        x : numpy.ndarray
            Input data to layer of shape (batch_size, in_dim).
        params : dict
            Dict of params for forward pass such as train or test mode, seed, etc.
            Unused in RNN layer.

        Returns
        -------
        p : numpy.ndarray
            Activation of the RNN layer output neurons, of shape (batch_size, out_dim).

        Notes
        -----
        Shapes are commented below.
        """
        # If first call, init h. If not, use the latest cached h.
        # used for inter-batch temporal information preservation
        h = deepcopy(self.cache["h"])
        self.cache["x"] = deepcopy(x)
        h_concat = np.zeros((x.shape[0], h.shape[1]))
        a_concat = np.zeros((x.shape[0], h.shape[1]))
        assert h.shape == (1, self.hidden_dim)

        for idx, x_ in enumerate(x):
            x_ = x_.reshape(1, -1)
            assert x_.shape == (1, self.in_dim)
            a = np.dot(x_, self.u) + np.dot(h, self.w) + self.b
            a_concat[idx] = a.reshape(1, -1)
            assert a.shape == (1, self.hidden_dim)
            h = self.activation_h.forward(a)
            h_concat[idx] = deepcopy(h)
            assert h.shape == (1, self.hidden_dim)

        # cache in the last hidden vector h for use in next batch
        # used for inter-batch temporal information preservation
        self.cache["h"] = deepcopy(h)
        self.cache["h_concat"] = deepcopy(h_concat)
        self.cache["a_concat"] = deepcopy(a_concat)
        assert h_concat.shape == (x.shape[0], h.shape[1])
        o = np.dot(h_concat, self.v) + self.c
        assert o.shape == (x.shape[0], self.out_dim)
        p = self.activation_o.forward(o)
        assert p.shape == (x.shape[0], self.out_dim)

        return p

    def backward(self, g_in, **params):
        """ Back-propagates gradients through the the activation of the layer and then the layer.
        Note that the RNN layer implements backpropagation through time (BPTT).

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
            Not implemented yet!

        Notes
        -----
        Shapes are commented below.
        """
        # x.shape = (x.shape[0], in_dim)
        x = deepcopy(self.cache["x"])
        # h_concat.shape = (x.shape[0], hidden_dim)
        h_concat = deepcopy(self.cache["h_concat"])
        a_concat = deepcopy(self.cache["a_concat"])

        # g_in.shape = (batch_size, )
        assert g_in.shape == (x.shape[0],)
        # g_a_o.shape = (batch_size, out_dim)
        g_a_o = self.activation_o.backward(g_in)
        assert g_a_o.shape == (x.shape[0], self.out_dim)

        # g_h_concat.shape = (batch_size, hidden_dim)
        g_h_concat = np.zeros((x.shape[0], self.hidden_dim))

        # v.shape = (hidden_dim, out_dim)
        # (1,hidden_dim) = (1,out_dim) * (hidden_dim, out_dim).T
        g_h_concat[-1] = np.dot(g_a_o[-1].reshape(1, -1), self.v.T)
        assert np.dot(g_a_o[-1].reshape(1, -1), self.v.T).shape == (1, self.hidden_dim)

        g_a = np.zeros((x.shape[0], self.hidden_dim))
        # (1, hidden_dim) = (1, hidden_dim) * (1, hidden_dim)
        # change cache (shapes)
        _ = self.activation_h.forward(a_concat[-1].reshape(1, -1))
        g_a[-1] = self.activation_h.backward(g_h_concat[-1]).reshape(1, -1)
        assert self.activation_h.backward(g_h_concat[-1].reshape(1, -1)).shape == (1, self.hidden_dim)

        for t in reversed(range(x.shape[0] - 1)):
            # (1,hidden_dim) = (1,out_dim) * (hidden_dim, out_dim).T
            # \+ (1,hidden_dim) * (hidden_dim, hidden_dim), maybe w.T?
            g_h_concat[t] = np.dot(g_a_o[t].reshape(1, -1), self.v.T) \
                            + np.dot(g_a[t + 1].reshape(1, -1), self.w)
            # change cache (shapes)
            _ = self.activation_h.forward(a_concat[t].reshape(1, -1))
            g_a[t] = self.activation_h.backward(g_h_concat[t])
            assert self.activation_h.backward(g_h_concat[t]).shape == (1, self.hidden_dim)

        assert g_h_concat.shape == (x.shape[0], self.hidden_dim)
        assert g_a.shape == (x.shape[0], self.hidden_dim)

        # (hidden_dim, out_dim) = (x.shape[0], hidden_dim).T * (x.shape[0], out_dim)
        g_v = np.dot(h_concat.T, g_a_o)
        assert g_v.shape == (self.hidden_dim, self.out_dim)
        self.grads["dv"] = deepcopy(g_v)

        # Auxiliar h matrix that includes h_prev
        h_aux = np.zeros(h_concat.shape)
        # h_init = np.zeros((1, self.hidden_dim))
        # h_aux[0, :] = h_init
        h_aux[0] = h_concat[-1].reshape(1, -1)
        h_aux[1:] = h_concat[0:-1]
        assert h_aux.shape == (x.shape[0], self.hidden_dim)

        # (hidden_dim, hidden_dim) = (x.shape[0], hidden_dim).T * (x.shape[0], hidden_dim)
        g_w = np.dot(h_aux.T, g_a)
        assert g_w.shape == (self.hidden_dim, self.hidden_dim)
        self.grads["dw"] = deepcopy(g_w)

        # (in_dim, hidden_dim) = (x.shape[0], in_dim).T * (x.shape[0], hidden_dim)
        g_u = np.dot(x.T, g_a)
        assert g_u.shape == (self.in_dim, self.hidden_dim)
        self.grads["du"] = deepcopy(g_u)

        # (1, hidden_dim) = sum((x.shape[0], self.hidden_dim), axis=0)
        g_b = np.sum(g_a, axis=0).reshape(1, -1)
        assert g_b.shape == (1, self.hidden_dim), f"g_b.shape={g_b.shape}"
        self.grads["db"] = deepcopy(g_b)

        # (1, out_dim) = sum((x.shape[0], self.out_dim), axis=0)
        g_c = np.sum(g_a_o, axis=0).reshape(1, -1)
        assert g_c.shape == (1, self.out_dim)
        self.grads["dc"] = deepcopy(g_c)

        # compute downstream grad!
        g_out = None
        return g_out

    def reset_hidden_state(self,):
        self.cache["h"] = np.zeros(self.h_shape)
        #print("resetting RNN h init vector\n")

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

    def get_u(self, ):
        """ Returns the u parameters.

        Parameters
        ----------
        None

        Returns
        -------
        numpy.ndarray
            The u parameters.

        Notes
        -----
        None
        """
        return deepcopy(self.u)

    def get_w(self, ):
        """ Returns the w parameters.

        Parameters
        ----------
        None

        Returns
        -------
        numpy.ndarray
            The w parameters.

        Notes
        -----
        None
        """
        return deepcopy(self.w)

    def get_b(self, ):
        """ Returns the b parameters.

        Parameters
        ----------
        None

        Returns
        -------
        numpy.ndarray
            The b parameters.

        Notes
        -----
        None
        """
        return deepcopy(self.b)

    def get_v(self, ):
        """ Returns the v parameters.

        Parameters
        ----------
        None

        Returns
        -------
        numpy.ndarray
            The v parameters.

        Notes
        -----
        None
        """
        return deepcopy(self.v)

    def get_c(self, ):
        """ Returns the c parameters.

        Parameters
        ----------
        None

        Returns
        -------
        numpy.ndarray
            The c parameters.

        Notes
        -----
        None
        """
        return deepcopy(self.c)

    def get_learnable_params(self):
        """ Get all learnable params.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            Dict of learanble params.

        Notes
        -----
        None
        """
        return {
            "u": self.get_u(), "w": self.get_w(), "b": self.get_b(),
            "v": self.get_v(), "c": self.get_c()
        }

    def set_u(self, u):
        """ Sets the u parameters.

        Parameters
        ----------
        u : numpy.ndarray
            The u parameters.

        Returns
        -------
        None

        Notes
        -----
        None
        """
        self.u = deepcopy(u)

    def set_w(self, w):
        """ Sets the w parameters.

        Parameters
        ----------
        w : numpy.ndarray
            The w parameters.

        Returns
        -------
        None

        Notes
        -----
        None
        """
        self.w = deepcopy(w)

    def set_b(self, b):
        """ Sets the b parameters.

        Parameters
        ----------
        b : numpy.ndarray
            The b parameters.

        Returns
        -------
        None

        Notes
        -----
        None
        """
        self.b = deepcopy(b)

    def set_v(self, v):
        """ Sets the v parameters.

        Parameters
        ----------
        v : numpy.ndarray
            The v parameters.

        Returns
        -------
        None

        Notes
        -----
        None
        """
        self.v = deepcopy(v)

    def set_c(self, c):
        """ Sets the c parameters.

        Parameters
        ----------
        c : numpy.ndarray
            The c parameters.

        Returns
        -------
        None

        Notes
        -----
        None
        """
        self.c = deepcopy(c)

    def set_learnable_params(self, **learnable_params):
        """ Set all learnable params.

        Parameters
        ----------
        learnable_params : dict
            Dict of learnable params.

        Returns
        -------
        None

        Notes
        -----
        None
        """
        self.set_u(learnable_params["u"])
        self.set_w(learnable_params["w"])
        self.set_b(learnable_params["b"])
        self.set_v(learnable_params["v"])
        self.set_c(learnable_params["c"])

    def get_du(self, ):
        """ Returns the gradients of u parameters.

        Parameters
        ----------
        None

        Returns
        -------
        ret : None or numpy.ndarray
            The gradients of u parameters, or None if does not exist yet.

        Notes
        -----
        None
        """
        if "du" in self.grads.keys():
            du = self.grads["du"]
            ret = deepcopy(du)
        else:
            ret = None

        return ret

    def get_dw(self, ):
        """ Returns the gradients of w parameters.

        Parameters
        ----------
        None

        Returns
        -------
        ret : None or numpy.ndarray
            The gradients of w parameters, or None if does not exist yet.

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
        """ Returns the gradients of b parameters.

        Parameters
        ----------
        None

        Returns
        -------
        ret : None or numpy.ndarray
            The gradients of b parameters, or None if does not exist yet.

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

    def get_dv(self, ):
        """ Returns the gradients of v parameters.

        Parameters
        ----------
        None

        Returns
        -------
        ret : None or numpy.ndarray
            The gradients of v parameters, or None if does not exist yet.

        Notes
        -----
        None
        """
        if "dv" in self.grads.keys():
            dv = self.grads["dv"]
            ret = deepcopy(dv)
        else:
            ret = None

        return ret

    def get_dc(self, ):
        """ Returns the gradients of c parameters.

        Parameters
        ----------
        None

        Returns
        -------
        ret : None or numpy.ndarray
            The gradients of c parameters, or None if does not exist yet.

        Notes
        -----
        None
        """
        if "dc" in self.grads.keys():
            dc = self.grads["dc"]
            ret = deepcopy(dc)
        else:
            ret = None

        return ret

    def get_learnable_params_grads(self):
        """ Get the gradients of the learnable params.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            Dict of grads of learanble params.

        Notes
        -----
        None
        """
        return {
            "du": self.get_du(), "dw": self.get_dw(), "db": self.get_db(),
            "dv": self.get_dv(), "dc": self.get_dc()
        }

    def get_reg_loss(self, ):
        return 0.0

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
        repr_str = "rnn: \n" \
                   + f"\t shape -- in: {self.in_dim}, out: {self.out_dim}, hidden: {self.hidden_dim}\n" \
                   + "\t u -- init: " + self.kernel_h_initializer.__repr__() + "\n" \
                   + "\t w -- init: " + self.kernel_h_initializer.__repr__() + "\n" \
                   + "\t b -- init: " + self.bias_h_initializer.__repr__() + "\n" \
                   + "\t v -- init: " + self.kernel_o_initializer.__repr__() + "\n" \
                   + "\t c -- init: " + self.bias_o_initializer.__repr__() + "\n" \
                   + ", reg: " + self.kernel_regularizer.__repr__() + "\n" \
                   + "\t activation: \n \t hidden: " + self.activation_h.__repr__() \
                   + "\t out: " + self.activation_o.__repr__() + "\n"
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

    def fit_rnn(self, x_train, y_train, x_val, y_val, n_epochs, batch_size, verbose, callbacks):

        assert self.compiled, "Model has to be compiled before fitting."
        assert isinstance(verbose, int) and verbose in [0, 1, 2], \
            f"verbose has to be an integer and in [0,1,2], but got {verbose} (type: {type(verbose)})"

        n_step = 0

        for n_epoch in range(n_epochs):
            if verbose in [1, 2]:
                print(f"starting epoch: {n_epoch + 1} ...")

            # Shuffle contexts in the beginning of each epoch
            indices_shuffle = list(range(len(x_train)))
            shuffle(indices_shuffle)
            x_train = [x for i, x in sorted(zip(indices_shuffle, x_train))]
            y_train = [x for i, x in sorted(zip(indices_shuffle, y_train))]

            for idx_context, (x_train_context, y_train_context) in enumerate(zip(x_train, y_train)):
                print(f"starting context: {idx_context + 1}/{len(x_train)} ...")
                n_batch = int(x_train_context.shape[0] / batch_size)

                if verbose in [2]:
                    batches = tqdm(range(n_batch), file=sys.stdout)
                else:
                    batches = range(n_batch)

                params_train = {"mode": "train", "seed": None}

                for b in batches:

                    x_batch = x_train_context[b * batch_size:(b + 1) * batch_size]
                    y_batch = y_train_context[b * batch_size + 1:(b + 1) * batch_size + 1]

                    # dirty solve: if cannot fit a y batch into context, skip the remaining skimmed-batch
                    if y_batch.shape[0] < batch_size:
                        continue

                    scores = self.forward(x_batch, **params_train)

                    layers_reg_loss = self.get_reg_loss()
                    data_loss = self.loss.compute_loss(scores, y_batch)
                    cost = data_loss + layers_reg_loss

                    self.backward(self.loss.grad(), **params_train)

                    trainable_params = \
                        self.optimizer.apply_grads(trainable_params=self.get_trainable_params(),
                                                   grads=self.get_gradients())

                    self.set_trainable_params(trainable_params)

                    self.loss_dict["loss_train"].append(data_loss)
                    self.lr_dict["lr"].append(self.optimizer.get_lr())

                    # should I do it here? yes
                    self.optimizer.apply_lr_schedule()

                    if verbose in [2]:
                        str_update = f"batch {b + 1}/{n_batch} (n_step: {n_step}), loss = {data_loss:.4f}"
                        batches.set_description(str_update)

                    for callback in callbacks:
                        callback(n_step)

                    n_step += 1

                # reset rnn h init here after each context
                # for the hp book a context is the entire book
                # for tweets, a context is one tweet
                for layer in self.layers:
                    #if isinstance(layer, RNN):
                    layer.reset_hidden_state()

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



class GradClipper():
    def __init__(self, repr_str):
        self.repr_str = repr_str

    def apply(self, grads_val):
        raise NotImplementedError

    def __call__(self, grads):
        # grads is a list of dicts, where each list is for a layer
        # and a dict is for the params' grads in that layer
        clipped_grads = deepcopy(grads)

        for idx in range(len(grads)):
            grad_dict = deepcopy(grads[idx])

            for g in grad_dict:
                clipped_grads[idx][g] = self.apply(grad_dict[g])

        return deepcopy(clipped_grads)

    def __repr__(self, ):
        return self.repr_str


class GradClipperByValue(GradClipper):
    def __init__(self, **kwargs):
        repr_str = "clipper by value"
        super().__init__(repr_str)
        self.val = kwargs["val"]

    def apply(self, grad_val):
        return np.maximum(np.minimum(grad_val, self.val), -self.val)


class GradClipperByNothing(GradClipper):
    def __init__(self, ):
        repr_str = "clipper who does nothing"
        super().__init__(repr_str)

    def apply(self, grad_val):
        return deepcopy(grad_val)



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

    def __init__(self, lr_schedule, grad_clipper, repr_str):
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
        self.grad_clipper = grad_clipper
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

    def __init__(self, lr_schedule, grad_clipper):
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
        repr_str = f"sgd with {lr_schedule.__repr__()} and {grad_clipper.__repr__()}"
        super().__init__(lr_schedule, grad_clipper, repr_str)

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
        if self.grad_clipper is not None:
            grads = deepcopy(self.grad_clipper(grads))

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

    def __init__(self, lr_schedule, grad_clipper, epsilon=1e-6):
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
        repr_str = f"adagrad with {lr_schedule.__repr__()} and {grad_clipper.__repr__()}"
        super().__init__(lr_schedule, grad_clipper, repr_str)
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
        if self.grad_clipper is not None:
            grads = deepcopy(self.grad_clipper(grads))

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


def plot_losses(history, filename):
    plt.plot(history["loss_train"], label="train")
    plt.grid()
    plt.title("Loss vs. update steps")
    plt.xlabel("update steps")
    plt.ylabel("Loss")
    plt.legend()
    path = f"{filename}_losses.png"
    plt.savefig(path)
    plt.show()


def plot_lr(history, filename):
    plt.plot(history["lr"], label="lr")
    plt.grid()
    plt.title("Learning rate vs. update steps")
    plt.xlabel("update steps")
    plt.ylabel("Learning rate")
    plt.legend()
    path = f"{filename}_lrs.png"
    plt.savefig(path)
    plt.show()


def give_emoji_free_text(text):
    """https://stackoverflow.com/a/50602709"""
    return emoji.get_emoji_regexp().sub(r'', text)


def add_eol_to_text(text, eol="."):
    return ''.join(list(text) + [eol])


def limit_text_length(df, col_name, max_length=140):
    mask = (df[col_name].str.len() <= max_length)
    df_filtered = df.loc[mask]
    print(f"reduced size from {df.shape[0]} to {df_filtered.shape[0]}\n")
    return df_filtered


def read_data():
    path = "data/dt/realdonaldtrump.csv"
    nrows = 105000
    df_raw = pd.read_csv(path, delimiter=",", usecols=["content"], nrows=nrows)
    print(f"loaded {df_raw.size} number of Trump tweets")

    print(df_raw["content"][:154])

    l = ['ه', 'ذ', 'ا', 'م', 'ق', 'د', 'ت',
         'ب', 'و', 'ع', 'ل', 'ي', 'ة', 'ف', 'س', 'ط', 'ن', 'ص', 'أ', 'ج', 'ز', 'ء', 'ش', 'ر', 'ह',
         'म', 'भ', 'ा', 'र', 'त', 'आ', 'न', 'े', 'क', 'ल', 'ि', 'ए', '्', 'प', 'ै', 'ं', '।', 'स', 'ँ', 'ु', 'छ', 'ी',
         'घ', 'ट', 'ो', 'ब',
         'ग', 'ー',
         '\u200f', 'º', '\u200e', 'è',
         '★', 'É', '♡', '«', '»', 'ı', '\x92', 'í', '☞', '•', '《', 'ĺ', 'ñ',
         '\U0010fc00', 'ō', 'á', 'ğ', 'â', 'ú', ]

    print(df_raw.shape)
    for e in l:
        df_raw = df_raw[~df_raw["content"].str.contains(e)]
    print(df_raw.shape)

    # might take some time
    df_raw["text_noemo"] = df_raw["content"].apply(lambda x: give_emoji_free_text(x))

    print(df_raw["text_noemo"][:154])

    df_filtered = limit_text_length(df_raw, col_name="text_noemo", max_length=139)

    print(df_filtered["text_noemo"][:100])
    print(df_filtered["text_noemo"][13])
    print(len(df_filtered["text_noemo"][13]))
    print(df_filtered["text_noemo"][98])
    print(len(df_filtered["text_noemo"][98]))

    # might take some time
    eol = "."
    df_filtered["text_noemo_eol"] = df_filtered["text_noemo"].apply(lambda x: add_eol_to_text(x, eol=eol))

    print(df_filtered["text_noemo_eol"][:100])
    print(df_filtered["text_noemo_eol"][13])
    print(len(df_filtered["text_noemo_eol"][13]))
    print(df_filtered["text_noemo_eol"][98])
    print(len(df_filtered["text_noemo_eol"][98]))

    dataset = df_filtered["text_noemo_eol"].to_list()

    return dataset


def pre_process_data(dataset):
    chars = unique_characters(dataset)
    print(f"The number of unique characters is {chars.size}")
    print("The unique characters in all contexts are:")
    print(chars)
    onehot_encoder = OneHotEncoder(length=len(chars))

    decoded_dataset = make_decoded_dataset(dataset)
    print(decoded_dataset[0][:100])
    encoded_dataset = make_encoded_dataset(decoded_dataset, chars)
    print(encoded_dataset[0][:100])
    onehot_encoded_dataset = make_one_hot_encoded_dataset(encoded_dataset, onehot_encoder)
    print(onehot_encoded_dataset[0][:100])

    print(f"There are {len(onehot_encoded_dataset)} conetexts in the dataset.")
    print(f"The context at idx 0 has {onehot_encoded_dataset[0].shape[0]} characters"
          f" and each character is one-hot encoded into a vector of length {onehot_encoded_dataset[0].shape[1]}")

    eol = "."
    print(f"The chosen EOL is at index {np.argwhere(eol == chars)[0]} in the unique characters list.")
    encoded_eol = encode([eol], chars)
    onehot_encoded_eol = onehot_encoder(encoded_eol, encode=True)[0]
    print(f"The one-hot encoded EOL vector looks like this:")
    print(onehot_encoded_eol)

    return onehot_encoded_dataset, encoded_dataset, chars, onehot_encoder, eol


def numerical_gradient_testing(onehot_encoded_dataset, encoded_dataset):
    # batch size is the character sequence length for one update step (one forward, one backward pass,
    # and a parameter update step)
    batch_size = 25
    x_train_context = onehot_encoded_dataset[0]
    y_train_context = encoded_dataset[0]
    x_train_context_batch = x_train_context[:batch_size]
    y_train_context_batch = y_train_context[1:batch_size + 1]

    init_params = {"coeff": 1.0, "mean": 0.0, "std": 0.01}
    seed = 100
    kernel_h_initializer = NormalInitializer(seed=seed, **init_params)
    bias_h_initializer = NormalInitializer(seed=seed + 1, **init_params)
    kernel_o_initializer = NormalInitializer(seed=seed + 2, **init_params)
    bias_o_initializer = NormalInitializer(seed=seed + 3, **init_params)
    kernel_regularizer = None

    loss_smoother = LossSmootherConstant()
    loss = CategoricalCrossEntropyLoss(loss_smoother=loss_smoother)

    rnn = RNN(in_dim=x_train_context.shape[1], out_dim=x_train_context.shape[1], hidden_dim=5,
              kernel_h_initializer=kernel_h_initializer,
              bias_h_initializer=bias_h_initializer,
              kernel_o_initializer=kernel_o_initializer,
              bias_o_initializer=bias_o_initializer,
              kernel_regularizer=kernel_regularizer,
              activation_h=TanhActivation(),
              activation_o=SoftmaxActivation())

    print(rnn)

    layers = [rnn]
    model = Model(layers)

    numerical_gradient_check_model(x_train_context_batch, y_train_context_batch, model, loss)


def train(onehot_encoded_dataset, encoded_dataset, chars, onehot_encoder, eol):
    seed = 100
    init_params = {"coeff": 1.0, "mean": 0.0, "std": 0.01}
    kernel_h_initializer = NormalInitializer(seed=seed, **init_params)
    bias_h_initializer = NormalInitializer(seed=seed, **init_params)
    kernel_o_initializer = NormalInitializer(seed=seed, **init_params)
    bias_o_initializer = NormalInitializer(seed=seed, **init_params)
    kernel_regularizer = None

    rnn = RNN(in_dim=onehot_encoded_dataset[0].shape[1], out_dim=onehot_encoded_dataset[0].shape[1],
              hidden_dim=100,
              kernel_h_initializer=kernel_h_initializer,
              bias_h_initializer=bias_h_initializer,
              kernel_o_initializer=kernel_o_initializer,
              bias_o_initializer=bias_o_initializer,
              kernel_regularizer=kernel_regularizer,
              activation_h=TanhActivation(),
              activation_o=SoftmaxActivation())

    layers = [rnn]

    model = Model(layers)

    loss_smoother = LossSmootherMovingAverage(alpha=0.999)
    loss = CategoricalCrossEntropyLoss(loss_smoother=loss_smoother)

    # optimizer = SGDOptimizer(lr_schedule=LRConstantSchedule(lr_initial))
    kwargs_gc = {"val": 5}
    grad_clipper = GradClipperByValue(**kwargs_gc)

    # lr_initial = 0.1
    # lr_schedule = LRConstantSchedule(lr_initial)
    lr_initial = 5e-2
    lr_max = 2e-1
    step_size = int(104225 / 2)
    lr_schedule = LRCyclingSchedule(lr_initial, lr_max, step_size)
    optimizer = AdaGradOptimizer(lr_schedule=lr_schedule, grad_clipper=grad_clipper)

    n_epochs = 5
    batch_size = 25

    metrics = [AccuracyMetrics()]

    model.compile_model(optimizer, loss, metrics)
    print(model)

    verbose = 2

    char_init = eol
    encode_lambda = lambda d: encode(d, chars)
    decode_lambda = lambda e: decode(e, chars)
    n_step = 1000
    ts = 140
    path_out = "synth_callback_dt.txt"

    synthetizer = CharByCharSynhthetizer(rnn, char_init, encode_lambda, onehot_encoder, decode_lambda,
                                         ts, n_step, path_out)
    callbacks = [synthetizer]

    history = model.fit_rnn(onehot_encoded_dataset, encoded_dataset,
                            None, None, n_epochs, batch_size, verbose, callbacks)

    plot_losses(history, filename="rnn_dt")
    plot_lr(history, filename="rnn_dt")

    return rnn


def synthetize(rnn, eol, chars, onehot_encoder):
    gen_times = 100

    for gen_time in range(gen_times):
        char_init = eol
        encode_lambda = lambda d: encode(d, chars)
        decode_lambda = lambda e: decode(e, chars)
        # n_step is just a dummy variable here for teh callback to work.
        n_step = 1
        ts = 140
        path_out = "synth_callback_dt_final.txt"

        synthetizer = CharByCharSynhthetizer(rnn, char_init, encode_lambda, onehot_encoder, decode_lambda,
                                             ts, n_step, path_out)

        synthetizer(step=n_step)


def main():
    dataset = read_data()
    onehot_encoded_dataset, encoded_dataset, chars, onehot_encoder, eol = pre_process_data(dataset)
    numerical_gradient_testing(onehot_encoded_dataset, encoded_dataset)
    rnn = train(onehot_encoded_dataset, encoded_dataset, chars, onehot_encoder, eol)
    synthetize(rnn, eol, chars, onehot_encoder)


if __name__ == "__main__":
    main()
