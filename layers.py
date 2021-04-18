import numpy as np
from copy import deepcopy


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
