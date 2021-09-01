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
    get_learnable_params()
        Get all learnable params.
    set_learnable_params(**learnable_params)
        Set all learnable params.
    get_learnable_params_grads()
        Get the gradients of the learnable params.
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
        if "seed" in params.keys():
            seed = params["seed"]
        assert mode in ["train", "test"]

        if mode == "train":
            if "seed" in params.keys():
                np.random.seed(seed)
            mask = (np.random.rand(*x.shape) < self.p) / self.p
            self.cache["mask"] = deepcopy(mask)
            # drop it boi!
            out = x * mask
        else:
            out = deepcopy(x)

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
