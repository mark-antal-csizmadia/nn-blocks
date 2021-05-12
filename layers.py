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
