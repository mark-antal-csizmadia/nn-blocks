import numpy as np
from copy import deepcopy
from tqdm import tqdm
import sys
import json


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

    def fit(self, x_train, y_train, x_val, y_val, n_epochs, batch_size):
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

            params_train = {"mode": "train", "seed": None}

            for b in batches:
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
