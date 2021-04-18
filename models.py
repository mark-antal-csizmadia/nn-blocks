import numpy as np
from copy import deepcopy
from tqdm import tqdm
import sys


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
                self.reg_loss += layer.get_reg_loss_w()

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
                dw = layer.get_dw()
                db = layer.get_db()
            else:
                dw, db = None, None
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
            if layer.if_has_learnable_params():
                w = layer.get_w()
                b = layer.get_b()
            else:
                w, b = None, None
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
            if layer.if_has_learnable_params():
                layer.set_w(deepcopy(w))
                layer.set_b(deepcopy(b))
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

                """scores = self.forward(x_batch, **params_train)

                layers_reg_loss = self.get_reg_loss()
                data_loss = self.loss.compute_loss(scores, y_batch)

                cost = data_loss + layers_reg_loss"""

                y_hat, scores, cost, data_loss, layers_reg_loss = self.predict(x_batch, y_batch, **params_train)

                self.backward(self.loss.grad(), **params_train)

                trainable_weights = \
                    self.optimizer.apply_grads(trainable_weights=self.get_trainable_weights(),
                                               grads=self.get_gradients())

                self.set_trainable_weights(trainable_weights)

                # should I do it here?
                self.optimizer.apply_lr_schedule()

            params_test = {"mode": "test", "seed": None}

            """scores_train = self.forward(x_train, **params_test)

            layers_reg_loss_train = self.get_reg_loss()
            data_loss_train = self.loss.compute_loss(scores_train, y_train)
            cost_train = data_loss_train + layers_reg_loss_train

            y_hat_train = np.argmax(scores_train, axis=1)"""
            y_hat_train, scores_train, cost_train, data_loss_train, layers_reg_loss_train = \
                self.predict(x_train, y_train, **params_test)

            # acc_train = self.metrics.accuracy.get_accuracy(y_train, y_hat_train)

            """scores_val = self.forward(x_val, **params_test)

            layers_reg_loss_val = self.get_reg_loss()
            data_loss_val = self.loss.compute_loss(scores_val, y_val)
            cost_val = data_loss_val + layers_reg_loss_val

            # n_val = y_val.shape[0]
            y_hat_val = np.argmax(scores_val, axis=1)"""

            y_hat_val, scores_val, cost_val, data_loss_val, layers_reg_loss_val = \
                self.predict(x_val, y_val, **params_test)

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

    def predict(self, x, y, **params):
        """ Predicts labels for a given data set x, and computes the scores, data loss, and cost of the prediction,

        Parameters
        ----------
        x : numpy.ndarray
            Data of shape (batch_size, in_dim) where in_dim is
            the input dimension of the first layer of the Model.
        y : numpy.ndarray
            True labels of data.
            Shape is (batch_size, )
        params : dict
            Dict of params for forward pass such as train or test mode, seed, etc.

        Returns
        -------
        y_hat : numpy.ndarray
            Predicted/inferred labels of data.
            Shape is (batch_size, )
        scores : numpy.ndarray
            Activation of last layer of the model - the scores of the network.
            Shape is (batch_size, out_dim) where out_dim is the output
            dimension of the last layer of the model - usually same as
            the number of classes.
        cost : float
            Cost of prediction. Note that cost = data_loss + layers_reg_loss
        data_loss : float
            Data loss of prediction.
        layers_reg_loss : float
            The regularization loss from the layers of the model.

        Notes
        -----
        None

        Raises
        ------
        AssertionError
            If the model has not yet been complied with the self.compiled method.
        """
        scores = self.forward(x, **params)
        layers_reg_loss = self.get_reg_loss()
        data_loss = self.loss.compute_loss(scores, y)
        cost = data_loss + layers_reg_loss
        y_hat = np.argmax(scores, axis=1)

        return y_hat, scores, cost, data_loss, layers_reg_loss

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
