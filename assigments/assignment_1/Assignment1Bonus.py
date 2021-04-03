import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from copy import deepcopy
from math import sqrt, ceil
import pandas as pd
from sklearn.model_selection import train_test_split


# Helpers
def load_cfar10_batch(path):
    """ Based on: https://towardsdatascience.com/cifar-10-image-classification-in-tensorflow-5b501f7dc77c
    """
    with open(path, mode='rb') as file:
        # note the encoding type is 'latin1'
        batch = pickle.load(file, encoding='latin1')

    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = np.array(batch['labels'])

    return features, labels


def load_label_names():
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def plot_losses(history, model_type, reg_rate_l2, lr_initial):
    title = f"Loss vs. epochs (lambda={reg_rate_l2}, lr={lr_initial} with {model_type})"
    filename = f"assets/bonus/loss_lambda_{reg_rate_l2}_lr_{lr_initial}_{model_type}.png"
    plt.plot(history["loss_train"], label="train")
    plt.plot(history["loss_val"], label="val")
    plt.grid()
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    #plt.show()
    plt.savefig(filename)
    plt.close()


def plot_accuracies(history, model_type, reg_rate_l2, lr_initial):
    title = f"Accuracy vs. epochs (lambda={reg_rate_l2}, lr={lr_initial} with {model_type})"
    filename = f"assets/bonus/acc_lambda_{reg_rate_l2}_lr_{lr_initial}_{model_type}.png"
    plt.plot(history["accuracy_train"], label="train")
    plt.plot(history["accuracy_val"], label="val")
    plt.grid()
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    #plt.show()
    plt.savefig(filename)
    plt.close()


def visualize_grid(Xs, ubound=255.0, padding=1):
    """
    The function is taken from: https://cs231n.github.io/
    Reshape a 4D tensor of image data to a grid for easy visualization.
    Inputs:
    - Xs: Data of shape (N, H, W, C)
    - ubound: Output grid will have values scaled to the range [0, ubound]
    - padding: The number of blank pixels between elements of the grid
    """
    (N, H, W, C) = Xs.shape
    grid_size = int(ceil(sqrt(N)))
    grid_height = H * grid_size + padding * (grid_size - 1)
    grid_width = W * grid_size + padding * (grid_size - 1)
    grid = np.zeros((grid_height, grid_width, C))
    next_idx = 0
    y0, y1 = 0, H
    for y in range(grid_size):
        x0, x1 = 0, W
        for x in range(grid_size):
            if next_idx < N:
                img = Xs[next_idx]
                low, high = np.min(img), np.max(img)
                grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
                # grid[y0:y1, x0:x1] = Xs[next_idx]
                next_idx += 1
            x0 += W + padding
            x1 += W + padding
        y0 += H + padding
        y1 += H + padding
    # grid_max = np.max(grid)
    # grid_min = np.min(grid)
    # grid = ubound * (grid - grid_min) / (grid_max - grid_min)
    return grid


def viz_kernel(w, im_shape, model_type, reg_rate_l2, lr_initial, figsize=(8, 8)):
    w = w.reshape(im_shape + (-1,)).transpose(3, 0, 1, 2)
    plt.figure(figsize=figsize)
    plt.imshow(visualize_grid(w, padding=3).astype('uint8'))
    plt.gca().axis('off')
    #plt.show()
    filename = f"assets/bonus/w_viz_lambda_{reg_rate_l2}_lr_{lr_initial}_{model_type}.png"
    plt.savefig(filename)
    plt.close()


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
    layers_reg_loss = model.get_reg_loss()
    l = loss.compute_loss(scores, y, layers_reg_loss)

    return l


def get_num_gradient(model, loss, x, y, verbose, **param_dict):
    """
    The function is based on: https://cs231n.github.io/optimization-1/
    """
    l = loss_func(model, loss, x, y, **param_dict)

    param_val = deepcopy(param_dict["param_val"])
    grad = np.zeros(param_val.shape)
    h = 0.00001

    if verbose:
        pbar = tqdm(total=param_val.size)

    it = np.nditer(param_val, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        # retain original param values
        old_value = param_val[ix]
        # increment by h
        param_val[ix] = old_value + h
        param_dict["param_val"] = deepcopy(param_val)
        # lxph = l(x + h)
        lxph = loss_func(model, loss, x, y, **param_dict)
        # decrease by h
        param_val[ix] = old_value - h
        param_dict["param_val"] = deepcopy(param_val)
        # lxmh = l(x - h)
        lxmh = loss_func(model, loss, x, y, **param_dict)
        # reset param value to original value
        param_val[ix] = old_value
        # numerical grad
        grad[ix] = (lxph - lxmh) / (2 * h)
        it.iternext()

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
            layers_reg_loss = 0

            l = loss.compute_loss(scores, y, model_new.get_reg_loss())

            model_new.backward(loss.grad())

            grads_analytic = model_new.get_gradients()

            grad_analytic = deepcopy(grads_analytic[layer_idx]["d" + param_str])

            rel_error = np.abs(grad_analytic - grad_numerical) \
                        / (np.maximum(np.abs(grad_analytic), np.abs(grad_numerical)))

            decimal = 6
            np.testing.assert_array_almost_equal(grad_numerical, grad_analytic, decimal=decimal)
            print(f"analytic and numerical grads are equal up to {decimal} decimals")
            print(f"max rel error={np.max(rel_error):.6e}")
            print(f"passed\n")

    print(f"completed grad check\n")

# my little nn library starts here nad will be long. sorry but the assignment description says that I should upload
# in one single file


class Loss():
    def __init__(self, ):
        self.cache = {}


class CategoricalHingeLoss(Loss):
    def __init__(self, ):
        super().__init__()

    def compute_loss(self, scores, y, layers_reg_loss):
        c = scores.shape[1]
        n = y.shape[0]

        correct_class_scores = scores[range(n), y].reshape(n, 1)
        margin = np.maximum(0, scores - correct_class_scores + 1)
        margin[range(n), y] = 0  # do not consider correct class in loss
        data_loss = margin.sum() / n

        loss = data_loss + layers_reg_loss

        margin[margin > 0] = 1
        valid_margin_count = margin.sum(axis=1)
        # Subtract in correct class (-s_y)
        margin[range(n), y] -= valid_margin_count
        margin /= n
        self.cache["g"] = deepcopy(margin)

        return loss

    def grad(self, ):
        if "g" in self.cache.keys():
            return deepcopy(self.cache["g"])
        else:
            return None


class CategoricalCrossEntropyLoss(Loss):
    def __init__(self, ):
        super().__init__()

    def compute_loss(self, scores, y, layers_reg_loss):
        """
        scores.shape=(batch_size, K)
        y.shape=(K,)
        l.shape = ()

        scores are probabilities from softmax
        """
        self.cache["g"] = deepcopy(y)

        n = y.shape[0]

        # correct_logprobs.shape = (batch_size, )
        correct_logprobs = -np.log(scores[range(n), y])

        # compute the loss: average cross-entropy loss and regularization
        data_loss = np.sum(correct_logprobs) / n

        loss = data_loss + layers_reg_loss

        return loss

    def grad(self, ):
        if "g" in self.cache.keys():
            return deepcopy(self.cache["g"])
        else:
            return None


class Activation():
    def __init__(self, ):
        self.cache = {}


class LinearActivation(Activation):
    def __init__(self, ):
        super().__init__()

    def forward(self, z):
        return deepcopy(z)

    def backward(self, g):
        return deepcopy(g)

    def __repr__(self):
        repr_str = "linear"
        return repr_str


class ReLUActivation(Activation):
    def __init__(self, ):
        super().__init__()

    def forward(self, z):
        a = np.maximum(0, z)
        self.cache["a"] = deepcopy(a)
        return a

    def backward(self, g_in):
        a = self.cache["a"]
        g_out = deepcopy(g_in)
        g_out[a <= 0] = 0.0
        return g_out

    def __repr__(self):
        repr_str = "relu"
        return repr_str


class SoftmaxActivation(Activation):
    def __init__(self, ):
        super().__init__()

    def forward(self, z):
        """
        z.shape = (batch_size, K)
        a.shape = (batch_size, K)
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
        # dscores.shape = (batch_size, K)
        # g_in is y, y.shape = (K,)G IN
        # g_out.shape = (batch_size, K)

        n = g_in.shape[0]
        a = self.cache["a"]
        g_out = deepcopy(a)
        g_out[range(n), g_in] -= 1
        g_out /= n

        return g_out

    def __repr__(self):
        repr_str = "softmax"
        return repr_str


class Initializer():
    def __init__(self, seed=None):
        self.seed = seed


class NormalInitializer(Initializer):
    def __init__(self, seed, **params):
        super().__init__(seed)
        self.coeff = params["coeff"]
        self.mean = params["mean"]
        self.std = params["std"]

    def initialize(self, size):
        np.random.seed(self.seed)
        return self.coeff * np.random.normal(loc=self.mean, scale=self.std, size=size)

    def __repr__(self):
        repr_str = "normal ~ " + f"{self.coeff} x ({self.mean}, {self.std}^2)"
        return repr_str


class XavierInitializer(Initializer):
    def __init__(self, seed, **params):
        super().__init__(seed)
        self.coeff = params["coeff"]
        self.mean = params["mean"]
        assert params["std"] is None, "Xavier init takes no std"

    def initialize(self, size):
        # size=(in_dim, out_dim)
        np.random.seed(self.seed)
        in_dim = size[0]
        self.std = 1 / np.sqrt(in_dim)
        return self.coeff * np.random.normal(loc=self.mean, scale=self.std, size=size)

    def __repr__(self):
        repr_str = "Xavier ~ " + f"{self.coeff} x ({self.mean}, {self.std}^2)"
        return repr_str


class Regularizer():
    def __init__(self, reg_rate):
        self.reg_rate = reg_rate


class L2Regularizer(Regularizer):
    def __init__(self, reg_rate):
        super().__init__(reg_rate)

    def loss(self, param):
        return 0.5 * self.reg_rate * np.sum(np.power(param, 2))

    def grad(self, param):
        return self.reg_rate * param

    def __repr__(self, ):
        repr_str = "l2"
        return repr_str


class LRSchedule():
    def __init__(self, lr_initial):
        self.lr_initial = lr_initial
        self.lr = self.lr_initial
        self.step = 0


class LRConstantSchedule(LRSchedule):
    def __init__(self, lr_initial):
        super().__init__(lr_initial)

    def apply_schedule(self, ):
        pass

    def get_lr(self, ):
        return self.lr


class LRExponentialDecaySchedule(LRSchedule):
    def __init__(self, lr_initial, decay_steps, decay_rate):
        super().__init__(lr_initial)
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate

    def apply_schedule(self, ):
        self.lr = self.lr_initial * self.decay_rate ** (self.step / self.decay_steps)
        self.step += 1

    def get_lr(self, ):
        return self.lr


class Optimizer():
    def __init__(self, lr_schedule):
        self.lr_schedule = lr_schedule
        self.lr = self.lr_schedule.get_lr()

    def apply_lr_schedule(self, ):
        self.lr_schedule.apply_schedule()
        self.lr = self.lr_schedule.get_lr()


class SGDOptimizer(Optimizer):
    def __init__(self, lr_schedule):
        super().__init__(lr_schedule)

    def apply_grads(self, trainable_weights, grads):
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


class Dense():
    def __init__(self, in_dim, out_dim, kernel_initializer, bias_initializer, kernel_regularizer, activation):

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
        return deepcopy(self.w)

    def get_b(self, ):
        return deepcopy(self.b)

    def set_w(self, w):
        self.w = deepcopy(w)

    def set_b(self, b):
        self.b = deepcopy(b)

    def get_dw(self, ):
        if "dw" in self.grads.keys():
            dw = self.grads["dw"]
            ret = deepcopy(dw)
        else:
            ret = None

        return ret

    def get_db(self, ):
        if "db" in self.grads.keys():
            db = self.grads["db"]
            ret = deepcopy(db)
        else:
            ret = None

        return ret

    def get_reg_loss_w(self, ):
        if self.kernel_regularizer is None:
            return 0.0
        else:
            return self.kernel_regularizer.loss(self.w)

    def get_reg_grad_w(self, ):
        if self.kernel_regularizer is None:
            return 0.0
        else:
            return self.kernel_regularizer.grad(self.w)

    def forward(self, x):
        """
        x.shape = (batch_size, in_dim)
        self.w.shape=(in_dim, out_dim)
        self.b.shape=(1, out_dim)
        z.shape = (batch, out_dim)
        a.shape = (batch, out_dim)
        """
        self.cache["x"] = deepcopy(x)
        z = np.dot(x, self.w) + self.b
        a = self.activation.forward(z)

        return a

    def backward(self, g_in):
        """
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
        repr_str = "dense: \n" \
                   + "\t w -- init:" + self.kernel_initializer.__repr__() \
                   + ", reg: " + self.kernel_regularizer.__repr__() + "\n" \
                   + "\t b -- init: " + self.bias_initializer.__repr__() + "\n" \
                   + "\t activation: " + self.activation.__repr__() + "\n"
        return repr_str


class Metrics():
    def __init__(self, ):
        pass


class AccuracyMetrics(Metrics):
    def __init__(self, ):
        super().__init__()
        self.name = "accuracy"

    def get_metrics(self, y, y_hat):
        assert y.shape == y_hat.shape

        n = y.shape[0]
        return np.where(y_hat == y)[0].size / n


class Model():
    def __init__(self, layers):
        self.layers = layers
        self.reg_loss = 0.0
        self.compiled = False

    def forward(self, x):
        scores = deepcopy(x)

        self.reg_loss = 0.0

        for layer in self.layers:
            scores_temp = layer.forward(scores)
            scores = deepcopy(scores_temp)
            self.reg_loss += layer.get_reg_loss_w()

        return scores

    def backward(self, y):
        g = deepcopy(y)

        for layer in list(reversed(self.layers)):
            g_temp = layer.backward(g)
            g = deepcopy(g_temp)

    def get_reg_loss(self, ):
        return self.reg_loss

    def get_gradients(self, ):
        grads = []
        for idx, layer in enumerate(self.layers):
            dw = layer.get_dw()
            db = layer.get_db()
            grads.append({"dw": deepcopy(dw), "db": deepcopy(db)})

        return deepcopy(grads)

    def get_trainable_weights(self, ):
        trainable_weights = []
        for idx, layer in enumerate(self.layers):
            w = layer.get_w()
            b = layer.get_b()
            trainable_weights.append({"w": deepcopy(w), "b": deepcopy(b)})

        return deepcopy(trainable_weights)

    def set_trainable_weights(self, trainable_weights):

        for idx, layer in enumerate(self.layers):
            trainable_weight_dict = deepcopy(trainable_weights[idx])
            w = trainable_weight_dict["w"]
            b = trainable_weight_dict["b"]
            layer.set_w(deepcopy(w))
            layer.set_b(deepcopy(b))

    def compile_model(self, optimizer, loss, metrics):
        self.optimizer = optimizer
        self.loss = loss

        metrics_train = {metric.name + "_train": [] for metric in metrics}
        metrics_val = {metric.name + "_val": [] for metric in metrics}
        self.metrics_dict = {**metrics_train, **metrics_val}
        self.loss_dict = {"loss_train": [], "loss_val": []}
        self.metrics = metrics

        self.compiled = True

    def fit(self, x_train, y_train, x_val, y_val, n_epochs, batch_size):
        assert self.compiled, "Model has to be compiled before fitting."

        for n_epoch in range(n_epochs):
            print(f"starting epoch: {n_epoch + 1} ...")

            # Shuffle data
            indices = np.arange(x_train.shape[0])
            np.random.shuffle(indices)
            x_train = x_train[indices]
            y_train = y_train[indices]

            n_batch = int(x_train.shape[0] / batch_size)

            batches = tqdm(range(n_batch))
            for b in batches:
                batches.set_description(f"batch {b + 1}/{n_batch}")
                x_batch = x_train[b * batch_size:(b + 1) * batch_size]
                y_batch = y_train[b * batch_size:(b + 1) * batch_size]

                scores = self.forward(x_batch)

                layers_reg_loss = self.get_reg_loss()
                l = self.loss.compute_loss(scores, y_batch, layers_reg_loss)

                self.backward(self.loss.grad())

                trainable_weights = \
                    self.optimizer.apply_grads(trainable_weights=self.get_trainable_weights(),
                                               grads=self.get_gradients())

                self.set_trainable_weights(trainable_weights)

            scores_train = self.forward(x_train)

            layers_reg_loss = self.get_reg_loss()
            l_train = self.loss.compute_loss(scores_train, y_train, layers_reg_loss)

            y_hat_train = np.argmax(scores_train, axis=1)

            scores_val = self.forward(x_val)

            layers_reg_loss = self.get_reg_loss()
            l_val = self.loss.compute_loss(scores_val, y_val, layers_reg_loss)

            y_hat_val = np.argmax(scores_val, axis=1)

            self.loss_dict["loss_train"].append(l_train)
            self.loss_dict["loss_val"].append(l_val)

            train_str = f"train loss = {l_train}"  # ", train acc = {acc_train}"
            val_str = f"val loss = {l_val}"

            for metrics in self.metrics:
                metrics_value_train = metrics.get_metrics(y_train, y_hat_train)
                self.metrics_dict[metrics.name + "_train"].append(metrics_value_train)
                train_str += f", train {metrics.name} = {metrics_value_train}"

                metrics_value_val = metrics.get_metrics(y_val, y_hat_val)
                self.metrics_dict[metrics.name + "_val"].append(metrics_value_val)
                val_str += f", val {metrics.name} = {metrics_value_val}"

            print(f"epoch {n_epoch + 1}/{n_epochs} \n "
                  f"\t -- {train_str} \n"
                  f"\t -- {val_str} \n\n")

            self.optimizer.apply_lr_schedule()

        return {**self.metrics_dict, **self.loss_dict}

    def predict(self, x, y):
        scores = self.forward(x)
        layers_reg_loss = self.get_reg_loss()
        l = self.loss.compute_loss(scores, y, layers_reg_loss)
        y_hat = np.argmax(scores, axis=1)

        accuracy_metrics = AccuracyMetrics()
        acc = accuracy_metrics.get_metrics(y, y_hat)

        print(f"test: -- loss={l:.4f}, accuracy={acc:.4f}")

        return l, acc

    def __repr__(self, ):
        repr_str = "model summary: \n"
        for idx, layer in enumerate(self.layers):
            repr_str = repr_str + f"layer {idx}: " + layer.__repr__() + "\n"

        return repr_str


def run_with_hyperparams_smax_cross_ent(x_train, y_train, x_val, y_val, x_test, y_test, reg_rate_l2, n_epochs, batch_size, lr_initial):

    model_type = "smax+cross_ent"
    print(f"starting with lambda={reg_rate_l2}, lr={lr_initial} with {model_type}")

    init_params_w = {"coeff": 1.0, "mean": 0.0, "std": None}
    init_params_b = {"coeff": 1.0, "mean": 0.0, "std": 0.01}

    in_dim = x_train.shape[1]
    out_dim = 10

    dense_1 = \
        Dense(in_dim=in_dim, out_dim=out_dim,
              kernel_initializer=XavierInitializer(seed=None, **init_params_w),
              bias_initializer=NormalInitializer(seed=None, **init_params_b),
              kernel_regularizer=L2Regularizer(reg_rate=reg_rate_l2),
              activation=SoftmaxActivation()
              )

    layers = [
        dense_1
    ]

    model = Model(layers)
    #print(model)

    loss = CategoricalCrossEntropyLoss()

    decay_steps = 80
    decay_rate = 0.9

    optimizer = SGDOptimizer(lr_schedule=LRExponentialDecaySchedule(lr_initial, decay_steps, decay_rate))
    metrics = [AccuracyMetrics()]

    model.compile_model(optimizer, loss, metrics)
    history = model.fit(x_train, y_train, x_val, y_val, n_epochs, batch_size)

    plot_losses(history, model_type, reg_rate_l2, lr_initial)
    plot_accuracies(history, model_type, reg_rate_l2, lr_initial)

    w = model.get_trainable_weights()[-1]["w"]
    im_shape = (32, 32, 3)
    viz_kernel(w, im_shape, model_type, reg_rate_l2, lr_initial, figsize=(8, 8))

    loss_train, accuracy_train = history["loss_train"][-1], history["accuracy_train"][-1]
    loss_val, accuracy_val = history["loss_val"][-1], history["accuracy_val"][-1]
    loss_test, accuracy_test = model.predict(x_test, y_test)

    results_dict = {
        "reg_rate_l2": reg_rate_l2,
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "lr_initial": lr_initial,
        "loss_train": loss_train,
        "accuracy_train": accuracy_train,
        "loss_val": loss_val,
        "accuracy_val": accuracy_val,
        "loss_test": loss_test,
        "accuracy_test": accuracy_test
    }

    return results_dict


def run_with_hyperparams_linear_hinge_loss(x_train, y_train, x_val, y_val, x_test, y_test, reg_rate_l2, n_epochs, batch_size, lr_initial):

    model_type = "linear+hinge"
    print(f"starting with lambda={reg_rate_l2}, lr={lr_initial} with {model_type}")

    init_params_w = {"coeff": 1.0, "mean": 0.0, "std": None}
    init_params_b = {"coeff": 1.0, "mean": 0.0, "std": 0.01}

    in_dim = x_train.shape[1]
    out_dim = 10

    dense_1 = \
        Dense(in_dim=in_dim, out_dim=out_dim,
              kernel_initializer=XavierInitializer(seed=None, **init_params_w),
              bias_initializer=NormalInitializer(seed=None, **init_params_b),
              kernel_regularizer=L2Regularizer(reg_rate=reg_rate_l2),
              activation=LinearActivation()
              )

    layers = [
        dense_1
    ]

    model = Model(layers)
    #print(model)

    loss = CategoricalHingeLoss()

    decay_steps = 80
    decay_rate = 0.9

    optimizer = SGDOptimizer(lr_schedule=LRExponentialDecaySchedule(lr_initial, decay_steps, decay_rate))

    metrics = [AccuracyMetrics()]

    model.compile_model(optimizer, loss, metrics)
    history = model.fit(x_train, y_train, x_val, y_val, n_epochs, batch_size)

    plot_losses(history, model_type, reg_rate_l2, lr_initial)
    plot_accuracies(history, model_type, reg_rate_l2, lr_initial)

    w = model.get_trainable_weights()[-1]["w"]
    im_shape = (32, 32, 3)
    viz_kernel(w, im_shape, model_type, reg_rate_l2, lr_initial, figsize=(8, 8))

    loss_train, accuracy_train = history["loss_train"][-1], history["accuracy_train"][-1]
    loss_val, accuracy_val = history["loss_val"][-1], history["accuracy_val"][-1]
    loss_test, accuracy_test = model.predict(x_test, y_test)

    results_dict = {
        "reg_rate_l2": reg_rate_l2,
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "lr_initial": lr_initial,
        "loss_train": loss_train,
        "accuracy_train": accuracy_train,
        "loss_val": loss_val,
        "accuracy_val": accuracy_val,
        "loss_test": loss_test,
        "accuracy_test": accuracy_test
    }

    return results_dict


def test_grad_check_with_hinge_loss(x_train, y_train):
    # Grad check
    params = {"coeff": 1.0, "mean": 0.0, "std": 0.01}

    in_dim = x_train.shape[1]
    out_dim = 10

    dense_1 = \
        Dense(in_dim=in_dim, out_dim=out_dim,
              kernel_initializer=NormalInitializer(seed=None, **params),
              bias_initializer=NormalInitializer(seed=None, **params),
              kernel_regularizer=None,
              activation=LinearActivation()
              )

    layers = [
        dense_1
    ]

    model = Model(layers)

    loss = CategoricalHingeLoss()

    verbose = True
    grad_check_without_reg(model, loss, x_train[:2], y_train[:2], verbose, seed=102)


def main():
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

    #np.testing.assert_allclose(train_ratios, val_ratios, rtol=1e-1, atol=0)
    #np.testing.assert_allclose(val_ratios, test_ratios, rtol=1e-1, atol=0)

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

    # grad check first
    test_grad_check_with_hinge_loss(x_train, y_train)

    # smax + cross entropy first
    hyperparams_dict = [
        {"reg_rate_l2": 0.1, "n_epochs": 60, "batch_size": 100, "lr_initial": 0.001},
        {"reg_rate_l2": 0.2, "n_epochs": 60, "batch_size": 100, "lr_initial": 0.001},
        {"reg_rate_l2": 0.4, "n_epochs": 60, "batch_size": 100, "lr_initial": 0.001},
        {"reg_rate_l2": 0.6, "n_epochs": 60, "batch_size": 100, "lr_initial": 0.001},
    ]

    results_list_of_dicts = []

    for hyperparam_dict in hyperparams_dict:
        reg_rate_l2 = hyperparam_dict["reg_rate_l2"]
        n_epochs = hyperparam_dict["n_epochs"]
        batch_size = hyperparam_dict["batch_size"]
        lr_initial = hyperparam_dict["lr_initial"]

        results_dict = run_with_hyperparams_smax_cross_ent(x_train, y_train, x_val, y_val, x_test, y_test, reg_rate_l2, n_epochs, batch_size, lr_initial)

        results_list_of_dicts.append(results_dict)

    df = pd.DataFrame(results_list_of_dicts)
    path = "assets/bonus/smax_cross_ent_results.csv"
    df.to_csv(path, index=False)

    # linear + hinge loss second
    hyperparams_dict = [
        {"reg_rate_l2": 0.5, "n_epochs": 60, "batch_size": 100, "lr_initial": 0.0001},
        {"reg_rate_l2": 0.5, "n_epochs": 60, "batch_size": 100, "lr_initial": 0.0001},
        {"reg_rate_l2": 0.8, "n_epochs": 60, "batch_size": 100, "lr_initial": 0.0001},
        {"reg_rate_l2": 0.8, "n_epochs": 60, "batch_size": 100, "lr_initial": 0.0001},
    ]

    results_list_of_dicts = []

    for hyperparam_dict in hyperparams_dict:
        reg_rate_l2 = hyperparam_dict["reg_rate_l2"]
        n_epochs = hyperparam_dict["n_epochs"]
        batch_size = hyperparam_dict["batch_size"]
        lr_initial = hyperparam_dict["lr_initial"]

        results_dict = run_with_hyperparams_linear_hinge_loss(x_train, y_train, x_val, y_val, x_test, y_test, reg_rate_l2,
                                                           n_epochs, batch_size, lr_initial)

        results_list_of_dicts.append(results_dict)

    df = pd.DataFrame(results_list_of_dicts)
    path = "assets/bonus/linear_hinge_results.csv"
    df.to_csv(path, index=False)


if __name__ == "__main__":
    main()
