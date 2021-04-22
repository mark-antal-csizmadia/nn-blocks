from tqdm import tqdm
import sys
from copy import deepcopy
import numpy as np


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
