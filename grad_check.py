from tqdm import tqdm
import sys
from copy import deepcopy
import numpy as np


def loss_func(model, loss, x, y, **param_dict):

    layer_idx = deepcopy(param_dict["layer_idx"])
    # w or b
    param_str = deepcopy(param_dict["param_str"])
    # np matrix
    param_val = deepcopy(param_dict["param_val"])

    trainable_params = model.get_trainable_params()
    trainable_params[layer_idx][param_str] = deepcopy(param_val)
    model.set_trainable_params(trainable_params)
    params = {"mode": "test"}
    scores = model.forward(x, **params)
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
    layer_n = len(model.layers)

    for layer_idx, trainable_params_in_layer in enumerate(model.get_trainable_params()):

        for param_str, param_val in trainable_params_in_layer.items():
            model_new = deepcopy(model)

            trainable_params = model_new.get_trainable_params()

            np.random.seed(seed)
            new_param_val = np.random.normal(loc=0, scale=0.01, size=param_val.shape)

            param_dict = {
                "layer_idx": layer_idx,
                "param_str": param_str,
                "param_val": new_param_val
            }

            trainable_params[layer_idx][param_str] = deepcopy(new_param_val)
            model_new.set_trainable_params(trainable_params)

            print(f"--layer: {layer_idx + 1}/{layer_n}, "
                  f"{param_str}.shape={param_val.shape} ({param_val.size} params)")

            grad_numerical = get_num_gradient(deepcopy(model_new), loss, x, y, verbose, **param_dict)

            scores = model_new.forward(x, **{"mode": "train"})
            assert model_new.get_reg_loss() == 0.0

            l = loss.compute_loss(scores, y)

            model_new.backward(loss.grad(), **{"mode": "train"})

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
