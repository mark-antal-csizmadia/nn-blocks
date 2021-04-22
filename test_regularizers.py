import numpy as np
from regularizers import L2Regularizer


def test_l2_regularizer(seed=np.random.randint(low=1, high=300)):
    np.random.seed(seed)
    reg_rate = np.random.uniform(low=0, high=1)
    l2_regularizer = L2Regularizer(reg_rate=reg_rate)

    size = (5, 3)
    param = np.random.normal(loc=0, scale=1, size=size)

    loss_true = 0.5 * reg_rate * np.sum(np.power(param, 2))
    loss = l2_regularizer.loss(param=param)

    np.testing.assert_array_equal(loss, loss_true)

    grad_true = reg_rate * param
    grad = l2_regularizer.grad(param=param)

    np.testing.assert_array_equal(grad, grad_true)

    print("test_l2_regularizer passed")
