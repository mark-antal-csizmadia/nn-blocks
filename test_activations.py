import numpy as np
from copy import deepcopy
from activations import SoftmaxActivation, LinearActivation, ReLUActivation
from losses import CategoricalCrossEntropyLoss
from grad_check import eval_numerical_gradient_array, eval_numerical_gradient


def test_linear_activation(seed=np.random.randint(low=1, high=300)):
    size = (5, 3)
    np.random.seed(seed)
    x = np.random.normal(loc=0, scale=1, size=size)
    a_true = deepcopy(x)

    linear_activation = LinearActivation()
    a = linear_activation.forward(x)
    np.testing.assert_array_equal(a, a_true)

    size = (5, 3)
    np.random.seed(seed + 1)
    g_in = np.random.normal(loc=0, scale=1, size=size)

    g_out = linear_activation.backward(g_in)
    fx = lambda x: LinearActivation.forward(linear_activation, x)
    g_out_num = eval_numerical_gradient_array(fx, x, g_in)
    np.testing.assert_array_almost_equal(g_out, g_out_num, decimal=10)

    print("test_linear_activation passed")


def test_relu_activation():
    """ Test cases from: https://cs231n.github.io/assignments2021/assignment2/ """
    x = np.linspace(-0.5, 0.5, num=12).reshape(3, 4)
    a_true = np.array([[0., 0., 0., 0., ],
                       [0., 0., 0.04545455, 0.13636364, ],
                       [0.22727273, 0.31818182, 0.40909091, 0.5, ]])

    relu_activation = ReLUActivation()
    a = relu_activation.forward(x)
    np.testing.assert_array_almost_equal(a, a_true, decimal=8)

    relu_activation = ReLUActivation()
    np.random.seed(231)
    x = np.random.randn(10, 10)
    g_in = np.random.randn(*x.shape)
    fx = lambda x: ReLUActivation.forward(relu_activation, x)
    g_out_num = eval_numerical_gradient_array(fx, x, g_in)
    g_out = relu_activation.backward(g_in)
    np.testing.assert_array_almost_equal(g_out, g_out_num, decimal=10)

    print("test_relu_activation passed")


def test_softmax_activation():
    """ Test cases from: https://cs231n.github.io/assignments2021/assignment2/ """

    def func(x):
        softmax_activation = SoftmaxActivation()
        a = softmax_activation.forward(x)
        categoical_cross_entropy_loss = CategoricalCrossEntropyLoss()
        loss = categoical_cross_entropy_loss.compute_loss(a, y)
        return loss

    # from: https://cs231n.github.io/linear-classify/
    x = np.array([[-2.85, 0.86, 0.28]])
    softmax_activation = SoftmaxActivation()
    a = softmax_activation.forward(x)
    a_true = np.array([[0.01544932, 0.63116335, 0.35338733]])
    np.testing.assert_almost_equal(a, a_true, decimal=7)

    np.random.seed(231)
    num_classes, num_inputs = 10, 50
    x = 0.001 * np.random.randn(num_inputs, num_classes)
    y = np.random.randint(num_classes, size=num_inputs)

    softmax_activation = SoftmaxActivation()
    a = softmax_activation.forward(x)
    categoical_cross_entropy_loss = CategoricalCrossEntropyLoss()
    loss = categoical_cross_entropy_loss.compute_loss(a, y)
    loss_grad = categoical_cross_entropy_loss.grad()
    g_out = softmax_activation.backward(loss_grad)
    g_out_num = eval_numerical_gradient(func, x, h=5e-6, verbose=False)

    np.testing.assert_array_almost_equal(g_out, g_out_num, decimal=2)

    print("test_softmax_activation passed")

