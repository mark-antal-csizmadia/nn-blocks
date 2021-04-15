import numpy as np
from copy import deepcopy
from activations import SoftmaxActivation, LinearActivation, ReLUActivation


def test_linear_activation():
    size = (5, 3)
    data = np.random.normal(loc=0, scale=1, size=size)

    linear_activation = LinearActivation()
    fw = linear_activation.forward(data)
    np.testing.assert_array_equal(data, fw)

    size = (5, 3)
    data = np.random.normal(loc=0, scale=1, size=size)
    bw = linear_activation.backward(data)
    np.testing.assert_array_equal(data, bw)

    print("test_linear_activation passed")


def test_relu_activation():
    size = (5, 3)
    data = np.random.normal(loc=0, scale=1, size=size)

    relu_activation = ReLUActivation()
    fw = relu_activation.forward(data)
    fw_true = np.maximum(0, data)

    size = (5, 3)
    data = np.random.normal(loc=0, scale=1, size=size)

    bw = relu_activation.backward(data)
    bw_true = np.copy(data)
    bw_true[fw_true <= 0] = 0.0

    np.testing.assert_array_equal(fw, fw_true)
    np.testing.assert_array_equal(bw, bw_true)

    print("test_relu_activation passed")


def test_softmax_activation():
    # from: https://cs231n.github.io/linear-classify/
    z = np.array([[-2.85, 0.86, 0.28]])
    softmax_activation = SoftmaxActivation()
    fw = softmax_activation.forward(z)
    fw_true = np.array([[0.01544932, 0.63116335, 0.35338733]])
    np.testing.assert_almost_equal(fw, fw_true)

    # automated testing
    labels = 3
    batch_size = 5
    size = (batch_size, labels)
    z = np.random.normal(loc=0, scale=1, size=size)

    softmax_activation = SoftmaxActivation()
    fw = softmax_activation.forward(z)

    z -= np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    fw_true = exp_z / np.sum(exp_z, axis=1, keepdims=True)

    size = (batch_size,)
    g_in = np.random.randint(low=0, high=labels, size=size)

    bw = softmax_activation.backward(g_in)

    n = g_in.shape[0]
    bw_true = deepcopy(fw_true)
    bw_true[range(n), g_in] -= 1
    bw_true /= n

    assert fw.shape == (batch_size, 3)
    assert bw.shape == (batch_size, 3)
    np.testing.assert_array_equal(fw, fw_true)
    np.testing.assert_array_equal(bw, bw_true)

    print("test_softmax_activation passed")
