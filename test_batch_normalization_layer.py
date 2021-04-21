import numpy as np
from layers import BatchNormalization
from grad_check import eval_numerical_gradient_array


def test_batch_normalization_layer():
    # Gradient check batchnorm backward pass
    np.random.seed(231)
    N, D = 4, 5
    x = 5 * np.random.randn(N, D) + 12
    gamma = np.random.randn(D)
    beta = np.random.randn(D)
    dout = np.random.randn(N, D)

    bn_param = {'mode': 'train'}

    def with_gamma(x, gamma, **bn_param):
        bn = BatchNormalization(momentum=0.9, epsilon=1e-5)
        bn.set_gamma(gamma)
        a = bn.forward(x, **bn_param)
        return a

    def with_beta(x, beta, **bn_param):
        bn = BatchNormalization(momentum=0.9, epsilon=1e-5)
        bn.set_beta(beta)
        a = bn.forward(x, **bn_param)
        return a

    bn = BatchNormalization(momentum=0.9, epsilon=1e-5)
    fx = lambda x: BatchNormalization.forward(bn, x, **bn_param)
    fg = lambda g: with_gamma(x, g, **bn_param)
    fb = lambda b: with_beta(x, b, **bn_param)

    dx_num = eval_numerical_gradient_array(fx, x, dout)
    dgamma_num = eval_numerical_gradient_array(fg, gamma.copy(), dout)
    dbeta_num = eval_numerical_gradient_array(fb, beta.copy(), dout)

    bn = BatchNormalization(momentum=0.9, epsilon=1e-5)
    bn.forward(x, **bn_param)
    dx = bn.backward(dout, **bn_param)
    dgamma = bn.grads["dgamma"]
    dbeta = bn.grads["dbeta"]

    np.testing.assert_array_almost_equal(dx, dx_num, decimal=10)
    np.testing.assert_array_almost_equal(dgamma, dgamma_num, decimal=10)
    np.testing.assert_array_almost_equal(dbeta, dbeta_num, decimal=10)
