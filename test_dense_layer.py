import numpy as np
from layers import Dense
from initializers import NormalInitializer
from activations import LinearActivation, ReLUActivation, SoftmaxActivation


def test_dense_param_init(seed=np.random.randint(low=1, high=300)):
    assert seed is not None, "seed cannot be None"

    in_dim = 5
    out_dim = 5
    coeff = 1.0
    mean = 0.0
    std = 1.0
    params = {"coeff": coeff, "mean": mean, "std": std}

    kernel_initializer = NormalInitializer(seed=seed, **params)
    bias_initializer = NormalInitializer(seed=seed, **params)

    w_true = kernel_initializer.initialize(size=(in_dim, out_dim))
    b_true = bias_initializer.initialize(size=(1, out_dim))

    kernel_regularizer = None
    activation = None

    dense = Dense(in_dim, out_dim, kernel_initializer, bias_initializer, kernel_regularizer, activation)

    w = dense.get_w()
    b = dense.get_b()

    assert w.shape == (in_dim, out_dim)
    assert b.shape == (1, out_dim)
    np.testing.assert_array_equal(w, w_true)
    np.testing.assert_array_equal(b, b_true)

    print("test_dense_param_init passed")


def test_dense_forward(seed=np.random.randint(low=1, high=300)):
    assert seed is not None, "seed cannot be None"

    in_dim = 5
    out_dim = 2
    batch_size = 3
    coeff = 1.0
    mean = 0.0
    std = 1.0
    params = {"coeff": coeff, "mean": mean, "std": std}

    kernel_initializer = NormalInitializer(seed=seed, **params)
    bias_initializer = NormalInitializer(seed=seed, **params)

    w_true = kernel_initializer.initialize(size=(in_dim, out_dim))
    b_true = bias_initializer.initialize(size=(1, out_dim))

    kernel_regularizer = None
    linear_activation = LinearActivation()
    relu_activation = ReLUActivation()
    softmax_activation = SoftmaxActivation()

    activations = [linear_activation, relu_activation, softmax_activation]

    for activation in activations:
        print(f"activation: {type(activation)}")

        dense = Dense(in_dim, out_dim, kernel_initializer, bias_initializer, kernel_regularizer, activation)

        w = dense.get_w()
        b = dense.get_b()

        assert w.shape == (in_dim, out_dim)
        assert b.shape == (1, out_dim)

        np.testing.assert_array_equal(w, w_true)
        np.testing.assert_array_equal(b, b_true)

        size = (batch_size, in_dim)
        np.random.seed(seed + 1)
        x = np.random.normal(loc=0, scale=1, size=size)

        z_true = np.dot(x, w_true) + b_true
        a_true = activation.forward(z_true)

        a = dense.forward(x)

        assert a.shape == (batch_size, out_dim)

        np.testing.assert_array_equal(a, a_true)

    print("test_dense_forward_with_linear_activation passing")


def test_dense_backward_relu_linear(seed=np.random.randint(low=1, high=300)):
    assert seed is not None, "seed cannot be None"

    in_dim = 5
    out_dim = 2
    batch_size = 3
    coeff = 1.0
    mean = 0.0
    std = 1.0
    params = {"coeff": coeff, "mean": mean, "std": std}

    kernel_initializer = NormalInitializer(seed=seed, **params)
    bias_initializer = NormalInitializer(seed=seed, **params)

    w_true = kernel_initializer.initialize(size=(in_dim, out_dim))
    b_true = bias_initializer.initialize(size=(1, out_dim))

    kernel_regularizer = None
    linear_activation = LinearActivation()
    relu_activation = ReLUActivation()

    activations = [linear_activation, relu_activation]

    for activation in activations:
        print(f"activation: {type(activation)}")

        dense = Dense(in_dim, out_dim, kernel_initializer, bias_initializer, kernel_regularizer, activation)

        w = dense.get_w()
        b = dense.get_b()

        assert w.shape == (in_dim, out_dim)
        assert b.shape == (1, out_dim)

        np.testing.assert_array_equal(w, w_true)
        np.testing.assert_array_equal(b, b_true)

        x_size = (batch_size, in_dim)
        np.random.seed(seed + 1)
        x = np.random.normal(loc=0, scale=1, size=x_size)

        z_true = np.dot(x, w_true) + b_true
        a_true = activation.forward(z_true)

        a = dense.forward(x)

        assert a.shape == (batch_size, out_dim)

        np.testing.assert_array_equal(a, a_true)

        g_in_size = (batch_size, out_dim)
        np.random.seed(seed + 2)
        g_in = np.random.normal(loc=0, scale=1, size=g_in_size)

        g_a_true = activation.backward(g_in)

        dw_true = np.dot(x.T, g_a_true)
        db_true = np.sum(g_a_true, axis=0, keepdims=True)

        g_out_true = np.dot(g_a_true, w_true.T)

        g_out = dense.backward(g_in)

        dw = dense.get_dw()
        db = dense.get_db()

        assert g_out.shape == (batch_size, in_dim)
        assert dw.shape == (in_dim, out_dim)
        assert db.shape == (1, out_dim), f"db.shape={db.shape}"

        np.testing.assert_array_equal(g_out, g_out_true)
        np.testing.assert_array_equal(dw, dw_true)
        np.testing.assert_array_equal(db, db_true)

    print("test_dense_backward_relu_linear passing")


def test_dense_backward_softmax(seed=np.random.randint(low=1, high=300)):
    assert seed is not None, "seed cannot be None"

    in_dim = 5
    out_dim = 2
    batch_size = 3
    coeff = 1.0
    mean = 0.0
    std = 1.0
    params = {"coeff": coeff, "mean": mean, "std": std}

    kernel_initializer = NormalInitializer(seed=seed, **params)
    bias_initializer = NormalInitializer(seed=seed, **params)

    w_true = kernel_initializer.initialize(size=(in_dim, out_dim))
    b_true = bias_initializer.initialize(size=(1, out_dim))

    kernel_regularizer = None
    softmax_activation = SoftmaxActivation()

    dense = Dense(in_dim, out_dim, kernel_initializer, bias_initializer, kernel_regularizer,
                  softmax_activation)

    w = dense.get_w()
    b = dense.get_b()

    assert w.shape == (in_dim, out_dim)
    assert b.shape == (1, out_dim)

    np.testing.assert_array_equal(w, w_true)
    np.testing.assert_array_equal(b, b_true)

    x_size = (batch_size, in_dim)
    np.random.seed(seed + 1)
    x = np.random.normal(loc=0, scale=1, size=x_size)

    z_true = np.dot(x, w_true) + b_true
    a_true = softmax_activation.forward(z_true)

    a = dense.forward(x)

    assert a.shape == (batch_size, out_dim)

    np.testing.assert_array_equal(a, a_true)

    g_in_size = (batch_size,)
    np.random.seed(seed + 2)
    g_in = np.random.randint(low=0, high=out_dim, size=g_in_size)

    g_out = dense.backward(g_in)

    g_a_true = softmax_activation.backward(g_in)

    dw_true = np.dot(x.T, g_a_true)
    db_true = np.sum(g_a_true, axis=0, keepdims=True)

    g_out_true = np.dot(g_a_true, w_true.T)

    dw = dense.get_dw()
    db = dense.get_db()

    assert g_out.shape == (batch_size, in_dim)
    assert dw.shape == (in_dim, out_dim)
    assert db.shape == (1, out_dim), f"db.shape={db.shape}"

    np.testing.assert_array_equal(g_out, g_out_true)
    np.testing.assert_array_equal(dw, dw_true)
    np.testing.assert_array_equal(db, db_true)

    print("test_dense_backward_softmax passing")
