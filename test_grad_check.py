import numpy as np
from initializers import NormalInitializer, XavierInitializer
from activations import LinearActivation, ReLUActivation, SoftmaxActivation
from layers import Dense
from models import Model
from losses import CategoricalCrossEntropyLoss, CategoricalHingeLoss
from grad_check import grad_check_without_reg


# unittest grad check without reg
# Grad check
def test_grad_check_without_reg_softmax_cat_cross_entropy_one_layer(seed=np.random.randint(low=1, high=300)):
    params = {"coeff": 1.0, "mean": 0.0, "std": 0.01}

    n_data = 2
    in_dim = 500
    out_dim = 5

    x_size = (n_data, in_dim)
    np.random.seed(seed)
    x = np.random.normal(loc=0, scale=1, size=x_size)

    y_size = (n_data,)
    np.random.seed(seed)
    y = np.random.randint(low=0, high=out_dim, size=y_size)

    dense_1 = \
        Dense(in_dim=in_dim, out_dim=out_dim,
              kernel_initializer=NormalInitializer(seed=None, **params),
              bias_initializer=NormalInitializer(seed=None, **params),
              kernel_regularizer=None,
              activation=SoftmaxActivation()
              )

    layers = [
        dense_1
    ]

    model = Model(layers)

    loss = CategoricalCrossEntropyLoss()

    verbose = True
    grad_check_without_reg(model, loss, x, y, verbose, seed=seed + 1)

    print("test_grad_check_without_reg_softmax_cat_cross_entropy_one_layer passed")


# unittest grad check without reg
# Grad check
def test_grad_check_without_reg_softmax_cat_cross_entropy_two_layers(seed=np.random.randint(low=1, high=300)):
    params = {"coeff": 1.0, "mean": 0.0, "std": None}

    n_data = 2
    in_dim = 100
    mid_dim = 10
    out_dim = 5

    x_size = (n_data, in_dim)
    np.random.seed(seed)
    x = np.random.normal(loc=0, scale=1, size=x_size)

    y_size = (n_data,)
    np.random.seed(seed)
    y = np.random.randint(low=0, high=out_dim, size=y_size)

    dense_1 = \
        Dense(in_dim=in_dim, out_dim=mid_dim,
              kernel_initializer=XavierInitializer(seed=None, **params),
              bias_initializer=XavierInitializer(seed=None, **params),
              kernel_regularizer=None,
              activation=ReLUActivation()
              )

    dense_2 = \
        Dense(in_dim=mid_dim, out_dim=out_dim,
              kernel_initializer=XavierInitializer(seed=None, **params),
              bias_initializer=XavierInitializer(seed=None, **params),
              kernel_regularizer=None,
              activation=SoftmaxActivation()
              )

    layers = [
        dense_1,
        dense_2
    ]

    model = Model(layers)

    loss = CategoricalCrossEntropyLoss()

    verbose = True
    grad_check_without_reg(model, loss, x, y, verbose, seed=seed + 1)

    print("test_grad_check_without_reg_softmax_cat_cross_entropy_two_layers passed")


# unittest grad check without reg
# Grad check
def test_grad_check_without_reg_linear_hinge_one_layer(seed=np.random.randint(low=1, high=300)):
    params = {"coeff": 1.0, "mean": 0.0, "std": 0.01}

    n_data = 2
    in_dim = 500
    out_dim = 5

    x_size = (n_data, in_dim)
    np.random.seed(seed)
    x = np.random.normal(loc=0, scale=1, size=x_size)

    y_size = (n_data,)
    np.random.seed(seed)
    y = np.random.randint(low=0, high=out_dim, size=y_size)

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
    grad_check_without_reg(model, loss, x, y, verbose, seed=seed + 1)

    print("test_grad_check_without_reg_linear_hinge_one_layer passed")
