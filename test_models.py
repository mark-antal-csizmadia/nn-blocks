import numpy as np
from initializers import NormalInitializer, XavierInitializer
from activations import LinearActivation, ReLUActivation, SoftmaxActivation
from regularizers import L2Regularizer
from layers import Dense, BatchNormalization
from models import Model
from losses import CategoricalCrossEntropyLoss, CategoricalHingeLoss, LossSmootherConstant
from grad_check import numerical_gradient_check_model


def build_model_2_layer_with_bn_with_loss_cross_entropy(reg_rate, in_dim, seed):
    params = {"coeff": 1.0, "mean": 0.0, "std": None}

    # in_dim = x.shape[1]
    out_dim = 10
    mid_dim = 20

    if reg_rate != 0.0:
        kernel_regularizer = L2Regularizer(reg_rate)
    else:
        kernel_regularizer = None

    dense_1 = \
        Dense(in_dim=in_dim, out_dim=mid_dim,
              kernel_initializer=XavierInitializer(seed=seed, **params),
              bias_initializer=XavierInitializer(seed=seed + 1, **params),
              kernel_regularizer=kernel_regularizer,
              activation=ReLUActivation()
              )
    bn_1 = BatchNormalization(momentum=0.9, epsilon=1e-5)
    dense_2 = \
        Dense(in_dim=mid_dim, out_dim=out_dim,
              kernel_initializer=XavierInitializer(seed=seed + 2, **params),
              bias_initializer=XavierInitializer(seed=seed + 3, **params),
              kernel_regularizer=kernel_regularizer,
              activation=SoftmaxActivation()
              )

    layers = [
        dense_1,
        bn_1,
        dense_2
    ]

    model = Model(layers)
    loss = CategoricalCrossEntropyLoss(loss_smoother=LossSmootherConstant())

    return model, loss


def test_models(seed=4):
    size = (2, 20)
    c = 10
    np.random.seed(seed + 1)
    x = np.random.normal(loc=0, scale=1, size=size)
    np.random.seed(seed + 2)
    y = np.random.randint(c, size=size[0])

    in_dim = x.shape[1]

    build_model_loss_func_list = [
        build_model_2_layer_with_bn_with_loss_cross_entropy
    ]

    np.random.seed(seed + 3)
    reg_rates = 10e-1 * np.random.randint(low=1, high=10, size=5)

    for reg_rate in reg_rates:
        for build_model_func in build_model_loss_func_list:
            model, loss = build_model_func(reg_rate, in_dim, seed)
            numerical_gradient_check_model(x, y, model, loss)
