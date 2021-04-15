import numpy as np
from initializers import NormalInitializer, XavierInitializer


def test_normal_initializer(seed=np.random.randint(low=1, high=300)):
    assert seed is not None, "seed cannot be None"

    size = (5, 8)
    coeff = 1.0
    mean = 0.0
    std = 1.0
    params = {"coeff": coeff, "mean": mean, "std": std}

    normal_initializer = NormalInitializer(seed=seed, **params)
    w = normal_initializer.initialize(size=size)

    np.random.seed(seed)
    w_true = coeff * np.random.normal(loc=mean, scale=std, size=size)

    np.testing.assert_array_equal(w, w_true)

    print("test_normal_initializer passed")


def test_xavier_initializer(seed=np.random.randint(low=1, high=300)):
    assert seed is not None, "seed cannot be None"

    size = (5, 8)
    coeff = 1.0
    mean = 0.0
    std = None
    params = {"coeff": coeff, "mean": mean, "std": std}

    xavier_initializer = XavierInitializer(seed=seed, **params)
    w = xavier_initializer.initialize(size=size)

    in_dim = size[0]
    std = 1 / np.sqrt(in_dim)
    np.random.seed(seed)
    w_true = coeff * np.random.normal(loc=mean, scale=std, size=size)

    np.testing.assert_array_equal(w, w_true)

    print("test_xavier_initializer passed")
