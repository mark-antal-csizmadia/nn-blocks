import numpy as np
from layers import Dropout
from copy import deepcopy


def test_dropout(seed=np.random.randint(low=1, high=300)):
    np.random.seed(seed)
    p = np.random.uniform(low=0, high=1)
    dropout = Dropout(p=p)
    size = (5, 8)
    mean = 0.0
    std = 1.0
    np.random.seed(seed)
    x = np.random.normal(loc=mean, scale=std, size=size)
    np.random.seed(seed + 1)
    g_in = np.random.normal(loc=mean, scale=std, size=size)
    modes = ["train", "test"]
    # modes = ["train"]
    for mode in modes:
        params = {"mode": mode, "seed": seed}
        fw = dropout.forward(x, **params)

        if mode == "train":
            np.random.seed(seed)
            mask = (np.random.rand(*x.shape) < p) / p
            fw_true = x * mask
        else:
            fw_true = deepcopy(x)

        np.testing.assert_array_equal(fw, fw_true)

        bw = dropout.backward(g_in, **params)

        if mode == "train":
            bw_true = g_in * mask
        else:
            bw_true = deepcopy(g_in)

        np.testing.assert_array_equal(bw, bw_true)
