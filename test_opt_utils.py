import numpy as np
from opt_utils import GradClipperByValue


def test_grad_clipper_by_value():
    val = 5
    kwargs = {"val": val}

    a = np.random.normal(loc=0, scale=val * 1.2, size=(5, 10))
    b = np.random.normal(loc=0, scale=1.2, size=(5, 10))
    grads = [{"a": a, "b": b}]

    gc = GradClipperByValue(**kwargs)
    gc_grads = gc(grads)

    for idx, grads_dict in enumerate(grads):
        for grad_key, grad in grads_dict.items():
            low_mask = grad < -val
            high_mask = val < grad
            np.testing.assert_array_equal(low_mask, gc_grads[idx][grad_key] == -val)
            np.testing.assert_array_equal(high_mask, gc_grads[idx][grad_key] == val)
