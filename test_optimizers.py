import numpy as np
from lr_schedules import LRConstantSchedule
from optimizers import SGDOptimizer, AdaGradOptimizer
from opt_utils import GradClipperByNothing


def test_sgd_optimizer():
    # n_layer = len(test_specs)
    # each dict is either the param dict or the grad dict of that layer
    trainable_weights = [
        {
            "w": np.random.normal(loc=0, scale=1, size=(10, 5)),
            "b": np.random.normal(loc=0, scale=1, size=(1, 5))
        },
        {
            "w": np.random.normal(loc=0, scale=1, size=(5, 3)),
            "b": np.random.normal(loc=0, scale=1, size=(1, 3))
        },
        {
            "w": np.random.normal(loc=0, scale=1, size=(3, 2)),
            "b": np.random.normal(loc=0, scale=1, size=(1, 2))
        }
    ]

    grads = [
        {
            "dw": np.random.normal(loc=0, scale=1, size=(10, 5)),
            "db": np.random.normal(loc=0, scale=1, size=(1, 5))
        },
        {
            "dw": np.random.normal(loc=0, scale=1, size=(5, 3)),
            "db": np.random.normal(loc=0, scale=1, size=(1, 3))
        },
        {
            "dw": np.random.normal(loc=0, scale=1, size=(3, 2)),
            "db": np.random.normal(loc=0, scale=1, size=(1, 2))
        }
    ]

    lr_initial = 0.6
    lr_schedule = LRConstantSchedule(lr_initial)

    sgd_optimizer = SGDOptimizer(lr_schedule=lr_schedule, grad_clipper=GradClipperByNothing())
    updated_trainable_weights = sgd_optimizer.apply_grads(trainable_weights, grads)

    for layer_weights, layer_grads, updated_weights in \
            zip(trainable_weights, grads, updated_trainable_weights):

        for param_str in layer_weights.keys():
            true_updated = layer_weights[param_str] - lr_schedule.get_lr() * layer_grads["d" + param_str]
            updated = updated_weights[param_str]
            np.testing.assert_array_equal(true_updated, updated)

    print("test_sgd_optimizer passed")
