from copy import deepcopy
import numpy as np


class GradClipper():
    def __init__(self, repr_str):
        self.repr_str = repr_str

    def apply(self, grads_val):
        raise NotImplementedError

    def __call__(self, grads):
        # grads is a list of dicts, where each list is for a layer
        # and a dict is for the params' grads in that layer
        clipped_grads = deepcopy(grads)

        for idx in range(len(grads)):
            grad_dict = deepcopy(grads[idx])

            for g in grad_dict:
                clipped_grads[idx][g] = self.apply(grad_dict[g])

        return deepcopy(clipped_grads)

    def __repr__(self, ):
        return self.repr_str


class GradClipperByValue(GradClipper):
    def __init__(self, **kwargs):
        repr_str = "clipper by value"
        super().__init__(repr_str)
        self.val = kwargs["val"]

    def apply(self, grad_val):
        return np.maximum(np.minimum(grad_val, self.val), -self.val)


class GradClipperByNothing(GradClipper):
    def __init__(self, ):
        repr_str = "clipper who does nothing"
        super().__init__(repr_str)

    def apply(self, grad_val):
        return deepcopy(grad_val)

