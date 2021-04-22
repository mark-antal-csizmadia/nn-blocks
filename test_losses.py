import numpy as np
from copy import deepcopy
from activations import SoftmaxActivation, LinearActivation
from losses import CategoricalCrossEntropyLoss, CategoricalHingeLoss


def test_categoical_cross_entropy_loss():
    # from: https://cs231n.github.io/linear-classify/
    scores = np.array([[-2.85, 0.86, 0.28]])
    softmax_activation = SoftmaxActivation()
    scores = softmax_activation.forward(scores)

    y = np.array([2])

    categoical_cross_entropy_loss = CategoricalCrossEntropyLoss()

    loss = categoical_cross_entropy_loss.compute_loss(scores, y)
    np.testing.assert_almost_equal(loss, 1.04, decimal=2)

    loss_grad_true = deepcopy(y)
    loss_grad = categoical_cross_entropy_loss.grad()

    np.testing.assert_array_equal(loss_grad, loss_grad_true)

    # automated test
    batch_size = 5
    out_dim = 10
    size = (batch_size, out_dim)
    scores = np.random.normal(loc=0, scale=1, size=size)

    softmax_activation = SoftmaxActivation()
    scores = softmax_activation.forward(scores)

    y = np.random.randint(low=0, high=out_dim, size=(batch_size,))

    categoical_cross_entropy_loss = CategoricalCrossEntropyLoss()

    loss = categoical_cross_entropy_loss.compute_loss(scores, y)

    n = y.shape[0]

    # correct_logprobs.shape = (batch_size, )
    correct_logprobs = -np.log(scores[range(n), y])

    # compute the loss: average cross-entropy loss and regularization
    loss_true = np.sum(correct_logprobs) / n

    np.testing.assert_almost_equal(loss, loss_true)

    loss_grad_true = deepcopy(y)
    loss_grad = categoical_cross_entropy_loss.grad()

    np.testing.assert_array_equal(loss_grad, loss_grad_true)

    print("test_categoical_cross_entropy_loss passed")


def test_categorical_hinge_loss():
    # from: https://cs231n.github.io/linear-classify/
    scores = np.array([[-2.85, 0.86, 0.28]])
    linear_activation = LinearActivation()
    scores = linear_activation.forward(scores)

    y = np.array([2])

    categoical_hinge_loss = CategoricalHingeLoss()

    loss = categoical_hinge_loss.compute_loss(scores, y)
    np.testing.assert_almost_equal(loss, 1.58, decimal=2)

    loss_grad = categoical_hinge_loss.grad()

    n = y.shape[0]

    correct_class_scores = scores[range(n), y].reshape(n, 1)
    margin = np.maximum(0, scores - correct_class_scores + 1)
    margin[range(n), y] = 0  # do not consider correct class in loss
    loss = margin.sum() / n

    margin[margin > 0] = 1
    valid_margin_count = margin.sum(axis=1)
    # Subtract in correct class (-s_y)
    margin[range(n), y] -= valid_margin_count
    margin /= n

    loss_grad_true = deepcopy(margin)
    np.testing.assert_array_equal(loss_grad, loss_grad_true)

    # automated test
    batch_size = 5
    out_dim = 10
    size = (batch_size, out_dim)
    scores = np.random.normal(loc=0, scale=1, size=size)

    linear_activation = LinearActivation()
    scores = linear_activation.forward(scores)

    y = np.random.randint(low=0, high=out_dim, size=(batch_size,))

    categoical_hinge_loss = CategoricalHingeLoss()

    loss = categoical_hinge_loss.compute_loss(scores, y)

    n = y.shape[0]

    c = scores.shape[1]
    n = y.shape[0]

    correct_class_scores = scores[range(n), y].reshape(n, 1)
    margin = np.maximum(0, scores - correct_class_scores + 1)
    margin[range(n), y] = 0  # do not consider correct class in loss
    loss_true = margin.sum() / n

    margin[margin > 0] = 1
    valid_margin_count = margin.sum(axis=1)
    # Subtract in correct class (-s_y)
    margin[range(n), y] -= valid_margin_count
    margin /= n
    loss_grad_true = deepcopy(margin)

    np.testing.assert_almost_equal(loss, loss_true)

    loss_grad = categoical_hinge_loss.grad()

    np.testing.assert_array_equal(loss_grad, loss_grad_true)

    print("test_categorical_hinge_loss passed")
