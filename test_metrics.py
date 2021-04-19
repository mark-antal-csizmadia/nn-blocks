import numpy as np
from metrics import AccuracyMetrics


def test_accuracy_metrics(seed=np.random.randint(low=1, high=300)):
    low = 0
    high = np.random.randint(low=1, high=10)
    n = 100
    c = high
    size = (n, c)

    np.random.seed(seed)
    y = np.random.randint(low=low, high=high, size=(n,))

    np.random.seed(seed + 1)
    scores = np.random.normal(loc=0, scale=1, size=size)

    y_hat_true = np.argmax(scores, axis=1)
    acc_true = np.where(y_hat_true == y)[0].size / n

    accuracy_metrics = AccuracyMetrics()
    acc = accuracy_metrics.compute(y, scores)

    np.testing.assert_array_equal(acc, acc_true)
