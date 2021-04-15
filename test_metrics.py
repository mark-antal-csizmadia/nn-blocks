import numpy as np
from metrics import AccuracyMetrics


def test_accuracy_metrics(seed=np.random.randint(low=1, high=300)):
    low = 1
    high = 100
    n = 100

    np.random.seed(seed)
    y = np.random.randint(low=low, high=high, size=(n,))

    np.random.seed(seed + 1)
    y_hat = np.random.randint(low=low, high=high, size=(n,))

    accuracy_metrics = AccuracyMetrics()

    acc = accuracy_metrics.get_metrics(y, y_hat)
    acc_true = np.where(y_hat == y)[0].size / n

    np.testing.assert_array_equal(acc, acc_true)

