import numpy as np
from copy import deepcopy
from lr_schedules import LRConstantSchedule, LRExponentialDecaySchedule, LRCyclingSchedule


def test_lr_constant_schedule():
    lr_initial = 0.1
    lr = deepcopy(lr_initial)
    lr_constant_schedule = LRConstantSchedule(lr_initial)

    iter_n = 100

    for i in range(iter_n):
        lr_constant_schedule.apply_schedule()
        lr = lr_constant_schedule.get_lr()
        np.testing.assert_array_equal(lr, lr_initial)

    print("test_lr_constant_schedule passed")


def test_lr_exponential_decay_schedule():
    lr_initial = 0.1
    lr = deepcopy(lr_initial)
    decay_steps = 10
    decay_rate = 0.9
    lr_exponential_decay_schedule = LRExponentialDecaySchedule(lr_initial, decay_steps, decay_rate)

    iter_n = 100

    for i in range(iter_n):
        lr_exponential_decay_schedule.apply_schedule()
        lr = lr_exponential_decay_schedule.get_lr()
        lr_true = lr_initial * decay_rate ** (i / decay_steps)
        np.testing.assert_array_equal(lr, lr_true)

    print("test_lr_exponential_decay_schedule passed")


def test_lr_cycling_schedule():
    lr_initial = 0.1
    lr = deepcopy(lr_initial)
    lr_max = 0.5
    step_size = 10

    lr_list = []
    lr_true_list = []

    lr_cycling_schedule = LRCyclingSchedule(lr_initial, lr_max, step_size)

    iter_n = 101

    for i in range(iter_n):
        lr_cycling_schedule.apply_schedule()
        lr = lr_cycling_schedule.get_lr()
        cycle = np.floor(1 + i / (2 * step_size))
        x = np.abs(i / step_size - 2 * cycle + 1)
        lr_true = lr_initial + (lr_max - lr_initial) * np.maximum(0, (1 - x))

        np.testing.assert_array_equal(lr, lr_true)

        lr_list.append(lr)
        lr_true_list.append(lr_true)

    print("test_lr_cycling_schedule passed")

    # plt.plot(lr_list)
    # plt.plot(lr_true_list)
