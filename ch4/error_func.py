import numpy as np


def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y, t):
    delta = 1e-7
    print(t*np.log(y + delta))
    return -np.sum(t * np.log(y + delta))


def cross_entropy_error_2(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size


if __name__ == '__main__':
    t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

    y = [[0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0],
         [0.1, 0.1, 0.5, 0.0, 0.1, 0.1, 0.0, 0.0, 0.05, 0.05]]
    # print(mean_squared_error(np.array(y), np.array(t)), end="\n\n")
    print(cross_entropy_error_2(np.array(y), np.array(t)), end="\n\n")

    y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
    # print(mean_squared_error(np.array(y), np.array(t)))
    print(cross_entropy_error_2(np.array(y), np.array(t)))
