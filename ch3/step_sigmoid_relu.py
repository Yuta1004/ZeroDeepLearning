import numpy as np
import matplotlib.pylab as plt


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ReLU(x):
    return np.maximum(0, x)


if __name__ == '__main__':
    x = np.arange(-5.0, 5.0, 0.1)
    step_y = step_function(x)
    sigmoid_y = sigmoid(x)
    ReLU_y = ReLU(x)
    plt.plot(x, step_y, label="step")
    plt.plot(x, sigmoid_y, linestyle="--", label="sigmoid")
    plt.plot(x, ReLU_y, linestyle="-.", label="ReLU")
    plt.legend()
    plt.ylim(-0.1, 1.1)
    plt.show()
