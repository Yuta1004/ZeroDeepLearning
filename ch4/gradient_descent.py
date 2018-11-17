import numpy as np
import matplotlib.pyplot as plt


def function_1(x):
    return x[0] ** 2 + x[1] ** 2


def function_2(x):
    return x ** 2


def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    for idx in range(x.size):
        x_tmp_val = x[idx]

        # f(x+h)
        x[idx] = x_tmp_val + h
        fxh1 = f(x)

        # f(x-h)
        x[idx] = x_tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = x_tmp_val

    return grad


def gradient_descent(f, init_x, lr=0.1, step_num=100):
    x = init_x
    x_history = []

    for _ in range(step_num):
        x_history.append(x.copy())
        grad = numerical_gradient(f, x)
        # print(x, lr, grad)
        x -= lr * grad

    return x, np.array(x_history)


if __name__ == '__main__':
    # x = np.arange(-10, 10, 0.1)
    # y = function_1(np.array([x, x]))

    init_x = np.array([-3.0, 4.0])
    end_x, x_history = gradient_descent(function_1, init_x)

    print(end_x)

    plt.plot(x_history[:, 0], x_history[:, 1], 'o')
    plt.xlim(-3.5, 3.5)
    plt.ylim(-4.5, 4.5)
    plt.xlabel("X0")
    plt.ylabel("X1")
    plt.show()

    # init_x = np.array([10.0])
    # end_x, x_history = gradient_descent(function_2, init_x)
    #
    # plt.plot(x, y)
    # plt.plot(x_history, function_2(x_history), 'o')
    # plt.show()
