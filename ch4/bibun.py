import numpy as np
import matplotlib.pylab as plt


def numerical_diff(f, x):
    h = 1e-4  # 0.0001
    return (f(x + h) - f(x - h)) / (2 * h)


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

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = x_tmp_val

    return grad


def tangent_line(base_cood, f, x):
    base_x = base_cood[0]
    base_y = base_cood[1]
    slope = numerical_diff(f, base_x)
    bias = base_y - slope * base_x

    print(slope)

    return x*slope+bias


def function_1(x):
    return 0.01*x**2 + 0.1*x


def function_2(x):
    return x ** 2


def function_3(x):
    return np.sum(x**2)


if __name__ == '__main__':
    x = np.arange(-20.0, 20.0, 0.1)
    # y = function_2(x)
    # plt.xlabel("x")
    # plt.ylabel("f(x)")
    #
    # base_x = -2
    # y2 = tangent_line((base_x, function_2(base_x)), function_2, x)
    #
    # base_x = 4
    # y3 = tangent_line((base_x, function_2(base_x)), function_2, x)
    #
    # plt.plot(x, y)
    # plt.plot(x, y2)
    # plt.plot(x, y3)
    # plt.show()

    print(numerical_gradient(function_3, np.array([3.0, 4.0])))
    print(numerical_gradient(function_3, np.array([0.0, 2.0])))
    print(numerical_gradient(function_3, np.array([3.0, 0.0])))
