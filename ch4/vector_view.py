import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    plt.figure()

    X = 0
    Y = 0
    U = -4
    V = -3.5

    plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale = 1)

    plt.xlim([-1, 2])
    plt.ylim([-1, 2])
    plt.grid()
    plt.draw()
    plt.show()
