# -*- coding: utf-8 -*-
import numpy as np


def AND(x1, x2):
    x = np.array([x1, x2])  # 入力
    w = np.array([0.5, 0.5])  # 重み
    b = -0.7  # バイアス
    tmp = np.sum(x * w) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(x * w) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.4
    tmp = np.sum(x * w) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y


if __name__ == '__main__':
    print("AND")
    print(AND(0, 0), AND(1, 0), AND(0, 1), AND(1, 1))
    print("")

    print("NAND")
    print(NAND(0, 0), NAND(1, 0), NAND(0, 1), NAND(1, 1))
    print("")

    print("OR")
    print(OR(0, 0), OR(1, 0), OR(0, 1), OR(1, 1))
    print("")

    print("XOR")
    print(XOR(0, 0), XOR(1, 0), XOR(0, 1), XOR(1, 1))
    print("")
