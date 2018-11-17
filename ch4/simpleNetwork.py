import sys
import os

sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss


if __name__ == '__main__':
    net = simpleNet()
    print(net.W)  # 重み

    x = np.array([0.6, 0.9])  # 入力データ
    p = net.predict(x)  # 行列計算(入力 * 重み)
    print(p)  # 計算結果
    print(np.argmax(p))  # 配列の中で最大値のインデックス

    t = np.array([0, 0, 1])  # 正解ラベル(one_hot)
    print(net.loss(x, t))  # 損失関数

    func = lambda W: net.loss(x, t)  # 勾配を求める関数
    dW = numerical_gradient(func, net.W)  # 勾配を計算(微分)
    print(dW)  # 勾配出力
