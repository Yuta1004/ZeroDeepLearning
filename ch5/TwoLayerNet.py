import os
import sys

sys.path.append(os.pardir)
import numpy as np
from ch5.layers import *  # 自分で書いたレイヤーを使う!
from common.gradient import numerical_gradient  # 数値微分を用いた勾配計算
from collections import OrderedDict


class TwoLayerNet:
    """2層のニューラルネットワーク

    input_size : 入力データの数
    hidden_size : 隠れ僧のニューロン数
    output_size : 出力データの数

    """
    def __init__(self, input_size, hidden_size, output_size):
        # 重み, バイアスの初期化
        # 重みの初期値に「Heの初期値」を使用する( √(2/n) * ガウス分布 )
        # n -> 前層のニューロンの数

        self.params = {}
        self.params['W1'] = np.sqrt(2 / input_size) * \
                            np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = np.sqrt(2 / hidden_size) * \
                            np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # レイヤの生成
        self.layers = OrderedDict()  # 順番付き辞書型
        self.layers['Affine1'] = \
            Affine(self.params['W1'], self.params['b1'])
        self.layers['ReLu1'] = ReLU()
        self.layers['Affine2'] = \
            Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        # レイヤに値を流していく(順伝播)
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        # 教師データがone_hot表記だったとき
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # 数値微分
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    # 誤差逆伝播法
    def gradient(self, x, t):
        # forward (順伝播)
        self.loss(x, t)

        # backward (逆伝播で勾配を求めて返す)
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads
