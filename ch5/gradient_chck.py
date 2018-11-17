import sys
import os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from ch5.TwoLayerNet import TwoLayerNet

if __name__ == '__main__':
    # データの読み込み
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, one_hot_label=True)

    # ネットワーク初期化
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    x_batch = x_train[:3]
    t_batch = t_train[:3]

    grad_numerical = network.numerical_gradient(x_batch, t_batch)
    grad_backprop = network.gradient(x_batch, t_batch)

    # 各重みの絶対誤差の平均を求める
    # np.abs = 絶対値
    for key in ('W1', 'b1', 'W2', 'b2'):
        diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
        print(key + str(diff))
