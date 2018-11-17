import sys
import os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from ch5.TwoLayerNet import TwoLayerNet

if __name__ == '__main__':
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, one_hot_label=True)

    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    # 学習回数
    iters_num = 10000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    iter_per_epoch = max(train_size/batch_size, 1)

    # 学習
    for count in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # 誤差逆伝播法で勾配を求める
        grad = network.gradient(x_batch, t_batch)

        # 重み, バイアスを微小値更新する
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grad[key]

        # 損失を記録する
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        # 1エポックごとに認識精度を計算する
        if count % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)

            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)

            print("train_acc, test_acc | " + str(train_acc) + ", " + str(test_acc))

    plt.plot(np.arange(0, len(train_loss_list)), train_loss_list)
    plt.xlabel("count")
    plt.ylabel("loss")
    plt.show()

    plt.plot(np.arange(0, len(train_acc_list)), train_acc_list, '--')
    plt.plot(np.arange(0, len(test_acc_list)), test_acc_list)
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.show()
