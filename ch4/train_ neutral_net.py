import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from ch4.two_layer_net import TwoLayerNet

if __name__ == '__main__':
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, one_hot_label=True)

    # ハイパーパラメータ
    iters_num = 10000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    # 1エポックあたりの繰り返し数
    iter_per_epoch = max(train_size / batch_size, 1)

    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    for count in range(iters_num):
        # ミニバッチの取得
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # 勾配の計算
        # grad = network.numerical_gradient(x_batch, t_batch)
        grad = network.gradient(x_batch, t_batch)

        # パラメータの更新
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grad[key]

        # 学習経過の記録
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        # 1エポック毎に認識精度を計算
        if count % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

    # 学習経過をプロットして表示
    plt.plot(np.arange(0, len(train_loss_list)), train_loss_list)
    plt.show()

    # 1エポック毎に計測し認識精度(train, test)を表示
    plt.plot(np.arange(0, len(train_acc_list)), train_acc_list, "--")
    plt.plot(np.arange(0, len(test_acc_list)), test_acc_list)
    plt.show()
