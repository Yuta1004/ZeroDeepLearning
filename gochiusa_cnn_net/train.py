import os
import sys

sys.path.append(os.pardir)
from gochiusa_cnn_net.load_dataset import load_dataset
from gochiusa_cnn_net.CNN_Network import CNN_Network
from common.optimizer import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    (x_train, t_train), (x_test, t_test) = \
       load_dataset(normalize=True, one_hot_label=True, flatten=False)

    network = CNN_Network()

    # エポック数など指定
    lr = 0.01
    epochs = 1500
    batch_size = 65
    train_size = x_train.shape[0]
    iter_per_epoch = max(train_size / batch_size, 1)
    max_iters = int(epochs * iter_per_epoch)

    # optimzer
    optimizer_param = {'lr': 0.01}
    optimizer = "SGD"
    optimizer_class_dict = {'sgd': SGD, 'momentum': Momentum, 'nesterov': Nesterov,
                            'adagrad': AdaGrad, 'rmsprpo': RMSprop, 'adam': Adam}
    optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    # 学習
    for count in range(max_iters):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # print(x_batch.shape, t_batch.shape)

        # 勾配を求める
        grads = network.gradient(x_batch, t_batch)
        optimizer.update(network.params, grads)

        # 損失を求める
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        if count % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print("===========EPOCH ("+str(count/iter_per_epoch)+"/"+str(epochs)+") " +
                  "| train_acc: " + str(train_acc) + ", test_acc: " + str(test_acc))

        print(str(count % iter_per_epoch) + "| loss " + str(loss))

    network.save_params()

    # グラフ表示
    plt.plot(np.array(0, len(train_acc_list)), train_acc_list)
    plt.plot(np.array(0, len(test_acc_list)), train_acc_list)
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.show()
