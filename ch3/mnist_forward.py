import os
import sys

sys.path.append(os.pardir)
from dataset.mnist import load_mnist
import numpy as np
import pickle


def sigmoid(value):
    return 1 / (1 + np.exp(-value))


def softmax(value):
    max_value = np.max(value)
    exp_value = np.exp(value - max_value)
    exp_sum_value = np.sum(exp_value)
    y = exp_value / exp_sum_value

    return y


def get_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(flatten=True, normalize=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


if __name__ == '__main__':
    x, t = get_data()
    network = init_network()

    batch_size = 1000  # バッチの数
    accuracy_cnt = 0

    # バッチ処理有
    for i in range(0, len(x), batch_size):
        x_batch = x[i:i+batch_size]
        y_batch = predict(network, x_batch)
        p = np.argmax(y_batch, axis=1)
        accuracy_cnt += np.sum(p == t[i:i+batch_size])

    # バッチ処理無
    # for count in range(len(x)):
    #     y = predict(network, x[count])
    #     p = np.argmax(y)
    #     if p == t[count]:
    #         accuracy_cnt += 1

    print("Accuracy: " + str(float(accuracy_cnt) / len(x)))
