from gochiusa_cnn_net.CNN_Network import CNN_Network
from gochiusa_cnn_net.load_dataset import load_dataset
from PIL import Image
import numpy as np
import time


def softmax(x):
    x = np.exp(x)
    return x / np.sum(x)


if __name__ == '__main__':
    (x_train, t_train), (x_test, t_test) = \
        load_dataset(one_hot_label=True, normalize=True, flatten=False)

    network = CNN_Network()
    network.load_params()

    name = ['chino', 'chiya', 'cocoa', 'maya', 'megu', 'rize', 'syaro']

    for i in range(1):
        x = np.array([x_test[i]])
        print(x)
        pre = network.predict(x)
        print(pre)
        prob = softmax(pre)[0]
        print(name[int(np.argmax(pre))], prob[np.argmax(prob)])

        img = Image.fromarray(np.uint8(x_test[i].transpose(1, 2, 0) * 255))
        img.show()

        time.sleep(1)
        img.close()
