from dataset.mnist import load_mnist
import numpy as np
from ch7.simple_conv_net import SimpleConvNet


if __name__ == '__main__':
    network = SimpleConvNet()
    network.load_params()

    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

    for idx in range(10):
        x = x_test[idx].reshape(1, 1, 28, 28)
        anc = network.predict(x)
        print(np.argmax(anc), t_test[idx])
