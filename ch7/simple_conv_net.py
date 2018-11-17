import os
import sys

sys.path.append(os.pardir)

from common.layers import *
from collections import OrderedDict
import pickle


class SimpleConvNet:
    """SimpleConvNet

    - Network
        Conv -> ReLU -> Pool -> Affine -> ReLu -> Affine -> Softmax

    - param
        input_dim: 入力データの次元(チャンネル, 高さ, 幅)
        conv_param: 畳み込み層のハイパーパラメータ(辞書型)
            * filter_num : フィルターの個数
            * filter_size: フィルターのサイズ
            * stride: ストライド
            * pad: パディング
        hidden_size: 隠れ層のニューロンの数
        output_size: 出力層のニューロンの数
        weight_init_std: 重みを初期化する際の標準偏差
    """

    def __init__(self, input_dim=(1, 28, 28), conv_param=None,
                 hidden_size=100, output_size=10, weight_init_std=0.01):
        if conv_param is None:
            conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1}

        # 畳み込み層のハイパーパラメータを取り出す
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        pad = conv_param['pad']
        stride = conv_param['stride']
        input_size = input_dim[1]

        # 畳み込み層, プーリング層の出力サイズを求める
        conv_output_size = int(1 + (input_size + 2 * pad - filter_size) / stride)
        pool_output_size = int(filter_num * (conv_output_size / 2) * (conv_output_size / 2))

        # 重み, バイアスの初期化
        self.params = {}
        self.params['W1'] = weight_init_std * \
                            np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = weight_init_std * \
                            np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * \
                            np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        # レイヤーを生成する
        self.layers = OrderedDict()  # 順番付き辞書型
        self.layers['Conv1'] = Convolution(self.params['W1'],
                                           self.params['b1'],
                                           stride, pad)
        self.layers['ReLu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'],
                                        self.params['b2'])
        self.layers['ReLu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'],
                                        self.params['b3'])

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)

        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        y = np.argmax(y, axis=1)

        return np.sum(y == t) / t.shape[0]

    def gradient(self, x, t):
        # 順伝播
        loss = self.loss(x, t)

        # 逆伝播
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())  # layers(辞書)をリスト型に変換
        layers.reverse()  # 逆順にする

        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Conv1'].dW
        grads['b1'] = self.layers['Conv1'].db
        grads['W2'] = self.layers['Affine1'].dW
        grads['b2'] = self.layers['Affine1'].db
        grads['W3'] = self.layers['Affine2'].dW
        grads['b3'] = self.layers['Affine2'].db

        return grads

    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, key in enumerate(['Conv1', 'Affine1', 'Affine2']):
            self.layers[key].W = self.params['W' + str(i+1)]
            self.layers[key].b = self.params['b' + str(i+1)]
