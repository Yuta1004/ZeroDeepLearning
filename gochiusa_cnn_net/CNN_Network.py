import os
import sys
sys.path.append(os.pardir)

from common.layers import *
from collections import OrderedDict
import pickle


class CNN_Network:

    """CNN_Network (from VGG)
    - Network
        Conv -> ReLU -> Conv -> ReLU -> Pool ->
        Conv -> ReLU -> Conv -> ReLU -> Pool ->
        Conv -> ReLU -> Conv -> ReLU -> Pool ->
        Affine -> ReLU -> DropOut -> Affine -> DropOut -> Softmax

    - param
        *input_dim: 入力画像の次元 (チャンネル, 高さ, 幅)
        *conv_param_1~6: 畳み込み層のハイパーパラメータ
            @filter_num: フィルターの数
            @filter_size: フィルターのサイズ
            @stride: ストライド
            @pad: パディング
        *hidden_size: 隠れ層のニューロン数
        *output_size: 出力層のニューロン数
    """

    def __init__(self, input_dim=(3, 32, 32),
                 conv_param_1=None, conv_param_2=None, conv_param_3=None,
                 conv_param_4=None, conv_param_5=None, conv_param_6=None,
                 hidden_size=100, output_size=7):
        # conv_paramの設定
        if conv_param_1 is None:
            conv_param_1 = {'filter_num': 16, 'filter_size': 3, 'stride': 1, 'pad': 1}
        if conv_param_2 is None:
            conv_param_2 = {'filter_num': 16, 'filter_size': 3, 'stride': 1, 'pad': 1}
        if conv_param_3 is None:
            conv_param_3 = {'filter_num': 32, 'filter_size': 3, 'stride': 1, 'pad': 1}
        if conv_param_4 is None:
            conv_param_4 = {'filter_num': 32, 'filter_size': 3, 'stride': 1, 'pad': 2}
        if conv_param_5 is None:
            conv_param_5 = {'filter_num': 64, 'filter_size': 3, 'stride': 1, 'pad': 1}
        if conv_param_6 is None:
            conv_param_6 = {'filter_num': 64, 'filter_size': 3, 'stride': 1, 'pad': 1}

        # 重みの初期化
        # 各ニューロンが前層のニューロンとどれくらい繋がりがあるか
        pre_node_nums = np.array([1*3*3, 16*3*3, 16*3*3, 32*3*3, 32*3*3, 64*3*3, 64*4*4, hidden_size])
        he_init_std = np.sqrt(2.0 / pre_node_nums)

        self.params = {}
        pre_channel_num = input_dim[0]
        for idx, conv_param in enumerate([conv_param_1, conv_param_2, conv_param_3,
                                         conv_param_4, conv_param_5, conv_param_6]):
            self.params['W'+str(idx+1)] = he_init_std[idx] * \
                                          np.random.randn(conv_param['filter_num'], pre_channel_num,
                                                          conv_param['filter_size'], conv_param['filter_size'])
            self.params['b'+str(idx+1)] = np.zeros(conv_param['filter_num'])
            pre_channel_num = conv_param['filter_num']
        self.params['W7'] = he_init_std[6] * \
                            np.random.randn(64*4*4, hidden_size)
        self.params['b7'] = np.zeros(hidden_size)
        self.params['W8'] = he_init_std[7] * \
                            np.random.randn(hidden_size, output_size)
        self.params['b8'] = np.zeros(output_size)

        # レイヤーの生成
        self.layers = OrderedDict()

        # group1
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                           conv_param_1['stride'], conv_param_1['pad'])
        self.layers['ReLU1'] = Relu()
        self.layers['Conv2'] = Convolution(self.params['W2'], self.params['b2'],
                                           conv_param_2['stride'], conv_param_2['pad'])
        self.layers['ReLU2'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)

        # group2
        self.layers['Conv3'] = Convolution(self.params['W3'], self.params['b3'],
                                           conv_param_3['stride'], conv_param_3['pad'])
        self.layers['ReLU3'] = Relu()
        self.layers['Conv4'] = Convolution(self.params['W4'], self.params['b4'],
                                           conv_param_4['stride'], conv_param_4['pad'])
        self.layers['ReLU4'] = Relu()
        self.layers['Pool2'] = Pooling(pool_h=2, pool_w=2, stride=2)

        # group3
        self.layers['Conv5'] = Convolution(self.params['W5'], self.params['b5'],
                                           conv_param_5['stride'], conv_param_5['pad'])
        self.layers['ReLU5'] = Relu()
        self.layers['Conv6'] = Convolution(self.params['W6'], self.params['b6'],
                                           conv_param_6['stride'], conv_param_6['pad'])
        self.layers['ReLU6'] = Relu()
        self.layers['Pool3'] = Pooling(pool_h=2, pool_w=2, stride=2)

        # group4(全結合 -> Softmax)
        self.layers['Affine1'] = Affine(self.params['W7'], self.params['b7'])
        self.layers['ReLU6'] = Relu()
        self.layers['DropOut1'] = Dropout(dropout_ratio=0.5)
        self.layers['Affine2'] = Affine(self.params['W8'], self.params['b8'])
        self.layers['DropOut2'] = Dropout(dropout_ratio=0.5)

        self.last_layer = SoftmaxWithLoss()

    # 推論
    def predict(self, x, train_flag=False):
        for layer in self.layers.values():
            if isinstance(layer, Dropout):
                x = layer.forward(x, train_flag)
            else:
                x = layer.forward(x)

        return x

    # 損失
    def loss(self, x, t):
        # 順伝播
        y = self.predict(x, True)

        # 損失を求める
        loss = self.last_layer.forward(y, t)

        return loss

    # 精度
    def accuracy(self, x, t):
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        y = self.predict(x)
        y = np.argmax(y, axis=1)

        return np.sum(y == t) / t.shape[0]

    # 勾配を求める
    def gradient(self, x, t):
        # 順伝播
        self.loss(x, t)

        # 逆伝播
        dout = 1
        dout = self.last_layer.backward(dout)

        back_layers = list(self.layers.values())
        back_layers.reverse()

        for layer in back_layers:
            dout = layer.backward(dout)

        grads = {}
        for idx in range(1, 7):
            grads['W'+str(idx)] = self.layers['Conv'+str(idx)].dW
            grads['b'+str(idx)] = self.layers['Conv'+str(idx)].db
        grads['W7'] = self.layers['Affine1'].dW
        grads['b7'] = self.layers['Affine1'].db
        grads['W8'] = self.layers['Affine2'].dW
        grads['b8'] = self.layers['Affine2'].db

        return grads

    # 重み, バイアスを保存
    def save_params(self, file_name="/Users/nakagamiyuta/Desktop/Programming/Python/zero_deep_learning/gochiusa_cnn_net/params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    # 重み, バイアスを読み込む
    def load_params(self, file_name="/Users/nakagamiyuta/Desktop/Programming/Python/zero_deep_learning/gochiusa_cnn_net/params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for idx in range(1, 7):
            self.layers['Conv'+str(idx)].W = self.params['W'+str(idx)]
            self.layers['Conv'+str(idx)].b = self.params['b'+str(idx)]
        self.layers['Affine1'].W = self.params['W7']
        self.layers['Affine1'].b = self.params['b7']
        self.layers['Affine2'].W = self.params['W8']
        self.layers['Affine2'].b = self.params['b8']
