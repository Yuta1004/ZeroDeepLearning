import sys
import os

sys.path.append(os.pardir)
from common.util import im2col
import numpy as np


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2*self.pad - FH) / self.stride)  # 出力サイズ(height)
        out_w = int(1 + (W + 2*self.pad - FW) / self.stride)  # 出力サイズ (width)

        col = im2col(x, FH, FW, stride=self.stride, pad=self.pad)  # 入力を二次元配列に(展開)
        col_W = self.W.reshape(FN, -1).T  # フィルターを二次元配列に(展開)
        out = np.dot(col, col_W) + self.b  # 行列演算

        """ 
        reshapeの引数に(-1)を与えると要素数の辻褄が合うように自動で整形してくれる
         * (10, 3, 5, 5) -> reshape(10, -1) = (10, 75)
            -> 3 * 5 * 5 = 75
        """

        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        return out


class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2*self.pool_h) / self.stride)  # 出力サイズ(height)
        out_w = int(1 + (W + 2 * self.pool_w) / self.stride)  # 出力サイズ(width)

        # 入力 -> 二次元配列 -> reshapeでプーリング適用範囲を展開する(図解 -> ゼロからP.228)
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        # 1次元軸での最大値を求める
        out = np.argmax(col, axis=1)

        # 最大値を出力サイズでreshapeして、(データ数, チャンネル, height, width)になるように
        #  transposeする(軸を入れ替える)
        out = out.reshape(N, out_h, out_h, C).transpose(0, 3, 1, 2)

        return out


if __name__ == '__main__':
    x1 = np.random.rand(1, 3, 7, 7)
    col1 = im2col(x1, 5, 5, stride=1, pad=0)
    print(col1.shape)
    print(col1[0])

    x2 = np.random.rand(10, 3, 7, 7)
    col2 = im2col(x2, 5, 5, stride=1, pad=0)
    print(col2.shape)
