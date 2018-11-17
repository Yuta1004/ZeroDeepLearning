import os
import sys

sys.path.append(os.pardir)
from gochiusa_cnn_net.CNN_Network import CNN_Network
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np


class MainWindow:
    def __init__(self):
        self.root = Tk()
        self.root.title("ごちうさキャラ判定")
        self.root.geometry('800x500')

        self.canvas = Canvas(self.root, width=800, height=600)
        self.canvas.place(x=0, y=0)

        self.image = None
        self.ranking = [""] * 7

        self.network = CNN_Network()
        self.network.load_params()
        self.chara_name = ['chino', 'chiya', 'cocoa', 'maya', 'megu', 'rize', 'syaro']
        self.processing = False

        self.dir = "/Users/nakagamiyuta/Desktop/Programming/Python/zero_deep_learning/gochiusa_cnn_net/anime_picture_face/all/"

        self.select_idx = 0
        self.max_idx = 0

        Button(self.root, text=" 読み込みフォルダ指定 ", command=self.chooseDirectory)\
            .place(x=500, y=100, anchor="c")
        Button(self.root, text=" <- ", command=self.leftSelect) \
            .place(x=430, y=130)
        Button(self.root, text=" -> ", command=self.rightSelect) \
            .place(x=480, y=130)

        self.Classification()
        self.update()

    def update(self):
        self.canvas.delete("all")
        if not self.image is None:
            self.canvas.create_image(20, 40, image=self.image, anchor=NW)
        # dir = self.dir.replace("nakagamiyuta", "************")
        self.canvas.create_text(20, 20, text=self.dir,
                                justify=LEFT,
                                anchor="w",
                                font=("Hirgino Maru Gothic Pro", 15))
        for idx, rank in enumerate(self.ranking):
            if idx == 0:
                self.canvas.create_text(430, 200+idx*30, text=rank,
                                        justify=LEFT,
                                        anchor="w",
                                        fill='red',
                                        font=("Hiragino Maru Gothic Pro", 18))
            else:
                self.canvas.create_text(430, 200 + idx * 30, text=rank,
                                        justify=LEFT,
                                        anchor="w",
                                        font=("Hiragino Maru Gothic Pro", 18))
            self.canvas.update()

    def Classification(self, idx=0):
        if self.processing:  # 処理中ならreturn
            return

        self.processing = True
        files = os.listdir(self.dir)
        jpg_files = []
        for file in files:
            if ".jpg" in file:
                jpg_files.append(file)
        self.max_idx = len(jpg_files) - 1

        if 0 <= idx < len(jpg_files):
            # 画像読み込み
            img_path = self.dir+jpg_files[idx]
            img = Image.open(img_path)

            # 推論
            pre_img = img.resize((32, 32))
            img_array = np.asarray(pre_img)
            img_array = img_array.reshape(32, 32, 3).transpose(2, 0, 1)
            img_array = img_array / 255
            img_array = np.array([img_array])
            pre = self.network.predict(img_array)
            pre_softmax = self.softmax(pre)[0]
            rank_idx = np.argsort(pre_softmax)[::-1]

            # 結果表示
            for idx, rank in enumerate(rank_idx):
                self.ranking[idx] = self.chara_name[rank] + " | " + str(pre_softmax[rank]*100) + "%"

            # 画像表示
            view_img = img.resize((400, 400))
            self.image = ImageTk.PhotoImage(view_img)

        self.update()
        self.processing = False

    def softmax(self, x):
        x = np.exp(x)
        return x / np.sum(x)

    def chooseDirectory(self):
        dir = filedialog.askdirectory()
        if dir != "":
            self.dir = dir + "/"
        self.Classification()
        self.update()

    def leftSelect(self):
        if self.processing:
            return
        if self.select_idx > 0:
            self.select_idx -= 1
        self.Classification(self.select_idx)

    def rightSelect(self):
        if self.processing:
            return
        if self.select_idx < self.max_idx:
            self.select_idx += 1
        self.Classification(self.select_idx)


if __name__ == '__main__':
    window = MainWindow()
    window.root.mainloop()
