import os
import sys
from PIL import Image
import numpy as np
import random

if __name__ == '__main__':
    folder_name = ['chino', 'chiya', 'cocoa', 'maya', 'megu', 'rize', 'syaro']

    csv = []

    for folder in folder_name:
        folder_path = './anime_picture_face/' + folder + '/'
        files = os.listdir(folder_path)
        for idx, file in enumerate(files):
            try:
                img_txt = str(folder_name.index(folder)) + ","
                img = Image.open(folder_path+file)
                img = img.resize((32, 32))
                img_array = np.asarray(img).reshape(-1)
                for _idx, number in enumerate(img_array):
                    if _idx != 0: img_txt += ","
                    img_txt += str(number)
                csv.append(img_txt)
            except OSError:
                pass

    random.shuffle(csv)
    with open("dataset.csv", "w") as f:
        for data in csv:
            f.write(data + "\n")
