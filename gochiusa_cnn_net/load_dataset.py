import numpy as np
from PIL import Image


def load_dataset(one_hot_label=True, normalize=True, flatten=False):
    with open("/Users/nakagamiyuta/Desktop/Python/zero_deep_learning/gochiusa_cnn_net/dataset.csv", 'rb') as f:
        f = f.read()
        f = f.split()

        label_list = []
        data_list = []

        for line in f:
            line = str(line)
            line = str(line[2:]).split(",")
            label = 0
            img = []
            for idx, number in enumerate(line):
                number = number.strip("'")
                number = int(number)
                if idx == 0:
                    label = number
                else:
                    img.append(number)
            data_list.append(img)
            label_list.append(label)

        # list -> numpyArray
        for idx in range(len(data_list)):
            data_list[idx] = np.array(data_list[idx]).reshape(32, 32, 3).transpose(2, 0, 1)

        if one_hot_label:
            for idx in range(len(label_list)):
                label = np.zeros(7)
                label[label_list[idx]] = 1
                label_list[idx] = label

        if flatten:
            for idx in range(len(data_list)):
                data_list[idx] = np.array(data_list[idx]).flatten()

        if normalize:
            for idx in range(len(data_list)):
                data_list[idx] = data_list[idx] / 255

        x_train = np.array(data_list[:650])
        t_train = np.array(label_list[:650])
        x_test = np.array(data_list[650:])
        t_test = np.array(label_list[650:])

        return (x_train, t_train), (x_test, t_test)


if __name__ == '__main__':
    (x_train, t_train), (x_test, t_test) = load_dataset(normalize=True, one_hot_label=True, flatten=False)
