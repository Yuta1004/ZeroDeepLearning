import os
import sys

sys.path.append(os.pardir)

import cv2
from gochiusa_cnn_net.CNN_Network import CNN_Network
import numpy as np
import copy
import tqdm


def softmax(x):
    x = np.exp(x)
    return x / np.sum(x)


if __name__ == '__main__':
    cascade_path = "lbpcascade_animeface.xml"

    # 動画のパス設定
    input_video = "./data/gochiusa_4.mp4"
    output_video = "./data/gochiusa_4_face.mp4"

    # 動画情報取得
    video = cv2.VideoCapture(input_video)
    all_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

    # 書き込み動画設定
    fourcc = int(cv2.VideoWriter_fourcc("m", "p", "4", "v"))
    output = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    network = CNN_Network()
    network.load_params()

    chara_name = ['chino', 'chiya', 'cocoa', 'maya', 'megu', 'rize', 'syaro']
    COLOR = [(255, 215, 139), (69, 228, 10), (127, 113, 227), (234, 77, 0),
             (109, 25, 255), (222, 31, 179), (0, 165, 203)]

    # 進捗バー設定
    bar = tqdm.tqdm(total=all_frame)

    # 分類器設定
    cascade = cv2.CascadeClassifier(cascade_path)

    ret, frame = video.read()
    while ret:  # 1フレームごとに処理する
        ret, frame = video.read()
        if not ret:
            break

        # グレースケール変換
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 顔の座標を分類器によって抽出
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))

        tmp_frame = copy.deepcopy(frame)
        for (x, y, w, h) in faces:
            face = tmp_frame[y:y + h, x:x + w]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (32, 32)).transpose(2, 0, 1)
            face = face / 255

            # 推論
            pre = network.predict(np.array([face]))
            pre_soft = softmax(pre)[0]
            max_idx = int(np.argmax(pre))

            if pre_soft[max_idx] >= 0.95:
                cv2.putText(frame, chara_name[max_idx] + " : " + str(pre_soft[max_idx].round(3) * 100) + "%",
                            (x, y - 10), cv2.FONT_HERSHEY_DUPLEX,
                            int(w / 150) if int(w / 150) >= 1 else 1,
                            COLOR[max_idx], 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR[max_idx], 2)

        output.write(frame)
        bar.update()

        if cv2.waitKey(1) == ord('q'):
            break

    bar.close()
    video.release()
    cv2.destroyAllWindows()
