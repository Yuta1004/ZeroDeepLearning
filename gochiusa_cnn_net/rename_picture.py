import os
import sys
sys.path.append(os.pardir)

if __name__ == '__main__':
    folder_name = ['chino', 'chiya', 'syaro', 'cocoa', 'rize', 'megu', 'maya']

    for folder in folder_name:
        folder_path = './anime_picture_face/' + folder + '/'
        files = os.listdir(folder_path)
        for idx, file in enumerate(files):
            os.rename(folder_path+file, folder_path+folder+'_'+str(idx)+'.jpg')
