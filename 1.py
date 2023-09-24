import os
import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
print("Clasifier imported!")

face_list = []
class_list = []

train_path = './Dataset/'
players = os.listdir(train_path)
# print(players)

for i, name in enumerate(players):
    player_path = train_path + name + '/'
    # print(player_path)
    for stockimg in os.listdir(player_path):
        path_per_pic = player_path + stockimg
        # print(path_per_pic)
        img = cv2.imread(path_per_pic, 0)

        detected_face = face_cascade.detectMultiScale(img, scaleFactor = 1.2, minNeighbors = 5)

        if len(detected_face) < 1:
            continue

        for face_rect in detected_face:
            x,y,h,w = face_rect
            face_img = img[y:y+h, x:x+w]

            face_list.append(face_img)
            class_list.append(i)

# print(face_list)
# print(cv2.__version__)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(face_list, np.array(class_list))

