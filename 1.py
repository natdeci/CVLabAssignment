import os
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
print("Clasifier imported!")

all_data = []
all_class = []
face_list = []
class_list = []

train_path = './Dataset/'
players = os.listdir(train_path)

for i, name in enumerate(players):
    player_path = train_path + name + '/'
    for stockimg in os.listdir(player_path):
        path_per_pic = player_path + stockimg
        imgstore = cv2.imread(path_per_pic)
        all_data.append(imgstore)
        all_class.append(i)

X_train, X_test, y_train, y_test = train_test_split(all_data, all_class, test_size=0.3, stratify=all_class)

for face_data, name_data in list(zip(X_train,y_train)):
    face_data = cv2.cvtColor(face_data, cv2.COLOR_BGR2GRAY)
    detected_face = face_cascade.detectMultiScale(face_data, scaleFactor = 1.2, minNeighbors = 5)
    if len(detected_face) < 1:
        continue
    
    for face_rect in detected_face:
        x,y,h,w = face_rect
        face_img = face_data[y:y+h, x:x+w]

        face_list.append(face_img)
        class_list.append(name_data)

print(face_list)
print(class_list)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(face_list, np.array(class_list))

# cv2.imshow('anjay', cv2.cvtColor(X_train[0], cv2.COLOR_BGR2GRAY))
# cv2.waitKey(0)

for tester in X_test:
    img_gray = cv2.cvtColor(tester, cv2.COLOR_BGR2GRAY)
    detected_face = face_cascade.detectMultiScale(img_gray, scaleFactor = 1.2, minNeighbors = 5)
    
    if len(detected_face) < 1:
        continue

    for face_rect in detected_face:
        x,y,h,w = face_rect
        face_img = img_gray[y:y+h, x:x+w]

        res, confidence = face_recognizer.predict(face_img)

        cv2.rectangle(tester, (x,y), (x+w, y+h), (255,0,0), 1)
        text = players[res] + ' : ' + str(confidence)
        cv2.putText(tester, text, (x,y-10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,255,0), 2)
        cv2.imshow('Result', tester)
        cv2.waitKey(0)

# for i, name in enumerate(players):
#     player_path = train_path + name + '/'
#     for stockimg in os.listdir(player_path):
#         path_per_pic = player_path + stockimg
#         img = cv2.imread(path_per_pic, 0)

#         detected_face = face_cascade.detectMultiScale(img, scaleFactor = 1.2, minNeighbors = 5)

#         if len(detected_face) < 1:
#             continue

#         for face_rect in detected_face:
#             x,y,h,w = face_rect
#             face_img = img[y:y+h, x:x+w]

#             face_list.append(face_img)
#             class_list.append(i)
