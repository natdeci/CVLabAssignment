import os
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

all_data = []
all_class = []
face_list = []
class_list = []
predict_list = []

def menuprint():
    print("Football Player Face Recognition")
    print("1. Train and Test Model")
    print("2. Predict")
    print("3. Exit")

def menu1():
    all_data = []
    all_class = []
    face_list = []
    class_list = []
    predict_list = []

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

    print("Training and Testing")
    for face_data, name_data in list(zip(X_train,y_train)):
        gray_face_data = cv2.cvtColor(face_data, cv2.COLOR_BGR2GRAY)
        detected_face = face_cascade.detectMultiScale(gray_face_data, scaleFactor = 1.2, minNeighbors = 5)
        if len(detected_face) < 1:
            continue
        
        for face_rect in detected_face:
            x,y,h,w = face_rect
            face_img = gray_face_data[y:y+h, x:x+w]

            face_list.append(face_img)
            class_list.append(name_data)
    
    face_recognizer.train(face_list, np.array(class_list))

    for tester in X_test:
        img_gray = cv2.cvtColor(tester, cv2.COLOR_BGR2GRAY)
        detected_face = face_cascade.detectMultiScale(img_gray, scaleFactor = 1.2, minNeighbors = 5)
        
        if len(detected_face) < 1:
            predict_list.append(10)
            continue

        # for face_rect in detected_face:
        #     x,y,h,w = face_rect
        #     face_img = img_gray[y:y+h, x:x+w]

        #     res, confidence = face_recognizer.predict(face_img)
        #     predict_list.append(res)
        else:
            x,y,h,w = detected_face[0]
            face_img = img_gray[y:y+h, x:x+w]

            res, confidence = face_recognizer.predict(face_img)
            predict_list.append(res)
    print("Training and Testing Finished")
    accuracy = accuracy_score(y_test, predict_list)
    print("Average Accuracy", accuracy)

def menu2():
    players = os.listdir("./Dataset/")
    predict_path = input("Input absolute path for image to predict >> ")
    print("85")
    predict_pict = cv2.imread(predict_path)
    print("87")
    predict_gray = cv2.cvtColor(predict_pict, cv2.COLOR_BGR2GRAY)
    print("89")
    detected_face = face_cascade.detectMultiScale(predict_gray, scaleFactor = 1.2, minNeighbors = 5)
    print("91")
    if len(detected_face) < 1:
        print("No Face Detected")
        return
    print("93")
    for face_rect in detected_face:
        x,y,h,w = face_rect
        print("98")
        face_img = predict_gray[y:y+h, x:x+w]
        print("100")
        res, confidence = face_recognizer.predict(face_img)
        print("102")
        cv2.rectangle(predict_pict, (x,y), (x+w, y+h), (255,0,0), 1)
        text = players[res] + ' : ' + str(confidence)
        cv2.putText(predict_pict, text, (x,y-10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,255,0), 2)
        cv2.imshow('Result', predict_pict)
        cv2.waitKey(0)

menum = 0
while menum != 3:
    menuprint()
    menum = int(input(">> "))
    if menum == 1:
        menu1()
    elif menum == 2:
        menu2()