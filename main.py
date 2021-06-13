
#Importing all the modules
import cv2
import numpy as np
from numpy.core.fromnumeric import ptp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os
import ssl
import time
print("Imported")
X = np.load('image.npz')
X = X['arr_0']
y = pd.read_csv("labels.csv")
y = y["labels"]

print("loded")


classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
           "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
n_classes = len(classes)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=9, train_size=7500, test_size=2500)

scaled_X_train = X_train/255.0
scaled_X_test = X_test/255.0
print("trained")

clf = LogisticRegression(solver='saga', multi_class='multinomial').fit(
    scaled_X_train, y_train)


y_predicted = clf.predict(scaled_X_test)
accuracy = accuracy_score(y_test, y_predicted)
print("The accuracy of the model is :- ", accuracy)

cap = cv2.VideoCapture(0)
print("cam started")
while(True):
  print("Inside while loop")
  try:
    ret, frame = cap.read()
    print("Read Frame")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print("converted to gray")

    height, width = gray.shape
    upper_left = (int(width / 2 - 60), int(height / 2 - 60))
    bottom_right = (int(width / 2 + 60), int(height / 2 + 60))
    cv2.rectangle(gray, upper_left, bottom_right, (0, 255, 0), 2)
    print("rect drawn")

    roi = gray[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]

    im_pil = Image.fromarray(roi)
    print("Array formed")

    image_bw = im_pil.convert('L')
    image_bw_resized = image_bw.resize((28, 28), Image.ANTIALIAS)
    print("resized")
    image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized)
    pixel_filter = 20

    min_pixel = np.percentile(image_bw_resized_inverted, pixel_filter)
    image_bw_resized_inverted_scaled = np.clip(
        image_bw_resized_inverted-min_pixel, 0, 255)
    max_pixel = np.max(image_bw_resized_inverted)
    print("On line 87")


    image_bw_resized_inverted_scaled = np.asarray(
        image_bw_resized_inverted_scaled)/max_pixel
    test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1, 784)
    print("array made")
    test_pred = clf.predict(test_sample)
    print("classifier made")
    print("Predicted alphabet is: ", test_pred)
    print("pred done")


    cv2.imshow('frame', gray)
    print("Frame show")
    if cv2.waitKey(1) & 0xFF == ord('q'):
          break
  except Exception as e:
    print(e)


cap.release()
cv2.destroyAllWindows()
