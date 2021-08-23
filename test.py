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

X = np.load('image.npz')
X = X['arr_0'][:, :28]
y = pd.read_csv("labels.csv")
y = y["labels"]

print(pd.Series(y).value_counts())

classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
           "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
n_classes = len(classes)

print(n_classes)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=9, train_size=3500, test_size=500)

scaled_X_train = X_train/255.0
scaled_X_test = X_test/255.0

clf = LogisticRegression(solver='saga', multi_class='multinomial').fit(
    scaled_X_train, y_train)


y_predicted = clf.predict(scaled_X_test)
accuracy = accuracy_score(y_test, y_predicted)
print("The accuracy of the model is :- ", accuracy)

cap = cv2.VideoCapture(0,  cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(img)
    image_bw = image_pil.convert('L')
    image_bw_resized = image_bw.resize((28, 28), Image.ANTIALIAS)
    pixel_filter = 20
    min_pixel = np.percentile(image_bw_resized, pixel_filter)
    image_bw_resized_inverted_scaled = np.clip(image_bw_resized-min_pixel, 0, 225)
    max_pixel = np.max(image_bw_resized)
    image_bw_resized_inverted_scaled
    test_image = np.asarray(image_bw_resized_inverted_scaled)/max_pixel
    test_sample = np.array(test_image)
    test_pred = clf.predict(test_sample)
    print(test_pred[0])
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
