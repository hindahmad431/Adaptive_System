from pathlib import Path
import glob
import cv2 as cv
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve 
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


# Read image from your local file system
original_image = cv.imread(r"C:\Users\hinda\Documents\FGDataset\020A24.jpg")

# Convert color image to grayscale for Viola-Jones
grayscale_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)

#change this path to your local path of haarcascade_frontalface_alt.xml 
face_cascade = cv.CascadeClassifier(r'C:\Users\hinda\Anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_alt.xml')

#run the classfier on multiScale to Deal with variation in face size
detected_faces = face_cascade.detectMultiScale(grayscale_image) 

for (column, row, width, height) in detected_faces:
    cv.rectangle(
        original_image,
        (column, row), #top_left
        (column + width, row + height),#bottom_right
        (0, 255, 0), #rectangle colour
        2 # rectangle thikness
    )

cv.imshow('DetectedFace', original_image)
cv.waitKey(0)
cv.destroyAllWindows()

#This Code credits goes to  Kristijan Ivancic, Traditional Face Detection With Python, https://realpython.com/traditional-face-detection-python/, Accessed May 2020