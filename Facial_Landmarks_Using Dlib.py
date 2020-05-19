#!/usr/bin/env python
# coding: utf-8

# In[18]:


import dlib
import face_recognition
import cv2


# In[24]:


img = cv2.imread(r"C:\Users\hinda\Documents\FG\035A18.jpg", 1)
face_landmarks_list = face_recognition.face_landmarks(img)


# In[27]:


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# In[33]:


faces_in_image = detector(img_gray, 0)


# In[ ]:


# loop through each face in image
for face in faces_in_image:

# assign the facial landmarks
    landmarks = predictor(img_gray, face)

# unpack the 68 landmark coordinates from the dlib object into a list 
    landmarks_list = []
    for i in range(0, landmarks.num_parts):
        landmarks_list.append((landmarks.part(i).x, landmarks.part(i).y))

# for each landmark, plot and write number
    for landmark_num, xy in enumerate(landmarks_list, start = 1):
        cv2.circle(img, (xy[0], xy[1]), 12, (168, 0, 20), -1)
        cv2.putText(img, str(landmark_num),(xy[0]-7,xy[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(255,255,255), 1)


# visualise the image with landmarks
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


#all the coding credits goes to Andrew Jones, Mapping Facial Landmarks in Python using OpenCV, 2019

