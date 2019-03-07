import numpy as np
import cv2 as cv
import math
from scipy import ndimage



img_before = cv.imread('us2.jpg')

cv.imshow("Before", img_before)    
key = cv.waitKey(0)

img_gray = cv.cvtColor(img_before, cv.COLOR_BGR2GRAY)
img_edges = cv.Canny(img_gray, 100, 100, apertureSize=3)
lines = cv.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)

angles = []

for x1, y1, x2, y2 in lines[0]:
    cv.line(img_before, (x1, y1), (x2, y2), (255, 0, 0), 3)
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    angles.append(angle)

median_angle = np.median(angles)
img = ndimage.rotate(img_before, median_angle)

xmax=0
ymax=0
dimensions = img.shape
height = img.shape[0]
width = img.shape[1] 
xmin= width
ymin = height



face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x,y,w,h) in faces:
        
    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    #new_image = img[y:y+h, x:x+w]
    if y<ymin:
        ymin=y
    if y+h>ymax:
        ymax=y+h
    if x<xmin:
        xmin=x
    if x+w>xmax:
        xmax=x+w
        
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        
try:        
    new_image = img[ymin:ymax, xmin:xmax]    
    cv.imshow('img',new_image)

except:
    print("Face could not be recognized") 
cv.waitKey(0)
cv.destroyAllWindows()





