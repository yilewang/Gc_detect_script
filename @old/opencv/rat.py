#!/bin/python

import cv2
import time
import numpy as np

pTime = 0
cTime = 0
cap = cv2.VideoCapture(r"C:\Users\Wayne\Downloads\rat\day_1\1-1.mp4")
object_detector = cv2.createBackgroundSubtractorMOG2()
while True:
    success, img = cap.read()
    roi = img[200:430, 60:600]
    mask = object_detector.apply(roi)
    _,mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contours = sorted(contours,key=lambda x:cv2.contourArea(x), reverse=True)
    for cnt in contours:
        area=cv2.contourArea(cnt)
        if area > 100:
            cv2.drawContours(roi, [cnt], -1, (0, 0, 255), 3)
            # x,y,w,h = cv2.boundingRect(cnt)
            # cv2.rectangle(roi, (x,y), (x+w, y+h), (0,255,0), 3)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3,(255, 0, 255), 3)
    cv2.imshow("Mask", mask)
    # cv2.imshow("Threshold", threshold)
    cv2.imshow("Image", roi)
    cv2.waitKey(1)

cv2.destroyAllWindows()