#!/bin/python

import cv2
import mediapipe as mp
import time


# import cv2
# cap = cv2.VideoCapture()
# # The device number might be 0 or 1 depending on the device and the webcam
# cap.open(0, cv2.CAP_DSHOW)
# while(True):
#     ret, frame = cap.read()
#     cv2.imshow('frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()

class handDetector():
    def __init__(self, mode=False, maxHands = 2, complexity = 1, detectionCon=0.5, trackCon = 0.5 ):
        self.mode = mode
        self.maxHands = maxHands
        self.complexity = complexity
        self.detectionCon= detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.complexity, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils



    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        #print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks: # if we detect the hands
            for handLms in self.results.multi_hand_landmarks: # if multiple hands

                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img
    
    def findPosition(self, img, handNo=0, draw=True): # the tractory of handNode
        lmList = []
        if self.results.multi_hand_landmarks: # if we detect the hands
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark): # calculate the x y axis
                #print(id, lm)
                h,w,c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h) # the center of our hand
                lmList.append([id, cx, cy])
                #print(id, cx, cy)
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 0), cv2.FILLED)
        return lmList


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3,(255, 0, 255), 3)


        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
