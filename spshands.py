import mediapipe as mp
import cv2
import time
import math


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        return img

    def findPosition(self, img, handNo = 0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id , lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                xList.append(cx)
                yList.append(cy)
            xmin, xmax= min(xList), max(xList)
            ymin, ymax= min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            # if draw:
            #     cv2.rectangle(img, (bbox[0]-20, bbox[1]-20),
            #                   (bbox[2]+20, bbox[3]+20), (0, 255, 0), 2)

        return bbox


def main():
    detector=handDetector()
    while True:
        img = detector.findHands(img)
        detector.findPosition(img)
        cv2.waitKey(1)




if __name__=="__main__":
    main()
