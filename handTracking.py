import cv2
import mediapipe as mp  # module to perform computer vision over data such as video or audio
import time


class handDetection():
    def __init__(self, mode=False, maxHands=2, detectionConfidence=0.5, trackConfidence=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConfidence = detectionConfidence
        self.trackConfidence = trackConfidence

        self.Hands = mp.solutions.hands
        self.hands = self.Hands.Hands(self.mode, self.maxHands, self.detectionConfidence,
                                    self.trackConfidence)  # creates hands object
        self.Draw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # object class only uses rgb images
        self.results = self.hands.process(rgb)  # method in class that will process frames for us and give us results
        # print(results.multi_hand_landmarks) #multi_hand_landmark checks when something is detected or not

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:  # handLms is a single hand
                if draw:
                    self.Draw.draw_landmarks(img, handLms,
                                            self.Hands.HAND_CONNECTIONS)  # connections draws lines between hands (all 21 landmarks)
        return img

    def find_position(self, img, handNu=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNu]

            for id, landmark in enumerate(myHand.landmark):
                height, width, channels = img.shape
                cx, cy = int(landmark.x * width), int(landmark.y * height)
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 20, (255, 0, 0), cv2.FILLED)
        return lmList


def main():
    prevTime = 0
    currTime = 0
    cap = cv2.VideoCapture(0)  # creates video object
    detect = handDetection()

    while True:
        success, img = cap.read()
        img = detect.find_hands(img)
        lmList = detect.find_position(img)
        if len(lmList) != 0:
            print(lmList[4])
        currTime = time.time()  # gives current time
        fps = 1 / (currTime - prevTime)
        prevTime = currTime
        cv2.putText(img, str(int(fps)), (40, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


