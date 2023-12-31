import cv2
import time
import numpy as np
import handTracking as ht
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


camWidth, camHeight = 500, 420

# check if webcam is working
cap = cv2.VideoCapture(0)
cap.set(3, camWidth)
cap.set(4, camHeight)
pTime = 0
detect = ht.handDetection(detectionCon = 0.7)

device = AudioUtilities.GetSpeakers()
interface = device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volumeRange = volume.GetVolumeRange()
minVolume = volumeRange[0]
maxVolume = volumeRange[1]
volume = 0
volumeBar = 400
volumePer = 0

while True:
    success, img = cap.read()
    img = detect.findHands(img)
    lmList = detect.findPosition(img, draw=False)
    if len(lmList)!= 0:
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(img, (x1, y1), 15, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.circle(img, (cx, cy), 15, (255, 0, 0), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)
        
        volume = np.interp(length, [50, 300], [minVolume, maxVolume])
        volumeBar = np.interp(length, [50, 300], [400, 150])
        volumePer = np.interp(length, [50, 300], [0, 100])
        print(int(length), volume)
        volume.SetMasterVolumeLevel(volume, None)
        
        if length < 50:
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
    cv2.rectangle(img, (50, int(volumeBar)), (85, 400), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, f'{int(volumePer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 0, 0), 3)
    
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 0, 0), 3)


        
        
        
    cv2.imshow("Img", img)
    cv2.waitKey(1)
