import cv2 
import numpy as np
import math
import pyautogui as p 
import time

# Read the camera
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

def nothing(x):
    pass
cv2.namedWindow("Hand Detection" ,cv2.WINDOW_NORMAL)
cv2.resizeWindow("Hand Detection",(300,300))
cv2.createTrackbar("Thresh","Hand Detection",0,255,nothing)

cv2.createTrackbar("Lower_H","Hand Detection",0,255,nothing)
cv2.createTrackbar("Lower_S","Hand Detection",0,255,nothing)
cv2.createTrackbar("Lower_V","Hand Detection",0,255,nothing)
cv2.createTrackbar("Upper_H","Hand Detection",255,255,nothing)
cv2.createTrackbar("Upper_S","Hand Detection",255,255,nothing)
cv2.createTrackbar("Upper_V","Hand Detection",255,255,nothing)



while True:
    _, frame = cap.read()
    frame = cv2.flip(frame,2)
    frame = cv2.resize(frame,(600,500))

    #Hand recognition
    cv2.rectangle (frame,(0,1),(300,500),(255,0,0),0)
    crop_image = frame[1:500, 0:300]

    hsv = cv2.cvtColor(crop_image,cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("Lower_H","Hand Detection")
    l_s = cv2.getTrackbarPos("Lower_S","Hand Detection")
    l_v = cv2.getTrackbarPos("Lower_V","Hand Detection")
    u_h = cv2.getTrackbarPos("Upper_H","Hand Detection")
    u_s = cv2.getTrackbarPos("Upper_S","Hand Detection")
    u_v = cv2.getTrackbarPos("Upper_V","Hand Detection")

    lower_bound = np.array([l_h, l_s, l_v])
    upper_bound = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    filter = cv2.bitwise_and(crop_image,crop_image,mask=mask)

    mask1 = cv2.bitwise_not(mask)
    m_g = cv2.getTrackbarPos("Thresh","Hand Detection")
    ret,thresh =cv2.threshold(mask1,m_g,255,cv2.THRESH_BINARY)
    dilate = cv2.dilate(thresh,(3,3),iterations = 6)

    cnts, hier = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


    try : 
        cm = max(cnts,key=lambda x: cv2.contourArea(x))
        epsilon = 0.0005*cv2.arcLength(cm,True)
        data = cv2.approxPolyDP(cm,epsilon,True)

        hull = cv2.convexHull(cm)

        cv2.drawContours(crop_image,[cm],-1,(50,50,150),2)
        cv2.drawContours(crop_image, [hull],-1,(0,255,0),2)

        hull = cv2.convexHull(cm,returnPoints=False)
        defects = cv2.convexityDefects(cm,hull)
        count_defects = 0 

        for i in range(defects.shape[0]):
            s, e, f, d= defects[i,0]

            start = tuple(cm[s][0])
            end = tuple(cm[e][0])
            far = tuple(cm[f][0])

            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1]- start[1] **2))
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]**2))
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1])**2)
            angle = (math.acos((b**2 + c**2 - a**2)/(2*b*c))*180)

            if angle <= 50:
                count_defects += 1

        if count_defects == 0 :
            cv2.putText(frame," ",(50,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,2))

        elif count_defects ==1:

            p.press("space")
            cv2.putText(frame,"Play/Pause",(50,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,2))
        elif count_defects == 2:
            p.press("UP")

            cv2.putText(frame,"Volume UP",(5,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,2)) 

        elif count_defects ==3 :
            p.press("Down")
            cv2.putText(frame,"Volume Down",(50,50),cv2.FONT_HERSHEY_SIMPLEX,(0,0,2)) 

        elif count_defects == 4:
            p.press("right")
            cv2.putText(frame, "Forward",(50,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,2))

    except:

        pass
    cv2.imshow("Thresh",thresh)
    cv2.imshow("filter==",filter)
    cv2.imshow("Result",frame)

    key = cv2.waitKey(25) &0xFF
    if key == 27:
        break
    
cap.release()
cv2.destroyAllWindows()