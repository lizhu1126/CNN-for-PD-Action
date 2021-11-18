import cv2
import mediapipe as mp
import math
from os import listdir

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.65,
        min_tracking_confidence=0.65)

# video path
cap = cv2.VideoCapture('D:/xxx.mp4')

# txt for Storing bone joint points
file_hand = open('D:xxx.txt', mode='w')

while True:
    ret, frame = cap.read()
    if not ret:
        #file_hand.close()
        break
    i = 1  # left or right hand
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if results.multi_handedness:
        for hand_label in results.multi_handedness:
            # print(hand_label)
             label = hand_label.classification[0].label
             print(label)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # print(hand_landmarks)
            h, w, c = frame.shape

            print(i)
            if i==1:
                x4=hand_landmarks.landmark[4].x*w
                y4=hand_landmarks.landmark[4].y*h
                z4=hand_landmarks.landmark[4].z*100
                x8=hand_landmarks.landmark[8].x*w
                y8=hand_landmarks.landmark[8].y*h
                z8=hand_landmarks.landmark[8].z*100
                print(i,x8,y8,z8)
                x=float('%.2f' %(x4-x8))
                y=float('%.2f' %(y4-y8))
                z=float('%.2f' %(z4-z8))
                dis = math.sqrt(x*x+y*y) # x*x+y*y+z*z
                dis='%.2f' % dis
                file_hand.write(str(dis)+'\n')
            i += 1
            draw = frame
            mp_drawing.draw_landmarks(draw, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    cv2.imshow('MediaPipe Hands', draw)

    if cv2.waitKey(1) & 0xFF == 27:
        file_hand.close()
        break

file_hand.close()
cap.release()
