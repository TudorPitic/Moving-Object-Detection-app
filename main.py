import numpy as np
import cv2
import math

def background_subtraction(cap):
    fps = cap.get(cv2.CAP_PROP_FPS)
    time = math.ceil(1000/fps)
    print(time)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    foreground_background1 = cv2.createBackgroundSubtractorKNN()

    while True:
        ret, frame = cap.read()
        if not ret:
            print('End')
            break
        frame1 = cv2.GaussianBlur(frame,(5,5),10)
        foreground_mask1 = foreground_background1.apply(frame1)
        foreground_mask1 = cv2.morphologyEx(foreground_mask1, cv2.MORPH_OPEN, kernel)
        ret, th1 = cv2.threshold(foreground_mask1, 200, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(th1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        foreground_mask1 = cv2.cvtColor(foreground_mask1, cv2.COLOR_GRAY2RGB)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 400 and area < 40000:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(foreground_mask1, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        output1 = frame
        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == 32:  # pause with space
            cv2.putText(output1,'PAUSED', (10,40), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255), 1, cv2.LINE_AA)
            cv2.imshow('KNN', output1)
            cv2.waitKey()

        cv2.imshow('KNN', output1)

    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    # cap = cv2.VideoCapture('videos/walking.avi')
    # cap = cv2.VideoCapture('videos/cars.mp4')
    # cap = cv2.VideoCapture('videos/day_2.avi')
    cap = cv2.VideoCapture('videos/football.mp4')
    # cap = cv2.VideoCapture(0)
    background_subtraction(cap)
