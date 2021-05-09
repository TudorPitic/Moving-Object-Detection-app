import numpy as np
import cv2

def background_subtraction(cap):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    foreground_background = cv2.createBackgroundSubtractorKNN()
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = False
    params.filterByInertia = False
    params.filterByConvexity = False
    params.minArea = 500
    print(params.filterByArea)
    print(params.filterByColor)
    print(params.filterByInertia)
    print(params.filterByConvexity)
    print(params.filterByCircularity)
    detector = cv2.SimpleBlobDetector_create(params)

    while True:
        ret, frame = cap.read()
        if not ret:
            print('End')
            break
        foreground_mask = foreground_background.apply(frame)
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)
        # foreground_mask = cv2.GaussianBlur(foreground_mask,(5,5),10)
        keypoints = detector.detect(foreground_mask)
        output = cv2.drawKeypoints(foreground_mask, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        cv2.imshow('Output', output)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

def optical_flow(cap):
    feature_params = dict(maxCorners = 100,
                          qualityLevel = 0.3,
                          minDistance = 7,
                          blockSize = 7)
    lucas_kanade_params = dict(winSize = (15, 15),
                               maxLevel = 2,
                               criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    #this creates a random color
    color = np.random.randint(0,255,(100,3))

    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    prev_corners = cv2.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)

    mask = np.zeros_like(prev_frame)

    while(True):
        ret, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        new_corners, status, errors = cv2.calcOpticalFlowPyrLK(prev_gray,
                                                               frame_gray,
                                                               prev_corners,
                                                               None,
                                                               **lucas_kanade_params)
        good_new = new_corners[status==1]
        good_old = prev_corners[status==1]

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a,b), (c,d), color[i].tolist(), 2)
            frame = cv2.circle(frame, (a,b), 5, color[i].tolist(), -1)

        img = cv2.add(frame, mask)

        cv2.imshow('Optical Flow - Lucas-Kanade', img)
        if cv2.waitKey(1) == 27:
            break
        prev_gray = frame_gray.copy()
        prev_corners = good_new.reshape(-1,1,2)
    cap.release()
    cv2.destroyAllWindows()

def dense_optical_flow(cap):
    ret, first_frame = cap.read()
    previous_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(first_frame)
    hsv[...,1] = 255

    while True:
        ret, frame2 = cap.read()
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        # Computes the dense optical flow using the Gunnar Farneback’s algorithm
        flow = cv2.calcOpticalFlowFarneback(previous_gray, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # use flow to calculate the magnitude (speed) and angle of motion
        # use these values to calculate the color to reflect speed and angle
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = angle * (180 / (np.pi / 2))
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        final = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Show our demo of Dense Optical Flow
        cv2.imshow('Dense Optical Flow', final)
        if cv2.waitKey(1) == 27:  # 27 is the Esc Key
            break

        # Store current image as previous image
        previous_gray = next
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    cap = cv2.VideoCapture('videos/walking.avi')
    # cap = cv2.VideoCapture(0)
    background_subtraction(cap)
    # optical_flow(cap)
    # dense_optical_flow(cap)