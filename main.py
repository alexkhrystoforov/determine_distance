import numpy as np
import cv2
import glob


# Load previously saved data
with np.load('data.npz') as X:
    mtx, dist = [X[i] for i in ('name1', 'name2')]


def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6 * 9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)

input_video = 'IMG_6506.MOV'
cap = cv2.VideoCapture(input_video)

frame_count = 0
key = None
ret, frame = cap.read()

frame_shape = frame.shape
frame_height = frame_shape[0]
frame_width = frame_shape[1]

fourcc = cv2.VideoWriter_fourcc(*'MJPG')

output_video = cv2.VideoWriter('output_video_white.avi', fourcc, 20.0, (frame_width, frame_height))


while ret:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # Find the rotation and translation vectors.
    _, rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)
    new_tvecs = np.linalg.norm(tvecs) * 2.2

    # project 3D points to image plane
    imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
    frame = draw(frame, corners2, imgpts)

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 50)
    fontScale = 1
    fontColor = (255, 0, 0)
    lineType = 2

    cv2.putText(frame, f'distance is : {str(new_tvecs)[:6]} cm',
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)

    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.imshow('img', frame)

    output_video.write(frame)

    # Pause on pressing of space.

    if key == ord(' '):
        wait_period = 0
    else:
        wait_period = 30

    # drawing, waiting, getting key, reading another frame

    key = cv2.waitKey(wait_period)
    ret, frame = cap.read()
    frame_count += 1

cap.release()
cv2.destroyAllWindows()
