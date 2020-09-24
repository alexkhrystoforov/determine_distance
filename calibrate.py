import cv2
import numpy as np
import glob
import os


def get_matrix(folder_name, patternSize):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    # вместо этих координат задать размеры

    objp = np.zeros((patternSize[1] * patternSize[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:patternSize[0], 0:patternSize[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = glob.glob(folder_name + '/*.png')
    gray = None
    i = 1
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Find the chess board corners
        print(i)
        ret, corners = cv2.findChessboardCorners(gray, patternSize)


        if ret:
            i += 1
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (9, 6), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, patternSize, corners2, ret)
            cv2.namedWindow('img', cv2.WINDOW_NORMAL)
            cv2.imshow('img', img)
            cv2.namedWindow('binary', cv2.WINDOW_NORMAL)
            cv2.imshow('binary', gray)
            cv2.waitKey(0)

    cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print(mtx)

    np.savez('data.npz', name1=mtx, name2=dist)


if __name__ == '__main__':
    get_matrix(folder_name='frames/', patternSize=(9, 6))
