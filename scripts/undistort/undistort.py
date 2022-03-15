# Modified from https://opencv24-python-tutorials.readthedocs.io/en/stable/py_tutorials/py_calib3d/py_calibration/py_calibration.html
import argparse
import numpy as np
import cv2
import os
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Undistort")
    parser.add_argument("--images_in", default="images", help="input path to the images")
    parser.add_argument("--images_out", default="images", help="output path to the images")
    parser.add_argument("--chessboards", default="chessboards", help="input path to the chessboards")
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.images_out, exist_ok=True)

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    h_chessboard = -1
    w_chessboard = -1
    tot_success= 0

    for i in os.listdir(args.chessboards):
        fname = os.path.join(args.chessboards, i)
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        h_chessboard, w_chessboard = gray.shape[:2]

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (7,6), None)

        # If found, add object points, image points (after refining them)
        if ret:
            tot_success += 1
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
            cv2.imshow('img',img)
            cv2.waitKey(500)

    print(tot_success)
    cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w_chessboard, h_chessboard), None, None)
    print(mtx)
    print(dist)

    # Undistort
    h = -1
    w = -1
    for i in tqdm(os.listdir(args.images_in)):
        fname = os.path.join(args.images_in, i)
        img = cv2.imread(fname)
        if h == -1 and w == -1:
            # Same resolution inputs
            h, w = img.shape[:2]
            new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w,h))

        dst = cv2.undistort(img, mtx, dist, None, new_mtx)

        # crop the image
        x,y,w,h = roi
        dst = dst[y:y+h, x:x+w]
        cv2.imwrite(os.path.join(args.images_out, i), dst)

    # Reconstruction error
    tot_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        tot_error += error

    print ("Total error: {}".format(tot_error / len(objpoints)))
