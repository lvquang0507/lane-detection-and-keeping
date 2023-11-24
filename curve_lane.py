import cv2 as cv
import numpy as np


def detect(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    blurred = cv.GaussianBlur(gray, ksize=(5, 5), sigmaX=0, sigmaY=0)
    _, bin_img = cv.threshold(blurred, 120, 255, cv.THRESH_BINARY)

    canny = cv.Canny(bin_img, 120, 210)
    warp = perspective_warp(
        gray, np.float32([(0.43, 0.65), (0.58, 0.65), (0.1, 1), (1, 1)])
    )
    cv.imshow("Warp", warp)


def roi(image, region):
    pass


def perspective_warp(image, src, dst=np.float32([(0, 0), (1, 0), (0, 1), (1, 1)])):
    height, width, *_ = image.shape

    img_size = np.float32([height, width])

    src = src * img_size
    dst = dst * img_size

    transform_mat = cv.getPerspectiveTransform(src, dst)

    warped = cv.warpPerspective(
        image, transform_mat, (width, height), flags=cv.INTER_LANCZOS4
    )
    return warped
