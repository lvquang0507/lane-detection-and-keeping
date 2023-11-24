import cv2 as cv
import numpy as np


def lane_detect(frame):
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    k_size = 5
    blur_img = cv.GaussianBlur(gray_img, ksize=(k_size, k_size), sigmaX=0.6, sigmaY=0.5)
    _, threshold_img = cv.threshold(blur_img, 125, 255, cv.THRESH_BINARY)
    # adp_img = cv.adaptiveThreshold(
    #     blur_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 9, 2
    # )
    canny_img = cv.Canny(threshold_img, 120, 210)

    msk = roi(canny_img)
    lines = cv.HoughLinesP(msk, 2, np.pi / 180, 5, minLineLength=50, maxLineGap=20)

    lines_img = draw_line(gray_img, lines)
    cv.imshow("lines", lines_img)
    cv.imshow("raw", gray_img)


def roi(image):
    mask = np.zeros_like(image)

    height, width, *_ = image.shape
    bottom_left = [width * 0, height * 0.9]
    bottom_right = [width * 1, height * 0.9]
    top_left = [width * 0.1, height * 0.25]
    top_right = [width * 0.9, height * 0.25]

    vtx = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)

    fill = cv.fillPoly(mask, vtx, 255)
    cv.imshow("filled", fill)

    masked_img = cv.bitwise_and(image, mask)

    return masked_img


def draw_line(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv.line(image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    else:
        print("mt")
    return line_image
